//! NNUE inference: the same network as `bot/nnue.py` (`NumpyNNUE` is the reference).
//!
//! Board: 64 bytes in row-major order (row 0 = rank 8), 0 = empty, else
//! 1 + owner * 5 + piece type, with owners R B Y G X = 0..4 and types K B N R P = 0..4.
//!
//! Features: for perspective (absolute seat) p, a piece is feature
//! (relative owner * 5 + type) * 64 + rotated square, where the relative owner is
//! (owner - p) mod 4 (dead pieces: 4) and the square is rotated p quarter turns with
//! (r, c) -> (7 - c, r). Accumulators: acc[p] = ft_b + sum of ft_w rows of active features.
//!
//! Forward pass for the mover k: x = [clip(acc[k]), clip(acc[k+1]), clip(acc[k+2]),
//! clip(acc[k+3]), globals]; each hidden layer is y = clip(W x + b) (+ x if residual);
//! the output layer gives the 6 pairwise logits.
//!
//! Large networks are limited by memory bandwidth: one forward pass reads every weight once
//! (255 MB for the "xl" preset). `forward_batch` evaluates many positions per pass (each
//! weight row is reused for the whole batch) and splits the rows across threads (rayon), so
//! a search should evaluate sibling positions together (`AccStack::child_logits`).

use rayon::prelude::*;

pub const NUM_FEATURES: usize = 1600;
/// Multiply-adds below which a layer runs on one thread (thread start-up costs more).
const PAR_MIN_MACS: usize = 1 << 21;
pub const NUM_GLOBALS: usize = 16;
pub const NUM_OUTPUTS: usize = 6;
const NUM_CODES: usize = 25;

#[derive(Clone, Debug)]
pub struct Layer {
    pub w: Vec<f32>, // (n_out, n_in), row-major
    pub b: Vec<f32>,
    pub n_in: usize,
    pub n_out: usize,
    pub residual: bool,
}

impl Layer {
    pub fn new(w: Vec<f32>, b: Vec<f32>, n_in: usize, n_out: usize, residual: bool) -> Result<Self, String> {
        if w.len() != n_in * n_out || b.len() != n_out {
            return Err(format!("layer shape mismatch: w {} for {}x{}, b {}", w.len(), n_out, n_in, b.len()));
        }
        if residual && n_in != n_out {
            return Err(format!("residual layer must be square, got {}x{}", n_out, n_in));
        }
        Ok(Layer { w, b, n_in, n_out, residual })
    }
}

#[derive(Debug)]
pub struct Net {
    pub h: usize,
    pub ft_w: Vec<f32>, // (1600, h), row-major
    pub ft_b: Vec<f32>,
    pub layers: Vec<Layer>,
    pub out: Layer,
    feature: Vec<u16>, // [p][code - 1][square]
    max_width: usize,
    /// Points head (nn/train.py): the output layer gives each seat's expected future points
    /// F (mover first) and logit(i above j) = beta * ((s_i + F_i) - (s_j + F_j)), with the
    /// current scores s from the globals. None: the output layer gives the 6 logits.
    pub points_beta: Option<f32>,
    /// Global inputs after the accumulators: the 16 of `Position::globals`, or more
    /// (`features::extended`), read off the first layer's width.
    pub n_globals: usize,
}

/// Dot product written as 8 independent lanes so it compiles to SIMD (AVX2 + FMA).
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut lanes = [0.0f32; 8];
    let ca = a.chunks_exact(8);
    let cb = b.chunks_exact(8);
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for i in 0..8 {
            lanes[i] = x[i].mul_add(y[i], lanes[i]);
        }
    }
    let mut s = (lanes[0] + lanes[4]) + (lanes[1] + lanes[5]) + (lanes[2] + lanes[6]) + (lanes[3] + lanes[7]);
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

#[inline]
fn add_row(acc: &mut [f32], row: &[f32]) {
    for (a, r) in acc.iter_mut().zip(row) {
        *a += r;
    }
}

#[inline]
fn sub_row(acc: &mut [f32], row: &[f32]) {
    for (a, r) in acc.iter_mut().zip(row) {
        *a -= r;
    }
}

pub fn feature_index(p: usize, code: u8, square: usize) -> usize {
    let owner = (code as usize - 1) / 5;
    let ty = (code as usize - 1) % 5;
    let rel = if owner == 4 { 4 } else { (owner + 4 - p) % 4 };
    let (mut r, mut c) = (square / 8, square % 8);
    for _ in 0..p {
        let nr = 7 - c;
        c = r;
        r = nr;
    }
    (rel * 5 + ty) * 64 + r * 8 + c
}

impl Net {
    pub fn new(h: usize, ft_w: Vec<f32>, ft_b: Vec<f32>, layers: Vec<Layer>, out: Layer) -> Result<Self, String> {
        if ft_w.len() != NUM_FEATURES * h || ft_b.len() != h {
            return Err(format!("feature transformer must be ({}, {}), got {} / {}", NUM_FEATURES, h, ft_w.len(), ft_b.len()));
        }
        let n_globals = layers.first().map_or(out.n_in, |l| l.n_in).saturating_sub(4 * h).max(NUM_GLOBALS);
        let mut width = 4 * h + n_globals;
        let mut max_width = width;
        for (i, l) in layers.iter().enumerate() {
            if l.n_in != width {
                return Err(format!("layer {} expects {} inputs, previous width is {}", i + 1, l.n_in, width));
            }
            width = l.n_out;
            max_width = max_width.max(width);
        }
        if out.n_in != width || (out.n_out != NUM_OUTPUTS && out.n_out != 4) {
            return Err(format!("output layer must be ({}, {}), got ({}, {})", NUM_OUTPUTS, width, out.n_out, out.n_in));
        }
        let mut feature = vec![0u16; 4 * NUM_CODES * 64];
        for p in 0..4 {
            for code in 1..=NUM_CODES as u8 {
                for sq in 0..64 {
                    feature[(p * NUM_CODES + code as usize - 1) * 64 + sq] = feature_index(p, code, sq) as u16;
                }
            }
        }
        Ok(Net { h, ft_w, ft_b, layers, out, feature, max_width, points_beta: None, n_globals })
    }

    /// The 6 pairwise logits from the output layer's values (see `points_beta`).
    #[inline]
    fn finish(&self, raw: &[f32], globals: &[f32]) -> [f32; NUM_OUTPUTS] {
        let mut o = [0.0f32; NUM_OUTPUTS];
        match self.points_beta {
            None => o.copy_from_slice(&raw[..NUM_OUTPUTS]),
            Some(beta) => {
                let v: [f32; 4] = std::array::from_fn(|i| 30.0 * globals[4 + i] + raw[i]);
                const PAIRS: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
                for (n, &(i, j)) in PAIRS.iter().enumerate() {
                    o[n] = beta * (v[i] - v[j]);
                }
            }
        }
        o
    }

    pub fn num_params(&self) -> usize {
        self.ft_w.len() + self.ft_b.len()
            + self.layers.iter().map(|l| l.w.len() + l.b.len()).sum::<usize>()
            + self.out.w.len() + self.out.b.len()
    }

    #[inline]
    fn row(&self, p: usize, code: u8, sq: usize) -> &[f32] {
        let f = self.feature[(p * NUM_CODES + code as usize - 1) * 64 + sq] as usize;
        &self.ft_w[f * self.h..(f + 1) * self.h]
    }

    fn check_board(board: &[u8; 64]) -> Result<(), String> {
        match board.iter().find(|&&c| c as usize > NUM_CODES) {
            Some(c) => Err(format!("invalid piece code {}", c)),
            None => Ok(()),
        }
    }

    /// The four accumulators (absolute seats, each h long) computed from scratch.
    pub fn refresh(&self, board: &[u8; 64], acc: &mut [f32]) -> Result<(), String> {
        Self::check_board(board)?;
        let h = self.h;
        for p in 0..4 {
            let a = &mut acc[p * h..(p + 1) * h];
            a.copy_from_slice(&self.ft_b);
            for (sq, &code) in board.iter().enumerate() {
                if code != 0 {
                    add_row(a, self.row(p, code, sq));
                }
            }
        }
        Ok(())
    }

    /// Update the accumulators in place from `old` to `new` (only changed squares).
    pub fn apply_diff(&self, acc: &mut [f32], old: &[u8; 64], new: &[u8; 64]) -> Result<(), String> {
        Self::check_board(new)?;
        let h = self.h;
        for sq in 0..64 {
            let (a, b) = (old[sq], new[sq]);
            if a == b {
                continue;
            }
            for p in 0..4 {
                let acc_p = &mut acc[p * h..(p + 1) * h];
                if a != 0 {
                    sub_row(acc_p, self.row(p, a, sq));
                }
                if b != 0 {
                    add_row(acc_p, self.row(p, b, sq));
                }
            }
        }
        Ok(())
    }

    /// 6 pairwise logits for mover k. `scratch` must hold 2 * max_width floats.
    pub fn forward(&self, acc: &[f32], k: usize, globals: &[f32], scratch: &mut Vec<f32>) -> [f32; NUM_OUTPUTS] {
        let h = self.h;
        scratch.resize(2 * self.max_width, 0.0);
        let (x, y) = scratch.split_at_mut(self.max_width);
        for i in 0..4 {
            let p = (k + i) % 4;
            for (dst, src) in x[i * h..(i + 1) * h].iter_mut().zip(&acc[p * h..(p + 1) * h]) {
                *dst = src.clamp(0.0, 1.0);
            }
        }
        let ng = self.n_globals;
        x[4 * h..4 * h + ng].copy_from_slice(&globals[..ng]);
        let (mut cur, mut nxt) = (x, y);
        for l in &self.layers {
            let input = &cur[..l.n_in];
            for o in 0..l.n_out {
                let v = (dot(&l.w[o * l.n_in..(o + 1) * l.n_in], input) + l.b[o]).clamp(0.0, 1.0);
                nxt[o] = if l.residual { v + input[o] } else { v };
            }
            std::mem::swap(&mut cur, &mut nxt);
        }
        let mut out = [0.0f32; NUM_OUTPUTS];
        let input = &cur[..self.out.n_in];
        for (o, v) in out.iter_mut().enumerate().take(self.out.n_out) {
            *v = dot(&self.out.w[o * self.out.n_in..(o + 1) * self.out.n_in], input) + self.out.b[o];
        }
        self.finish(&out, globals)
    }

    /// Multiply-adds of one forward pass (hidden + output layers).
    pub fn body_macs(&self) -> usize {
        self.layers.iter().map(|l| l.w.len()).sum::<usize>() + self.out.w.len()
    }

    /// 6 pairwise logits for each of `ks.len()` positions. `accs` holds their accumulators
    /// back to back (4 * h each), `globals` their 16 globals each.
    pub fn forward_batch(&self, accs: &[f32], ks: &[usize], globals: &[f32]) -> Vec<[f32; NUM_OUTPUTS]> {
        let (h, n) = (self.h, ks.len());
        let ng = self.n_globals;
        let w0 = 4 * h + ng;
        let mut x = vec![0.0f32; n * w0];
        for (b, &k) in ks.iter().enumerate() {
            let acc = &accs[b * 4 * h..(b + 1) * 4 * h];
            let row = &mut x[b * w0..(b + 1) * w0];
            for i in 0..4 {
                let p = (k + i) % 4;
                for (dst, src) in row[i * h..(i + 1) * h].iter_mut().zip(&acc[p * h..(p + 1) * h]) {
                    *dst = src.clamp(0.0, 1.0);
                }
            }
            row[4 * h..].copy_from_slice(&globals[b * ng..(b + 1) * ng]);
        }
        let mut width = w0;
        for l in &self.layers {
            let yt = matmul_t(l, &x, n); // (n_out, n): row o = output o for every position
            let mut next = vec![0.0f32; n * l.n_out];
            for b in 0..n {
                for o in 0..l.n_out {
                    let v = (yt[o * n + b] + l.b[o]).clamp(0.0, 1.0);
                    next[b * l.n_out + o] = if l.residual { v + x[b * width + o] } else { v };
                }
            }
            x = next;
            width = l.n_out;
        }
        let yt = matmul_t(&self.out, &x, n);
        (0..n)
            .map(|b| {
                let mut o = [0.0f32; NUM_OUTPUTS];
                for (i, v) in o.iter_mut().enumerate().take(self.out.n_out) {
                    *v = yt[i * n + b] + self.out.b[i];
                }
                self.finish(&o, &globals[b * ng..(b + 1) * ng])
            })
            .collect()
    }
}

/// W x for a batch: x is (n, n_in) row-major; returns (n_out, n) without bias. Each weight
/// row is read once and used for all n positions; rows are split across threads when the
/// layer is big enough.
fn matmul_t(l: &Layer, x: &[f32], n: usize) -> Vec<f32> {
    let mut yt = vec![0.0f32; l.n_out * n];
    let row = |(o, out): (usize, &mut [f32])| {
        let w = &l.w[o * l.n_in..(o + 1) * l.n_in];
        for (b, v) in out.iter_mut().enumerate() {
            *v = dot(w, &x[b * l.n_in..(b + 1) * l.n_in]);
        }
    };
    if l.w.len() * n >= PAR_MIN_MACS {
        yt.par_chunks_mut(n).enumerate().with_min_len(8).for_each(row);
    } else {
        yt.chunks_mut(n).enumerate().for_each(row);
    }
    yt
}

/// A stack of accumulators for depth-first search: push a new board, pop back.
/// Buffers are reused, so a search allocates only when it reaches a new maximum depth.
pub struct AccStack {
    pub net: std::sync::Arc<Net>,
    accs: Vec<Vec<f32>>,
    boards: Vec<[u8; 64]>,
    depth: usize,
    scratch: Vec<f32>,
}

impl AccStack {
    pub fn new(net: std::sync::Arc<Net>, board: &[u8; 64]) -> Result<Self, String> {
        let mut acc = vec![0.0f32; 4 * net.h];
        net.refresh(board, &mut acc)?;
        Ok(AccStack { net, accs: vec![acc], boards: vec![*board], depth: 0, scratch: Vec::new() })
    }

    pub fn depth(&self) -> usize {
        self.depth
    }

    pub fn acc(&self) -> &[f32] {
        &self.accs[self.depth]
    }

    pub fn push(&mut self, board: &[u8; 64]) -> Result<(), String> {
        let next = self.depth + 1;
        if self.accs.len() <= next {
            self.accs.push(vec![0.0f32; 4 * self.net.h]);
            self.boards.push([0u8; 64]);
        }
        let (done, rest) = self.accs.split_at_mut(next);
        rest[0].copy_from_slice(&done[self.depth]);
        self.net.apply_diff(&mut rest[0], &self.boards[self.depth], board)?;
        self.boards[next] = *board;
        self.depth = next;
        Ok(())
    }

    pub fn pop(&mut self) -> Result<(), String> {
        if self.depth == 0 {
            return Err("pop from the root position".into());
        }
        self.depth -= 1;
        Ok(())
    }

    pub fn logits(&mut self, k: usize, globals: &[f32]) -> [f32; NUM_OUTPUTS] {
        let net = self.net.clone();
        if net.body_macs() >= PAR_MIN_MACS {
            return net.forward_batch(&self.accs[self.depth], &[k], globals)[0];
        }
        net.forward(&self.accs[self.depth], k, globals, &mut self.scratch)
    }

    /// Logits of several child positions of the current one in one batched pass: each child
    /// board's accumulators are the current ones plus that child's board difference.
    pub fn child_logits(&self, boards: &[[u8; 64]], ks: &[usize], globals: &[f32]) -> Result<Vec<[f32; NUM_OUTPUTS]>, String> {
        let h4 = 4 * self.net.h;
        let mut accs = vec![0.0f32; boards.len() * h4];
        for (i, b) in boards.iter().enumerate() {
            let a = &mut accs[i * h4..(i + 1) * h4];
            a.copy_from_slice(&self.accs[self.depth]);
            self.net.apply_diff(a, &self.boards[self.depth], b)?;
        }
        Ok(self.net.forward_batch(&accs, ks, globals))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn rng(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    }

    fn net(h: usize, hidden: &[usize], residual: bool) -> Net {
        let mut s = 7u64;
        let mut v = |n: usize| (0..n).map(|_| 0.2 * rng(&mut s)).collect::<Vec<f32>>();
        let ft_w = v(NUM_FEATURES * h);
        let ft_b = v(h);
        let mut width = 4 * h + NUM_GLOBALS;
        let mut layers = vec![];
        for &w in hidden {
            layers.push(Layer::new(v(w * width), v(w), width, w, residual && w == width).unwrap());
            width = w;
        }
        let out = Layer::new(v(NUM_OUTPUTS * width), v(NUM_OUTPUTS), width, NUM_OUTPUTS, false).unwrap();
        Net::new(h, ft_w, ft_b, layers, out).unwrap()
    }

    #[test]
    fn rotation_matches_python() {
        // Red's king on d1 (row 7, col 3) seen by Red is feature (0*5+0)*64 + 7*8 + 3.
        assert_eq!(feature_index(0, 1, 7 * 8 + 3), 59);
        // the same square seen by Blue (one quarter turn): (7, 3) -> (4, 7); Red is "previous" (3)
        assert_eq!(feature_index(1, 1, 7 * 8 + 3), (3 * 5) * 64 + 4 * 8 + 7);
        // dead pieces are owner 4 for everyone
        assert_eq!(feature_index(2, 21, 0) / 64, 4 * 5);
    }

    #[test]
    fn incremental_matches_refresh() {
        let n = Arc::new(net(16, &[24, 24, 24], true));
        let mut board = [0u8; 64];
        board[59] = 1;
        board[10] = 9;
        board[33] = 22;
        let mut st = AccStack::new(n.clone(), &board).unwrap();
        let mut b2 = board;
        b2[10] = 0;
        b2[18] = 9; // a move
        b2[33] = 0; // a capture elsewhere
        st.push(&b2).unwrap();
        let mut fresh = vec![0.0; 4 * 16];
        n.refresh(&b2, &mut fresh).unwrap();
        for (a, b) in st.acc().iter().zip(&fresh) {
            assert!((a - b).abs() < 1e-5);
        }
        st.pop().unwrap();
        n.refresh(&board, &mut fresh).unwrap();
        for (a, b) in st.acc().iter().zip(&fresh) {
            assert!((a - b).abs() < 1e-5);
        }
        assert!(st.pop().is_err());
        let g = [0.5f32; NUM_GLOBALS];
        let l = st.logits(2, &g);
        assert!(l.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn batch_matches_single() {
        let n = Arc::new(net(8, &[40, 40, 40, 40], true));
        let mut board = [0u8; 64];
        board[59] = 1;
        board[4] = 6;
        let st = AccStack::new(n.clone(), &board).unwrap();
        let mut kids = vec![];
        for sq in [12usize, 20, 28] {
            let mut b = board;
            b[4] = 0;
            b[sq] = 6;
            kids.push(b);
        }
        let ks = [1usize, 2, 3];
        let g: Vec<f32> = (0..3 * NUM_GLOBALS).map(|i| (i % 7) as f32 / 7.0).collect();
        let batch = st.child_logits(&kids, &ks, &g).unwrap();
        for (i, b) in kids.iter().enumerate() {
            let mut acc = vec![0.0; 4 * 8];
            n.refresh(b, &mut acc).unwrap();
            let mut scratch = vec![];
            let single = n.forward(&acc, ks[i], &g[i * NUM_GLOBALS..(i + 1) * NUM_GLOBALS], &mut scratch);
            for (x, y) in batch[i].iter().zip(&single) {
                assert!((x - y).abs() < 1e-4, "{} vs {}", x, y);
            }
        }
    }

    #[test]
    fn shape_errors_are_reported() {
        let l = Layer::new(vec![0.0; 10], vec![0.0; 2], 4, 2, false);
        assert!(l.is_err());
        assert!(Layer::new(vec![0.0; 8], vec![0.0; 2], 4, 2, true).is_err());
    }
}

//! NNUE-T: a threat-aware NNUE with a Bradley-Terry head (DESIGN.md section 3).
//!
//! This file is the single definition of the features: `bin/featurize.rs` writes them for
//! training (`train/train_nn2.py` mirrors the forward pass in PyTorch, and `nn2_check` compares
//! the two), and `Nn2::strengths` evaluates leaves in the search (as `Table::N2`).
//!
//! Features of viewing seat v (rel(o) = (o - v) mod 4 for a seat, 4 for dead pieces):
//!   piece  (rel(owner) * 5 + type) * 64 + rot_square(sq, v)                       0..1600
//!   threat 1600 + ((rel(owner) * 5 + type) * 4 + rel(attacker)) * 2 + defended     ..1760
//!          for every live piece and every active seat other than its owner that attacks it
//!   kzone  1760 + rel(king owner) * 4 + rel(attacker), value = attacked squares of the 3x3
//!          zone around the king (attackers: active seats other than the owner)          ..1776
//! Values are counts. Globals (28, mover-relative, seat i = (mover + i) % 4): active[4],
//! points/30 [4], (points - mean)/20 [4], repetitions/3, min(turn/400, 1), dead kings/3,
//! players left/4, tanh(points gap / 10) for the 6 pairs, one-hot absolute seat of the mover [4],
//! two players left, endgame flag (a live pawn within 2 squares of promotion).
//!
//! Net: acc_i = clamp(ft_b + sum cnt * ft_w[f], 0, 1) (A wide) for the 4 viewing seats in turn
//! order from the mover; x = [acc_0..acc_3, globals]; h1 = clamp(W1 x + b1, 0, 1);
//! h2 = clamp(W2 h1 + b2, 0, 1); S_i = (W3 h2 + b3)_i + sum cnt * psq[f] (seat i's features)
//! + pts_w * points(seat i). P(seat i above seat j) = sigmoid(S_i - S_j).
//!
//! File (`*.nn2`): a text line `nn2 1 A H1 H2 NF NG`, a newline, then little-endian f32:
//! ft_w [NF][A], ft_b [A], psq [NF], w1 [H1][4A+NG], b1 [H1], w2 [H2][H1], b2 [H2],
//! w3 [4][H2], b3 [4], pts_w [1].

use std::path::Path;

use crate::greedy::{cheapest_attacker, endgame_now};
use crate::pst::rot_square;
use crate::rules::{code, owner_of, type_of, Position, DEAD, EMPTY, KING};

pub const NF: usize = 1776;
pub const NG: usize = 28;
pub const MAXF: usize = 64; // features per viewing seat kept (piece features come first)
const PAIRS: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

/// What the features need from a position (a game record carries exactly this).
#[derive(Clone, Copy, Debug)]
pub struct Raw {
    pub board: [u8; 64],
    pub turn: u32,
    pub scores: [i32; 4],
    pub eliminated: u8,
    pub reps: u32,
}

impl Raw {
    pub fn of(pos: &Position) -> Raw {
        Raw { board: pos.board, turn: pos.turn, scores: pos.scores, eliminated: pos.eliminated, reps: pos.repetitions() }
    }
    pub fn mover(&self) -> usize {
        (self.turn % 4) as usize
    }
    fn active(&self, c: usize) -> bool {
        self.eliminated & (1 << c) == 0
    }
}

/// Sparse features of the 4 viewing seats (mover first) and the globals.
pub struct Feats {
    pub idx: [[u16; MAXF]; 4],
    pub cnt: [[u8; MAXF]; 4],
    pub n: [usize; 4],
    pub glob: [f32; NG],
    pub overflow: usize,
}

pub fn features(r: &Raw) -> Feats {
    let b = &r.board;
    let m = r.mover();
    let mut f = Feats { idx: [[0; MAXF]; 4], cnt: [[0; MAXF]; 4], n: [0; 4], glob: [0.0; NG], overflow: 0 };
    // absolute facts first, relabelled per viewer below
    let mut pieces: Vec<(usize, usize, usize)> = Vec::with_capacity(48); // (owner 0..4, type, sq)
    let mut threats: Vec<(usize, usize, usize, usize)> = Vec::with_capacity(32); // (owner, type, attacker, defended)
    let mut kz: Vec<(usize, usize, u8)> = Vec::with_capacity(12); // (king owner, attacker, count)
    for sq in 0..64 {
        let p = b[sq];
        if p == EMPTY {
            continue;
        }
        let (o, t) = (owner_of(p), type_of(p));
        pieces.push((o, t, sq));
        if o == DEAD {
            continue;
        }
        let mut defended = None;
        for a in 0..4 {
            if a == o || !r.active(a) {
                continue;
            }
            if cheapest_attacker(b, sq, a) > 0 {
                let d = *defended.get_or_insert_with(|| (cheapest_attacker(b, sq, o) > 0) as usize);
                threats.push((o, t, a, d));
            }
        }
    }
    for o in 0..4 {
        if !r.active(o) {
            continue;
        }
        let Some(ks) = b.iter().position(|&p| p == code(o, KING)) else { continue };
        let (kr, kc) = ((ks / 8) as i32, (ks % 8) as i32);
        for a in 0..4 {
            if a == o || !r.active(a) {
                continue;
            }
            let mut n = 0u8;
            for dr in -1..=1 {
                for dc in -1..=1 {
                    let (rr, cc) = (kr + dr, kc + dc);
                    if (0..8).contains(&rr) && (0..8).contains(&cc) && cheapest_attacker(b, (rr * 8 + cc) as usize, a) > 0 {
                        n += 1;
                    }
                }
            }
            if n > 0 {
                kz.push((o, a, n));
            }
        }
    }
    for i in 0..4 {
        let v = (m + i) % 4;
        let rel = |o: usize| if o == DEAD { 4 } else { (o + 4 - v) % 4 };
        let push = |f: &mut Feats, idx: usize, c: u8| {
            // merge repeated threat buckets into one count
            let n = f.n[i];
            if idx >= 1600 {
                for j in 0..n {
                    if f.idx[i][j] as usize == idx {
                        f.cnt[i][j] = f.cnt[i][j].saturating_add(c);
                        return;
                    }
                }
            }
            if n < MAXF {
                f.idx[i][n] = idx as u16;
                f.cnt[i][n] = c;
                f.n[i] = n + 1;
            } else {
                f.overflow += 1;
            }
        };
        for &(o, t, sq) in &pieces {
            push(&mut f, (rel(o) * 5 + t) * 64 + rot_square(sq, v), 1);
        }
        for &(o, t, a, d) in &threats {
            push(&mut f, 1600 + ((rel(o) * 5 + t) * 4 + rel(a)) * 2 + d, 1);
        }
        for &(o, a, c) in &kz {
            push(&mut f, 1760 + rel(o) * 4 + rel(a), c);
        }
    }
    // globals
    let seats = [m, (m + 1) % 4, (m + 2) % 4, (m + 3) % 4];
    let s: [f32; 4] = seats.map(|c| r.scores[c] as f32);
    let mean = (s[0] + s[1] + s[2] + s[3]) / 4.0;
    let g = &mut f.glob;
    for i in 0..4 {
        g[i] = r.active(seats[i]) as u8 as f32;
        g[4 + i] = s[i] / 30.0;
        g[8 + i] = (s[i] - mean) / 20.0;
    }
    g[12] = r.reps as f32 / 3.0;
    g[13] = (r.turn as f32 / 400.0).min(1.0);
    g[14] = b.iter().filter(|&&p| p == code(DEAD, KING)).count() as f32 / 3.0;
    let active = (0..4).filter(|&c| r.active(c)).count();
    g[15] = active as f32 / 4.0;
    for (n, &(a, c)) in PAIRS.iter().enumerate() {
        g[16 + n] = ((s[a] - s[c]) / 10.0).tanh();
    }
    g[22 + m] = 1.0;
    g[26] = (active == 2) as u8 as f32;
    g[27] = endgame_now(b) as u8 as f32;
    f
}

#[derive(Clone, Debug)]
pub struct Nn2 {
    pub a: usize,
    pub h1: usize,
    pub h2: usize,
    ft_w: Vec<f32>,
    ft_b: Vec<f32>,
    psq: Vec<f32>,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    w3: Vec<f32>,
    b3: Vec<f32>,
    pts_w: f32,
}

impl Nn2 {
    pub fn load(path: &Path) -> Result<Nn2, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {}", path.display(), e))?;
        let nl = bytes.iter().position(|&c| c == b'\n').ok_or("nn2: no header line")?;
        let head = std::str::from_utf8(&bytes[..nl]).map_err(|_| "nn2: bad header")?;
        let t: Vec<&str> = head.split_whitespace().collect();
        if t.len() != 7 || t[0] != "nn2" || t[1] != "1" {
            return Err(format!("nn2: bad header {:?}", head));
        }
        let num = |i: usize| t[i].parse::<usize>().map_err(|e| format!("nn2 header: {}", e));
        let (a, h1, h2, nf, ng) = (num(2)?, num(3)?, num(4)?, num(5)?, num(6)?);
        if nf != NF || ng != NG {
            return Err(format!("nn2: features {}x{}, this build has {}x{}", nf, ng, NF, NG));
        }
        let floats: Vec<f32> = bytes[nl + 1..].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let nin = 4 * a + NG;
        let want = NF * a + a + NF + h1 * nin + h1 + h2 * h1 + h2 + 4 * h2 + 4 + 1;
        if floats.len() != want || (bytes.len() - nl - 1) % 4 != 0 {
            return Err(format!("nn2: {} floats, want {}", floats.len(), want));
        }
        let mut at = 0;
        let mut take = |n: usize| {
            let v = floats[at..at + n].to_vec();
            at += n;
            v
        };
        Ok(Nn2 {
            a, h1, h2,
            ft_w: take(NF * a), ft_b: take(a), psq: take(NF),
            w1: take(h1 * nin), b1: take(h1), w2: take(h2 * h1), b2: take(h2),
            w3: take(4 * h2), b3: take(4), pts_w: take(1)[0],
        })
    }

    /// Strengths of the seats in turn order from the mover (S_0 = mover).
    pub fn forward(&self, f: &Feats, scores_rel: [f32; 4]) -> [f32; 4] {
        let a = self.a;
        let nin = 4 * a + NG;
        let mut x = vec![0f32; nin];
        let mut psq = [0f32; 4];
        for i in 0..4 {
            let acc = &mut x[i * a..(i + 1) * a];
            acc.copy_from_slice(&self.ft_b);
            for j in 0..f.n[i] {
                let (idx, c) = (f.idx[i][j] as usize, f.cnt[i][j] as f32);
                let row = &self.ft_w[idx * a..(idx + 1) * a];
                for k in 0..a {
                    acc[k] += c * row[k];
                }
                psq[i] += c * self.psq[idx];
            }
            for v in acc.iter_mut() {
                *v = v.clamp(0.0, 1.0);
            }
        }
        x[4 * a..].copy_from_slice(&f.glob);
        let mut h1 = vec![0f32; self.h1];
        for o in 0..self.h1 {
            let w = &self.w1[o * nin..(o + 1) * nin];
            let s: f32 = w.iter().zip(&x).map(|(p, q)| p * q).sum();
            h1[o] = (s + self.b1[o]).clamp(0.0, 1.0);
        }
        let mut h2 = vec![0f32; self.h2];
        for o in 0..self.h2 {
            let w = &self.w2[o * self.h1..(o + 1) * self.h1];
            let s: f32 = w.iter().zip(&h1).map(|(p, q)| p * q).sum();
            h2[o] = (s + self.b2[o]).clamp(0.0, 1.0);
        }
        let mut out = [0f32; 4];
        for i in 0..4 {
            let w = &self.w3[i * self.h2..(i + 1) * self.h2];
            out[i] = w.iter().zip(&h2).map(|(p, q)| p * q).sum::<f32>() + self.b3[i] + psq[i] + self.pts_w * scores_rel[i];
        }
        out
    }

    /// Absolute-seat strengths of a position.
    pub fn strengths(&self, pos: &Position) -> [f32; 4] {
        let r = Raw::of(pos);
        let m = r.mover();
        let f = features(&r);
        let rel = self.forward(&f, [0, 1, 2, 3].map(|i| r.scores[(m + i) % 4] as f32));
        let mut s = [0f32; 4];
        for i in 0..4 {
            s[(m + i) % 4] = rel[i];
        }
        s
    }

    /// The 6 pairwise logits of mover k (the Table interface).
    pub fn pst_logits(&self, pos: &Position, k: usize) -> [f32; 6] {
        let s = self.strengths(pos);
        let mut out = [0f32; 6];
        for (n, &(a, b)) in PAIRS.iter().enumerate() {
            out[n] = s[(k + a) % 4] - s[(k + b) % 4];
        }
        out
    }
}

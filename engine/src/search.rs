//! max^n search over the NNUE, a port of `bot/maxn.py` (the Python version stays as the
//! reference; the tests check both pick the same moves at a fixed depth).
//!
//! Every player picks the move that maximizes their own utility: their expected chess.com
//! FFA score (mean over opponents of P(finish above), ties half), from the NNUE at the
//! leaves and from the final scores at finished games. Utilities are in absolute seat
//! order (R, B, Y, G) and sum to 2.
//!
//! - iterative deepening (depth 1, 2, ...) inside a time budget, or to a fixed depth
//! - quiescence on point-scoring captures at the leaves (the mover may also stand pat)
//! - a transposition table keyed by board, side to move, scores and eliminated seats
//! - move ordering: the table's move first, then captures by points (stable, like Python)
//! - immediate pruning: a utility of 1 (above all three) can't be beaten
//! - all sibling leaves are evaluated together in one batched network pass
//! - if the time runs out before depth 1 finishes, the best child by one batched evaluation
//!   is played (depth 0 in the result)
//!
//! The position and the accumulators are made and unmade together, so nothing is copied.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::nnue::{AccStack, Net, NUM_OUTPUTS};
use crate::rules::{splitmix, Move, Position};

pub type Utilities = [f64; 4];

const PAIRS: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
const WIN: f64 = 1.0 - 1e-9;

/// Utilities (absolute seats) from the 6 pairwise logits of mover k (bot.nnue.utilities).
pub fn utilities_from_logits(logits: &[f32; NUM_OUTPUTS], k: usize) -> Utilities {
    let mut rel = [0.0f64; 4];
    for (n, &(i, j)) in PAIRS.iter().enumerate() {
        let p = 1.0 / (1.0 + (-(logits[n] as f64)).exp());
        rel[i] += p;
        rel[j] += 1.0 - p;
    }
    let mut abs = [0.0f64; 4];
    for i in 0..4 {
        abs[(k + i) % 4] = rel[i] / 3.0;
    }
    abs
}

/// Final FFA utilities from the scores (bot.maxn.exact_utilities).
pub fn exact_utilities(pos: &Position) -> Utilities {
    let s = pos.scores;
    let mut u = [0.0f64; 4];
    for i in 0..4 {
        let mut r = 0.0;
        for j in 0..4 {
            if i != j {
                r += if s[i] > s[j] { 1.0 } else if s[i] == s[j] { 0.5 } else { 0.0 };
            }
        }
        u[i] = r / 3.0;
    }
    u
}

#[derive(Clone, Copy)]
struct Entry {
    depth: u32,
    value: Utilities,
    best: Option<Move>,
}

#[derive(Clone, Debug, Default)]
pub struct SearchResult {
    pub best: Move,
    pub depth: u32,
    pub nodes: u64,
    pub value: Option<Utilities>,
    pub forced: bool,
    pub seconds: f64,
}

pub struct Searcher {
    pub net: Arc<Net>,
    pub time_ms: Option<f64>,
    pub depth: Option<u32>,
    pub max_depth: u32,
    pub qdepth: u32,
    pub tt_size: usize,
    tt: HashMap<u64, Entry>,
    nodes: u64,
    deadline: Option<Instant>,
}

impl Searcher {
    pub fn new(net: Arc<Net>, time_ms: Option<f64>, depth: Option<u32>, max_depth: u32, qdepth: u32,
               tt_size: usize) -> Result<Self, String> {
        if time_ms.is_none() && depth.is_none() {
            return Err("give a time budget or a fixed depth".into());
        }
        Ok(Searcher { net, time_ms, depth, max_depth, qdepth, tt_size, tt: HashMap::new(), nodes: 0, deadline: None })
    }

    pub fn clear(&mut self) {
        self.tt.clear();
    }

    pub fn best_move(&mut self, pos: &mut Position) -> Result<SearchResult, String> {
        let t0 = Instant::now();
        let moves = pos.legal_moves();
        if moves.is_empty() {
            return Err("no legal moves".into());
        }
        if moves.len() == 1 {
            return Ok(SearchResult { best: moves[0], forced: true, ..Default::default() });
        }
        self.deadline = self.time_ms.map(|ms| t0 + Duration::from_secs_f64(ms / 1000.0));
        self.nodes = 0;
        if self.tt.len() > self.tt_size {
            self.tt.clear();
        }
        let mut acc = AccStack::new(self.net.clone(), &pos.board)?;
        let me = pos.current_player();
        let mut res = SearchResult { best: moves[0], ..Default::default() };
        for d in 1..=self.depth.unwrap_or(self.max_depth) {
            match self.search(pos, &mut acc, d, true) {
                None => break, // out of time: keep the last completed depth
                Some((v, m)) => {
                    res.best = m.unwrap_or(moves[0]);
                    res.value = Some(v);
                    res.depth = d;
                    if v[me] >= WIN {
                        break;
                    }
                }
            }
        }
        if res.value.is_none() {
            // Out of time before depth 1 finished (a large network on a short clock): play
            // the best child by one batched evaluation, without quiescence, not moves[0].
            let kids = self.children(pos, &acc, &moves);
            let i = (0..moves.len()).fold(0, |b, i| if kids[i].1[me] > kids[b].1[me] { i } else { b });
            res.best = moves[i];
            res.value = Some(kids[i].1);
        }
        res.nodes = self.nodes;
        res.seconds = t0.elapsed().as_secs_f64();
        Ok(res)
    }

    /// Count a node; true when the time is up (checked every 64 nodes).
    #[inline]
    fn tick(&mut self) -> bool {
        self.nodes += 1;
        self.nodes & 63 == 0 && self.deadline.is_some_and(|d| Instant::now() > d)
    }

    fn key(pos: &Position) -> u64 {
        let mut h = splitmix(((pos.turn % 4) as u64) | ((pos.eliminated as u64) << 2));
        for s in pos.scores {
            h = splitmix(h ^ (s as u32 as u64));
        }
        pos.hash ^ h
    }

    fn ordered(pos: &Position, tt_move: Option<Move>) -> Vec<Move> {
        let mut moves = pos.legal_moves();
        moves.sort_by_key(|&m| -pos.capture_points(m)); // stable, like Python's sorted
        if let Some(t) = tt_move {
            if let Some(i) = moves.iter().position(|&m| m == t) {
                let m = moves.remove(i);
                moves.insert(0, m);
            }
        }
        moves
    }

    fn evaluate(&self, pos: &Position, acc: &mut AccStack) -> Utilities {
        let k = pos.current_player();
        utilities_from_logits(&acc.logits(k, &pos.globals(k)), k)
    }

    /// Utilities of each child after `moves`: exact for finished games, otherwise from one
    /// batched network pass over all live children. Returns (finished, utilities) per move.
    fn children(&mut self, pos: &mut Position, acc: &AccStack, moves: &[Move]) -> Vec<(bool, Utilities)> {
        let mut out = vec![(false, [0.0; 4]); moves.len()];
        let (mut boards, mut ks, mut globals, mut live) = (vec![], vec![], vec![], vec![]);
        for (i, &m) in moves.iter().enumerate() {
            pos.push(m).expect("legal move");
            if pos.game_over().is_some() {
                out[i] = (true, exact_utilities(pos));
            } else {
                let k = pos.current_player();
                boards.push(pos.board);
                ks.push(k);
                globals.extend_from_slice(&pos.globals(k));
                live.push(i);
            }
            pos.pop().expect("pop");
        }
        if !live.is_empty() {
            self.nodes += live.len() as u64;
            let logits = acc.child_logits(&boards, &ks, &globals).expect("child boards");
            for (n, &i) in live.iter().enumerate() {
                out[i] = (false, utilities_from_logits(&logits[n], ks[n]));
            }
        }
        out
    }

    fn push(pos: &mut Position, acc: &mut AccStack, m: Move) {
        pos.push(m).expect("legal move");
        acc.push(&pos.board).expect("board");
    }

    fn pop(pos: &mut Position, acc: &mut AccStack) {
        acc.pop().expect("acc pop");
        pos.pop().expect("pos pop");
    }

    /// None when the time ran out (the position and accumulators are restored either way).
    fn search(&mut self, pos: &mut Position, acc: &mut AccStack, depth: u32, root: bool) -> Option<(Utilities, Option<Move>)> {
        if self.tick() {
            return None;
        }
        if pos.game_over().is_some() {
            return Some((exact_utilities(pos), None));
        }
        if depth == 0 {
            let stand = self.evaluate(pos, acc);
            return self.quiesce(pos, acc, self.qdepth, stand).map(|v| (v, None));
        }
        let key = Self::key(pos);
        let hit = self.tt.get(&key).copied();
        if let Some(h) = hit {
            if h.depth >= depth && !root {
                return Some((h.value, h.best));
            }
        }
        let me = pos.current_player();
        let moves = Self::ordered(pos, hit.and_then(|h| h.best));
        if moves.is_empty() {
            return Some((self.evaluate(pos, acc), None));
        }
        let leaves = if depth == 1 { self.children(pos, acc, &moves) } else { Vec::new() };
        let (mut best_v, mut best_m): (Option<Utilities>, Option<Move>) = (None, None);
        for (i, &m) in moves.iter().enumerate() {
            let v = if depth == 1 {
                if self.tick() {
                    return None;
                }
                let (finished, stand) = leaves[i];
                if finished {
                    stand
                } else {
                    Self::push(pos, acc, m);
                    let r = self.quiesce(pos, acc, self.qdepth, stand);
                    Self::pop(pos, acc);
                    r?
                }
            } else {
                Self::push(pos, acc, m);
                let r = self.search(pos, acc, depth - 1, false);
                Self::pop(pos, acc);
                r?.0
            };
            if best_v.is_none_or(|b| v[me] > b[me]) {
                best_v = Some(v);
                best_m = Some(m);
                if v[me] >= WIN {
                    break;
                }
            }
        }
        let best_v = best_v.unwrap();
        self.tt.insert(key, Entry { depth, value: best_v, best: best_m });
        Some((best_v, best_m))
    }

    /// Stand pat (`stand`, this position's evaluation) or play a point-scoring capture.
    fn quiesce(&mut self, pos: &mut Position, acc: &mut AccStack, q: u32, stand: Utilities) -> Option<Utilities> {
        if q == 0 {
            return Some(stand);
        }
        let me = pos.current_player();
        let mut caps: Vec<Move> = pos.legal_moves().into_iter().filter(|&m| pos.capture_points(m) > 0).collect();
        if caps.is_empty() {
            return Some(stand);
        }
        caps.sort_by_key(|&m| -pos.capture_points(m));
        let kids = self.children(pos, acc, &caps);
        let mut best = stand;
        for (i, &m) in caps.iter().enumerate() {
            if self.tick() {
                return None;
            }
            let (finished, ev) = kids[i];
            let v = if finished || q == 1 {
                ev
            } else {
                Self::push(pos, acc, m);
                let r = self.quiesce(pos, acc, q - 1, ev);
                Self::pop(pos, acc);
                r?
            };
            if v[me] > best[me] {
                best = v;
            }
        }
        Some(best)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nnue::{Layer, NUM_FEATURES, NUM_GLOBALS};

    fn tiny_net() -> Arc<Net> {
        let mut s = 3u64;
        let mut v = |n: usize, scale: f32| {
            (0..n).map(|_| { s = splitmix(s); ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * scale }).collect::<Vec<f32>>()
        };
        let h = 8;
        let w0 = 4 * h + NUM_GLOBALS;
        let l1 = Layer::new(v(16 * w0, 0.5), v(16, 0.5), w0, 16, false).unwrap();
        let out = Layer::new(v(6 * 16, 1.0), v(6, 0.2), 16, 6, false).unwrap();
        Arc::new(Net::new(h, v(NUM_FEATURES * h, 0.2), v(h, 0.2), vec![l1], out).unwrap())
    }

    #[test]
    fn search_restores_the_position_and_is_legal() {
        let mut pos = Position::initial();
        for _ in 0..12 {
            let m = pos.legal_moves()[0];
            pos.push(m).unwrap();
        }
        let before = (pos.board, pos.turn, pos.scores, pos.depth());
        let mut s = Searcher::new(tiny_net(), None, Some(3), 12, 2, 100_000).unwrap();
        let r = s.best_move(&mut pos).unwrap();
        assert_eq!((pos.board, pos.turn, pos.scores, pos.depth()), before);
        assert!(pos.legal_moves().contains(&r.best));
        assert_eq!(r.depth, 3);
        let u = r.value.unwrap();
        assert!((u.iter().sum::<f64>() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn time_budget_is_respected() {
        let mut pos = Position::initial();
        let mut s = Searcher::new(tiny_net(), Some(50.0), None, 30, 2, 100_000).unwrap();
        let r = s.best_move(&mut pos).unwrap();
        assert!(r.seconds < 0.2, "{}", r.seconds);
        assert!(r.depth >= 1);
        assert_eq!(pos.depth(), 0);
    }

    #[test]
    fn exact_utilities_sum_to_two() {
        let mut pos = Position::initial();
        pos.scores = [10, 3, 10, 0];
        let u = exact_utilities(&pos);
        assert!((u[0] - 2.5 / 3.0).abs() < 1e-12 && u[3] == 0.0);
        assert!((u.iter().sum::<f64>() - 2.0).abs() < 1e-12);
    }
}

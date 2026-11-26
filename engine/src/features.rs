//! Extra network inputs: per seat, a few threat and material numbers the small network can
//! weigh (nn/README.md). Seats are relative to the mover k (k, k+1, k+2, k+3), like the
//! accumulators. An eliminated seat gets zeros.
//!
//! Per seat c, 5 numbers:
//!   0 material: active non-king piece points / 20
//!   1 threat: expected loss of c's attacked pieces before c's next move, each piece's
//!     worst attacker weighted 0.9 / 0.6 / 0.4 by how soon it moves, net of defenders
//!     (defended: value - attacker value); if c is to move, c saves the biggest one. King = 10.
//!     / 10
//!   2 now: if c is the mover, the best capture gain it has now (victim minus attacker if
//!     the victim is defended); otherwise the biggest loss the mover can inflict on c now. / 10
//!   3 king attacked by any active opponent (0 / 1)
//!   4 pawns within 2 squares of promotion / 4

use crate::greedy::attackers;
use crate::rules::{owner_of, type_of, Position, DEAD, EMPTY, KING, PAWN, PAWN_DIRECTION};

pub const NUM_EXT: usize = 20;
const THREAT_P: [f32; 3] = [0.9, 0.6, 0.4];

fn value(ty: usize) -> f32 {
    [10.0, 5.0, 3.0, 5.0, 1.0][ty]
}

fn pawn_to_go(color: usize, r: i32, c: i32) -> i32 {
    let (dr, dc) = PAWN_DIRECTION[color];
    if dr != 0 {
        if dr > 0 { 7 - r } else { r }
    } else if dc > 0 {
        7 - c
    } else {
        c
    }
}

/// Loss to `c` if `att` captures the piece of `c` on (r, c): 0 if `att` doesn't attack it.
#[inline]
fn loss(board: &[u8; 64], r: i32, col: i32, ty: usize, c: usize, att: usize) -> f32 {
    let a = attackers(board, r, col, att);
    if a == 0 {
        return 0.0;
    }
    let v = value(ty);
    if ty == KING {
        return v;
    }
    if attackers(board, r, col, c) > 0 { (v - a as f32).max(0.0) } else { v }
}

pub fn extended(pos: &Position, k: usize) -> [f32; NUM_EXT] {
    let mut out = [0.0f32; NUM_EXT];
    let board = &pos.board;
    let mover = pos.current_player();
    for i in 0..4 {
        let c = (k + i) % 4;
        if !pos.is_active(c) {
            continue;
        }
        // opponents in the order they move before c's next turn
        let mut order: Vec<usize> = Vec::with_capacity(3);
        let to_move = mover == c;
        if to_move {
            for d in 1..4 {
                let o = (c + d) % 4;
                if pos.is_active(o) {
                    order.push(o);
                }
            }
        } else {
            for d in 0..4 {
                let o = (mover + d) % 4;
                if o == c {
                    break;
                }
                if pos.is_active(o) {
                    order.push(o);
                }
            }
        }
        let (mut material, mut king_att, mut pawns, mut now) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let mut losses: Vec<f32> = Vec::new();
        for sq in 0..64 {
            let p = board[sq];
            if p == EMPTY || owner_of(p) != c {
                continue;
            }
            let ty = type_of(p);
            let (r, col) = ((sq / 8) as i32, (sq % 8) as i32);
            if ty == KING {
                if (0..4).any(|o| o != c && pos.is_active(o) && attackers(board, r, col, o) > 0) {
                    king_att = 1.0;
                }
            } else {
                material += value(ty);
            }
            if ty == PAWN && pawn_to_go(c, r, col) <= 2 {
                pawns += 1.0;
            }
            let mut worst = 0.0f32;
            for (n, &o) in order.iter().enumerate() {
                let l = loss(board, r, col, ty, c, o);
                if l > 0.0 {
                    let idx = (n + if to_move { 1 } else { 0 }).min(2);
                    worst = worst.max(l * THREAT_P[idx]);
                    if n == 0 && !to_move {
                        now = now.max(l);
                    }
                }
            }
            if worst > 0.0 {
                losses.push(worst);
            }
        }
        if to_move && !losses.is_empty() {
            let mut mi = 0;
            for j in 1..losses.len() {
                if losses[j] > losses[mi] {
                    mi = j;
                }
            }
            losses.remove(mi);
        }
        if to_move {
            // best capture gain for the mover now
            for sq in 0..64 {
                let p = board[sq];
                if p == EMPTY || owner_of(p) == c || owner_of(p) == DEAD {
                    continue;
                }
                let o = owner_of(p);
                let l = loss(board, (sq / 8) as i32, (sq % 8) as i32, type_of(p), o, c);
                now = now.max(l);
            }
        }
        let threat: f32 = losses.iter().sum();
        out[i * 5] = material / 20.0;
        out[i * 5 + 1] = threat / 10.0;
        out[i * 5 + 2] = now / 10.0;
        out[i * 5 + 3] = king_att;
        out[i * 5 + 4] = pawns / 4.0;
    }
    out
}

/// The network's global inputs for mover k: the 16 of `Position::globals`, then the
/// extended ones when the network takes them (`n` = its global width, 16 or 36).
pub fn globals_for(pos: &Position, k: usize, n: usize, out: &mut Vec<f32>) {
    out.extend_from_slice(&pos.globals(k));
    if n > 16 {
        out.extend_from_slice(&extended(pos, k)[..n - 16]);
    }
}

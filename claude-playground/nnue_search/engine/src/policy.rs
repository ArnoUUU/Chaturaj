//! Policy v1 (idea T02), the Rust side of `claude-playground/train/policy_pg.py`.
//!
//! A linear softmax over legal moves, trained on the moves of three top players. A move's score
//! is the sum of the weights of its active features (657 in all, layout as in policy_pg.py):
//!
//!   from            piece type x mover-frame from-square        5 x 64   0..320
//!   to              piece type x mover-frame to-square          5 x 64   320..640
//!   captured        none, K B N R P, dead                        7       640..647
//!   victim seat     next, across, prev, none                     4       647..651
//!   checks          1, 2, 3 enemy kings newly attacked           3       651..654
//!   promo           the move promotes                            1       654
//!   mover_attacked  from-square attacked by an opponent before   1       655
//!   dest_attacked   to-square attacked by an opponent after      1       656
//!
//! Mover-frame squares are pst::rot_square(sq, k) with k the mover's seat index (the mover sits
//! at the bottom, as in the piece-square tables). Scores are logits: the search uses them to pick
//! the children to keep (`Searcher::policy_width`) and to order them (`policy_order`).
//!
//! File format (`policy.txt`): `#` comments, `policy 1`, then `from <T> <64 floats>` and
//! `to <T> <64 floats>` for T in K B N R P, then `captured`, `victim`, `checks` (3), `promo`,
//! `mover_attacked`, `dest_attacked` (1 each), with their floats.

use std::collections::HashSet;
use std::path::Path;

use crate::pst::{rot_square, TYPE_LETTERS};
use crate::rules::{from_sq, is_attacked_by, is_promotion_square, owner_of, to_sq, type_of, Move, Position, DEAD, EMPTY, KING, PAWN, code};

pub const N_FEAT: usize = 657;
pub const F_FROM: usize = 0;
pub const F_TO: usize = 320;
pub const F_CAP: usize = 640;
pub const F_VIC: usize = 647;
pub const F_CHK: usize = 651;
pub const F_PROMO: usize = 654;
pub const F_MATT: usize = 655;
pub const F_DATT: usize = 656;

const NO_FEATURE: usize = usize::MAX;

#[derive(Clone, Debug, PartialEq)]
pub struct Policy {
    pub w: Vec<f32>,
}

impl Policy {
    pub fn parse(text: &str) -> Result<Policy, String> {
        let mut w = vec![0f32; N_FEAT];
        let mut version_ok = false;
        let mut seen: HashSet<String> = HashSet::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut tok = line.split_whitespace();
            let name = tok.next().unwrap_or("");
            if name == "policy" {
                if tok.next() != Some("1") {
                    return Err("policy file: unknown version".into());
                }
                version_ok = true;
                continue;
            }
            let (start, size, key) = match name {
                "from" | "to" => {
                    let ty = tok.next().ok_or("policy file: table without a type")?;
                    let t = TYPE_LETTERS.iter().position(|&x| x == ty).ok_or(format!("policy file: type {}", ty))?;
                    let base = if name == "from" { F_FROM } else { F_TO };
                    (base + t * 64, 64, format!("{} {}", name, ty))
                }
                "captured" => (F_CAP, 7, name.to_string()),
                "victim" => (F_VIC, 4, name.to_string()),
                "checks" => (F_CHK, 3, name.to_string()),
                "promo" => (F_PROMO, 1, name.to_string()),
                "mover_attacked" => (F_MATT, 1, name.to_string()),
                "dest_attacked" => (F_DATT, 1, name.to_string()),
                other => return Err(format!("policy file: unknown line start {:?}", other)),
            };
            let vals: Vec<f32> = tok
                .map(|x| x.parse::<f32>().map_err(|e| format!("policy file: {}", e)))
                .collect::<Result<_, _>>()?;
            if vals.len() != size {
                return Err(format!("policy file: {} wants {} values, got {}", name, size, vals.len()));
            }
            w[start..start + size].copy_from_slice(&vals);
            seen.insert(key);
        }
        if !version_ok {
            return Err("policy file: missing 'policy 1' line".into());
        }
        if seen.len() != 16 {
            return Err(format!("policy file: {} of 16 groups", seen.len()));
        }
        Ok(Policy { w })
    }

    pub fn load(path: &Path) -> Result<Policy, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {}", path.display(), e))?;
        Policy::parse(&text).map_err(|e| format!("{}: {}", path.display(), e))
    }

    /// The active feature indices of move `m` (NO_FEATURE for the ones that are off), with the
    /// position as it is on return. `m` must be legal.
    fn features(pos: &mut Position, m: Move) -> [usize; 8] {
        let mover = pos.current_player();
        let (from, to) = (from_sq(m), to_sq(m));
        let piece = pos.board[from];
        let t = type_of(piece);
        let mut out = [NO_FEATURE; 8];
        out[0] = F_FROM + t * 64 + rot_square(from, mover);
        out[1] = F_TO + t * 64 + rot_square(to, mover);
        let target = pos.board[to];
        let (cls, vic) = if target == EMPTY {
            (0, 3)
        } else if owner_of(target) == DEAD {
            (6, 3)
        } else {
            let rel = (owner_of(target) + 4 - mover) % 4;
            (1 + type_of(target), rel - 1)
        };
        out[2] = F_CAP + cls;
        out[3] = F_VIC + vic;
        let matt = (0..4).any(|c| c != mover && is_attacked_by(&pos.board, from, c));
        let before = enemy_kings_attacked(&pos.board, mover);
        pos.push(m).expect("legal move");
        let after = enemy_kings_attacked(&pos.board, mover);
        let datt = (0..4).any(|c| c != mover && is_attacked_by(&pos.board, to, c));
        pos.pop().expect("pop");
        let newly = (after & !before).count_ones() as usize;
        if newly > 0 {
            out[4] = F_CHK + newly.min(3) - 1;
        }
        if t == PAWN && is_promotion_square(mover, to) {
            out[5] = F_PROMO;
        }
        if matt {
            out[6] = F_MATT;
        }
        if datt {
            out[7] = F_DATT;
        }
        out
    }

    /// The policy score (logit) of move `m` in `pos`; the position is restored.
    pub fn score(&self, pos: &mut Position, m: Move) -> f32 {
        Policy::features(pos, m).iter().filter(|&&i| i != NO_FEATURE).map(|&i| self.w[i]).sum()
    }

    /// The scores of `moves` in order.
    pub fn score_moves(&self, pos: &mut Position, moves: &[Move]) -> Vec<f32> {
        moves.iter().map(|&m| self.score(pos, m)).collect()
    }
}

/// Bitmask of the enemy seats (not the mover) whose king the mover attacks.
fn enemy_kings_attacked(board: &[u8; 64], mover: usize) -> u8 {
    let mut out = 0u8;
    for enemy in 0..4 {
        if enemy == mover {
            continue;
        }
        if let Some(k) = board.iter().position(|&p| p == code(enemy, KING)) {
            if is_attacked_by(board, k, mover) {
                out |= 1 << enemy;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::{mv, Position};

    fn sq(name: &str) -> usize {
        let b = name.as_bytes();
        (8 - (b[1] - b'0') as usize) * 8 + (b[0] - b'a') as usize
    }

    #[test]
    fn table_file_parses_and_rejects_bad_input() {
        let text = include_str!("../tests/fixtures/policy_table.txt");
        let p = Policy::parse(text).unwrap();
        assert_eq!(p.w.len(), N_FEAT);
        assert!(p.w.iter().any(|&x| x != 0.0));
        assert!(Policy::parse("policy 2\n").is_err());
        assert!(Policy::parse(text.replace("captured", "captures").as_str()).is_err());
        let short = text.lines().filter(|l| !l.starts_with("promo")).collect::<Vec<_>>().join("\n");
        assert!(Policy::parse(&short).is_err());
    }

    #[test]
    fn score_is_the_sum_of_the_active_weights() {
        // a zero policy scores every move 0; one weight set to 2 shows up where the feature is on
        let mut pos = Position::initial();
        let zero = Policy { w: vec![0.0; N_FEAT] };
        let m = mv(sq("b2"), sq("b3"));
        assert_eq!(zero.score(&mut pos, m), 0.0);
        let mut p = zero.clone();
        p.w[F_FROM + 4 * 64 + rot_square(sq("b2"), 0)] = 2.0;     // Red's pawn from b2
        assert!((p.score(&mut pos, m) - 2.0).abs() < 1e-6);
        assert_eq!(pos.depth(), 0);
    }

    fn parse_fixture_line_codes(s: &str) -> Vec<u8> {
        s.split_whitespace().map(|x| x.parse::<i32>().unwrap() as u8).collect()
    }

    /// Python (policy_pg.py) scores of every legal move of 20 test positions, from the same
    /// weights, must match these to 1e-4.
    #[test]
    fn parity_with_python_fixture() {
        let pol = Policy::parse(include_str!("../tests/fixtures/policy_table.txt")).unwrap();
        let text = include_str!("../tests/fixtures/policy_parity.txt");
        let mut cases: Vec<(Position, Vec<(Move, f32)>)> = vec![];
        for line in text.lines() {
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some(rest) = line.strip_prefix("pos ") {
                let mut t = rest.split_whitespace();
                let turn: u32 = t.next().unwrap().parse().unwrap();
                let elim: u8 = t.next().unwrap().parse().unwrap();
                let codes = parse_fixture_line_codes(&t.collect::<Vec<_>>().join(" "));
                assert_eq!(codes.len(), 64);
                let mut board = [0u8; 64];
                board.copy_from_slice(&codes);
                let pos = Position::from_parts(board, turn, [0; 4], elim, vec![(board, 1)]).unwrap();
                cases.push((pos, vec![]));
            } else if let Some(rest) = line.strip_prefix("mv ") {
                let mut t = rest.split_whitespace();
                let m: Move = t.next().unwrap().parse().unwrap();
                let s: f32 = t.next().unwrap().parse().unwrap();
                cases.last_mut().unwrap().1.push((m, s));
            }
        }
        assert_eq!(cases.len(), 20);
        let mut max_diff = 0f32;
        let mut moves = 0;
        for (mut pos, expect) in cases {
            let legal = pos.legal_moves();
            assert_eq!(legal.len(), expect.len());
            for (i, &(m, want)) in expect.iter().enumerate() {
                assert_eq!(legal[i], m, "legal move order");
                let got = pol.score(&mut pos, m);
                max_diff = max_diff.max((got - want).abs());
                moves += 1;
            }
        }
        assert!(moves > 200, "{} moves", moves);
        assert!(max_diff < 1e-4, "max |diff| {}", max_diff);
    }
}

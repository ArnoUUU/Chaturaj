//! Tapered piece-square tables (idea 17), the Rust side of `claude-playground/train/texel_pg.py`.
//!
//! File format (`pst.txt`, written by texel_pg.py): `#` comments, `pst 1`, `c <float>`, then
//! `open <T> <64 floats>` and `end <T> <64 floats>` for T in K B N R P (rules.rs codes 0..4).
//! Squares are in the owner's frame: square = row * 8 + col, row 0 = rank 8, and a piece's
//! owner-frame square is its board square turned `owner` quarter turns, (r, c) -> (7 - c, r).
//!
//! Score of an absolute seat o (live pieces only; dead pieces score 0; an eliminated seat keeps
//! its points term):
//!   S[o] = sum over o's pieces of (1 - phase) * open[type][sq] + phase * end[type][sq]
//!          + c * points(o)
//!   phase = clip(1 - n_active / 32, 0, 1), n_active = pieces of active (non-dead) players.
//! The pairwise logits of mover k are S[(k+a)%4] - S[(k+b)%4] for the pairs of nnue::PAIRS.

use std::path::Path;

use crate::rules::{owner_of, type_of, DEAD, EMPTY, Position};

pub const TYPE_LETTERS: [&str; 5] = ["K", "B", "N", "R", "P"];
const PAIRS: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

#[derive(Clone, Debug, PartialEq)]
pub struct Pst {
    pub c: f32,
    pub open: [[f32; 64]; 5],
    pub end: [[f32; 64]; 5],
}

/// The square `sq` (row * 8 + col) turned `q` quarter turns, (r, c) -> (7 - c, r).
#[inline]
pub fn rot_square(sq: usize, q: usize) -> usize {
    let (mut r, mut c) = (sq / 8, sq % 8);
    for _ in 0..(q % 4) {
        let nr = 7 - c;
        c = r;
        r = nr;
    }
    r * 8 + c
}

impl Pst {
    pub fn parse(text: &str) -> Result<Pst, String> {
        let mut c: Option<f32> = None;
        let mut open = [[0f32; 64]; 5];
        let mut end = [[0f32; 64]; 5];
        let mut seen = [[false; 5]; 2];
        let mut version_ok = false;
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut tok = line.split_whitespace();
            match tok.next() {
                Some("pst") => {
                    if tok.next() != Some("1") {
                        return Err("pst file: unknown version".into());
                    }
                    version_ok = true;
                }
                Some("c") => {
                    c = Some(tok.next().ok_or("pst file: c without a value")?.parse().map_err(|e| format!("pst c: {}", e))?);
                }
                Some(tag @ ("open" | "end")) => {
                    let ty_name = tok.next().ok_or("pst file: table without a type")?;
                    let t = TYPE_LETTERS.iter().position(|&x| x == ty_name).ok_or(format!("pst file: type {}", ty_name))?;
                    let mut vals = [0f32; 64];
                    let mut n = 0;
                    for v in tok {
                        if n >= 64 {
                            return Err(format!("pst file: {} {} has more than 64 values", tag, ty_name));
                        }
                        vals[n] = v.parse().map_err(|e| format!("pst file: {}", e))?;
                        n += 1;
                    }
                    if n != 64 {
                        return Err(format!("pst file: {} {} has {} values", tag, ty_name, n));
                    }
                    if tag == "open" {
                        open[t] = vals;
                        seen[0][t] = true;
                    } else {
                        end[t] = vals;
                        seen[1][t] = true;
                    }
                }
                other => return Err(format!("pst file: unknown line start {:?}", other)),
            }
        }
        if !version_ok {
            return Err("pst file: missing 'pst 1' line".into());
        }
        let c = c.ok_or("pst file: missing c")?;
        if seen.iter().any(|row| row.iter().any(|&s| !s)) {
            return Err("pst file: missing tables".into());
        }
        Ok(Pst { c, open, end })
    }

    pub fn load(path: &Path) -> Result<Pst, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {}", path.display(), e))?;
        Pst::parse(&text).map_err(|e| format!("{}: {}", path.display(), e))
    }

    /// Score of each absolute seat (R, B, Y, G), from the board and the points.
    pub fn seat_scores(&self, pos: &Position) -> [f32; 4] {
        let n_active = pos.board.iter().filter(|&&p| p != EMPTY && owner_of(p) != DEAD).count();
        let phase = (1.0 - n_active as f32 / 32.0).clamp(0.0, 1.0);
        let mut s = [0f32; 4];
        for (sq, &p) in pos.board.iter().enumerate() {
            if p == EMPTY {
                continue;
            }
            let (o, t) = (owner_of(p), type_of(p));
            if o == DEAD {
                continue;
            }
            let osq = rot_square(sq, o);
            s[o] += (1.0 - phase) * self.open[t][osq] + phase * self.end[t][osq];
        }
        for o in 0..4 {
            s[o] += self.c * pos.scores[o] as f32;
        }
        s
    }

    /// The 6 pairwise logits of mover k (nnue::PAIRS order), as the network's outputs.
    pub fn pst_logits(&self, pos: &Position, k: usize) -> [f32; 6] {
        let s = self.seat_scores(pos);
        let mut out = [0f32; 6];
        for (n, &(a, b)) in PAIRS.iter().enumerate() {
            out[n] = s[(k + a) % 4] - s[(k + b) % 4];
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::{code, starting_board};

    const TABLE: &str = include_str!("../tests/fixtures/pst_table.txt");
    const PARITY: &str = include_str!("../tests/fixtures/pst_parity.txt");

    #[test]
    fn rotation_matches_the_python_convention() {
        // (r, c) -> (7 - c, r): square 0 (a8 corner, row 0 col 0) goes to row 7 col 0 = 56
        assert_eq!(rot_square(0, 1), 56);
        assert_eq!(rot_square(56, 1), 63);
        for sq in 0..64 {
            assert_eq!(rot_square(sq, 4), sq);
            assert_eq!(rot_square(rot_square(sq, 1), 3), sq);
        }
    }

    #[test]
    fn initial_pawns_map_to_the_same_owner_frame_squares_for_every_seat() {
        let board = starting_board();
        let mut per_seat = Vec::new();
        for o in 0..4 {
            let mut sq: Vec<usize> = (0..64)
                .filter(|&s| board[s] == code(o, 4))
                .map(|s| rot_square(s, o))
                .collect();
            sq.sort();
            per_seat.push(sq);
        }
        // all four seats agree: their four pawns sit on squares 48..51 (row 6, rank 2) of the owner's frame
        assert!(per_seat.windows(2).all(|w| w[0] == w[1]), "{:?}", per_seat);
        assert_eq!(per_seat[0], vec![48, 49, 50, 51]);
    }

    #[test]
    fn table_file_parses_and_rejects_bad_input() {
        let t = Pst::parse(TABLE).unwrap();
        assert!(t.c > 0.0);
        assert!(Pst::parse("pst 2\nc 0.1\n").is_err());
        assert!(Pst::parse("pst 1\nc 0.1\nopen K 1 2 3\n").is_err());
        assert!(Pst::parse(&TABLE.replace("open K", "open Q")).is_err());
    }

    #[test]
    fn logits_match_python_on_real_game_positions() {
        let t = Pst::parse(TABLE).unwrap();
        let mut n = 0;
        for line in PARITY.lines().filter(|l| l.starts_with("pos ")) {
            let parts: Vec<&str> = line.split(" | ").collect();
            assert_eq!(parts.len(), 3, "{}", line);
            let head: Vec<&str> = parts[0].split_whitespace().collect();
            // pos <label> <k> <turn> <elim> <s0> <s1> <s2> <s3>
            let k: usize = head[2].parse().unwrap();
            let turn: u32 = head[3].parse().unwrap();
            let elim: u8 = head[4].parse().unwrap();
            let scores: [i32; 4] = [head[5], head[6], head[7], head[8]].map(|x| x.parse().unwrap());
            let codes: Vec<u8> = parts[1].split_whitespace().map(|x| x.parse().unwrap()).collect();
            assert_eq!(codes.len(), 64);
            let mut board = [0u8; 64];
            board.copy_from_slice(&codes);
            let want: Vec<f32> = parts[2].split_whitespace().map(|x| x.parse().unwrap()).collect();
            let pos = Position::from_parts(board, turn, scores, elim, vec![]).unwrap();
            let got = t.pst_logits(&pos, k);
            for j in 0..6 {
                assert!((got[j] - want[j]).abs() < 1e-4, "{} logit {}: rust {} python {}", head[1], j, got[j], want[j]);
            }
            n += 1;
        }
        assert!(n >= 10, "fixture has {} positions", n);
    }

    #[test]
    fn dead_pieces_score_nothing() {
        let t = Pst::parse(TABLE).unwrap();
        let mut board = [0u8; 64];
        board[40] = code(4, 4); // a dead pawn
        let pos = Position::from_parts(board, 0, [0; 4], 0, vec![]).unwrap();
        assert_eq!(t.seat_scores(&pos), [0.0; 4]);
    }
}

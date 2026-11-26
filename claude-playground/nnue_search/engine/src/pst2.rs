//! Table v2 (idea 21), the Rust side of `claude-playground/train/texel_v2.py`.
//!
//! The tapered piece-square tables of pst.rs, with two sets (4 players left, 3 or fewer), plus
//! 13 per-seat features weighted by fitted numbers: hanging pieces per type, mobility per type
//! (/ 10), king danger, in check, and check threats. Attacks are rules::is_attacked_by and
//! mobility is rules::piece_moves, so the features match the Python loop reference
//! (texel_v2.ref_features) and the numpy generator; the tests check both.
//!
//! File format (`pst2.txt`): `#` comments, `pst2 1`, `c <float>`, `hang` (5 floats K B N R P),
//! `mob` (5 floats), `kd <float>`, `chk <float>`, `threat <float>`, then 20 table lines
//! `open4 T <64>`, `end4 T <64>`, `open3 T <64>`, `end3 T <64>` for T in K B N R P. Squares are
//! in the owner's frame (pst.rs `rot_square`).
//!
//! Score of an absolute seat s (the same as the Python model):
//!   S[s] = sum over s's live pieces of (1 - phase) open[set][t][sq] + phase end[set][t][sq]
//!        + c * points(s) + sum_f w[f] * F[s][f]
//!   set = 0 when every seat has a king (4 players left), else 1; phase as in pst.rs.
//!   F[s] = 0 for a seat without a king (eliminated).
//! Pairwise logits of mover k: S[(k+a)%4] - S[(k+b)%4] for the pairs of nnue::PAIRS.

use std::path::Path;

use crate::pst::{rot_square, TYPE_LETTERS};
use crate::rules::{code, is_attacked_by, owner_of, piece_moves, type_of, Position, DEAD, EMPTY, KING};

pub const NFEAT: usize = 13;
const PAIRS: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

#[derive(Clone, Debug, PartialEq)]
pub struct Pst2 {
    pub c: f32,
    pub hang: [f32; 5],
    pub mob: [f32; 5],
    pub kd: f32,
    pub chk: f32,
    pub threat: f32,
    /// [set][type][owner-frame square]; set 0 = 4 players left, 1 = 3 or fewer
    pub open: [[[f32; 64]; 5]; 2],
    pub end: [[[f32; 64]; 5]; 2],
}

impl Pst2 {
    pub fn parse(text: &str) -> Result<Pst2, String> {
        let mut version_ok = false;
        let mut c: Option<f32> = None;
        let (mut hang, mut mob) = (None, None);
        let (mut kd, mut chk, mut threat) = (None, None, None);
        let mut open = [[[0f32; 64]; 5]; 2];
        let mut end = [[[0f32; 64]; 5]; 2];
        let mut seen = std::collections::HashSet::new();
        let floats = |toks: &mut std::str::SplitWhitespace, n: usize| -> Result<Vec<f32>, String> {
            let v: Vec<f32> = toks.map(|x| x.parse::<f32>().map_err(|e| format!("pst2 file: {}", e))).collect::<Result<_, _>>()?;
            if v.len() != n {
                return Err(format!("pst2 file: want {} values, got {}", n, v.len()));
            }
            Ok(v)
        };
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut tok = line.split_whitespace();
            match tok.next() {
                Some("pst2") => {
                    if tok.next() != Some("1") {
                        return Err("pst2 file: unknown version".into());
                    }
                    version_ok = true;
                }
                Some("c") => c = Some(tok.next().ok_or("pst2 file: c without a value")?.parse().map_err(|e| format!("pst2 c: {}", e))?),
                Some("hang") => {
                    let v = floats(&mut tok, 5)?;
                    hang = Some([v[0], v[1], v[2], v[3], v[4]]);
                }
                Some("mob") => {
                    let v = floats(&mut tok, 5)?;
                    mob = Some([v[0], v[1], v[2], v[3], v[4]]);
                }
                Some("kd") => kd = Some(floats(&mut tok, 1)?[0]),
                Some("chk") => chk = Some(floats(&mut tok, 1)?[0]),
                Some("threat") => threat = Some(floats(&mut tok, 1)?[0]),
                Some(tag @ ("open4" | "end4" | "open3" | "end3")) => {
                    let ty_name = tok.next().ok_or("pst2 file: table without a type")?;
                    let t = TYPE_LETTERS.iter().position(|&x| x == ty_name).ok_or(format!("pst2 file: type {}", ty_name))?;
                    let v = floats(&mut tok, 64)?;
                    let set = if tag.ends_with('4') { 0 } else { 1 };
                    let dst = if tag.starts_with("open") { &mut open[set][t] } else { &mut end[set][t] };
                    dst.copy_from_slice(&v);
                    seen.insert(tag.to_string() + ty_name);
                }
                other => return Err(format!("pst2 file: unknown line start {:?}", other)),
            }
        }
        if !version_ok {
            return Err("pst2 file: missing 'pst2 1' line".into());
        }
        let c = c.ok_or("pst2 file: missing c")?;
        let hang = hang.ok_or("pst2 file: missing hang")?;
        let mob = mob.ok_or("pst2 file: missing mob")?;
        let kd = kd.ok_or("pst2 file: missing kd")?;
        let chk = chk.ok_or("pst2 file: missing chk")?;
        let threat = threat.ok_or("pst2 file: missing threat")?;
        if seen.len() != 20 {
            return Err(format!("pst2 file: {} of 20 tables", seen.len()));
        }
        Ok(Pst2 { c, hang, mob, kd, chk, threat, open, end })
    }

    pub fn load(path: &Path) -> Result<Pst2, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {}", path.display(), e))?;
        Pst2::parse(&text).map_err(|e| format!("{}: {}", path.display(), e))
    }

    fn weights(&self) -> [f32; NFEAT] {
        let mut w = [0f32; NFEAT];
        w[0..5].copy_from_slice(&self.hang);
        w[5..10].copy_from_slice(&self.mob);
        w[10] = self.kd;
        w[11] = self.chk;
        w[12] = self.threat;
        w
    }

    /// The 13 features of every absolute seat (zeros for a seat without a king).
    pub fn features(pos: &Position) -> [[f32; NFEAT]; 4] {
        let b = &pos.board;
        let mut out = [[0f32; NFEAT]; 4];
        let king_sq: [Option<usize>; 4] = [0, 1, 2, 3].map(|s| b.iter().position(|&p| p == code(s, KING)));
        for s in 0..4 {
            let Some(ks) = king_sq[s] else { continue };
            let f = &mut out[s];
            let opp_attacks = |sq: usize| (0..4).any(|c| c != s && is_attacked_by(b, sq, c));
            let mut moves = Vec::with_capacity(32);
            for sq in 0..64 {
                let p = b[sq];
                if p == EMPTY || owner_of(p) != s {
                    continue;
                }
                let t = type_of(p);
                if opp_attacks(sq) && !is_attacked_by(b, sq, s) {
                    f[t] += 1.0;
                }
                moves.clear();
                piece_moves(b, sq, &mut moves);
                f[5 + t] += moves.len() as f32 / 10.0;
            }
            let (r, c) = ((ks / 8) as i32, (ks % 8) as i32);
            let mut danger = 0u32;
            for dr in -1..=1 {
                for dc in -1..=1 {
                    let (rr, cc) = (r + dr, c + dc);
                    if (0..8).contains(&rr) && (0..8).contains(&cc) && opp_attacks((rr * 8 + cc) as usize) {
                        danger += 1;
                    }
                }
            }
            f[10] = danger as f32;
            f[11] = if opp_attacks(ks) { 1.0 } else { 0.0 };
            let mut threats = 0u32;
            for c2 in 0..4 {
                if c2 == s {
                    continue;
                }
                if let Some(kc) = king_sq[c2] {
                    if is_attacked_by(b, kc, s) {
                        threats += 1;
                    }
                }
            }
            f[12] = threats as f32;
        }
        out
    }

    /// Score of each absolute seat (dead pieces score 0; the points term stays for every seat).
    pub fn seat_scores(&self, pos: &Position) -> [f32; 4] {
        let kings = (0..4).filter(|&s| pos.board.contains(&code(s, KING))).count();
        let set = if kings >= 4 { 0 } else { 1 };
        let n_alive = pos.board.iter().filter(|&&p| p != EMPTY && owner_of(p) != DEAD).count();
        let phase = (1.0 - n_alive as f32 / 32.0).clamp(0.0, 1.0);
        let mut s = [0f32; 4];
        for (sq, &p) in pos.board.iter().enumerate() {
            if p == EMPTY || owner_of(p) == DEAD {
                continue;
            }
            let (o, t) = (owner_of(p), type_of(p));
            let osq = rot_square(sq, o);
            s[o] += (1.0 - phase) * self.open[set][t][osq] + phase * self.end[set][t][osq];
        }
        let w = self.weights();
        let feats = Pst2::features(pos);
        for o in 0..4 {
            s[o] += self.c * pos.scores[o] as f32;
            s[o] += w.iter().zip(feats[o].iter()).map(|(a, b)| a * b).sum::<f32>();
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
    use crate::rules::{Move, Position};

    const TABLE: &str = include_str!("../tests/fixtures/pst2_table.txt");
    const PARITY: &str = include_str!("../tests/fixtures/pst2_parity.txt");
    const ATTACKS: &str = include_str!("../tests/fixtures/attack_fixture.txt");

    fn parse_codes(s: &str) -> [u8; 64] {
        let v: Vec<u8> = s.split_whitespace().map(|x| x.parse().unwrap()).collect();
        assert_eq!(v.len(), 64);
        let mut b = [0u8; 64];
        b.copy_from_slice(&v);
        b
    }

    #[test]
    fn attack_maps_and_mobility_match_the_python_generator() {
        // 200 random boards: is_attacked_by on every square and seat, piece_moves counts per
        // seat and type, and the features; the Python side is the numpy generator (texel_v2.py)
        let (mut n, mut bad_att, mut bad_mob) = (0, 0, 0);
        let mut worst = 0f32;
        for line in ATTACKS.lines().filter(|l| l.starts_with("pos ")) {
            let parts: Vec<&str> = line.split(" | ").collect();
            assert_eq!(parts.len(), 4, "{}", line);
            let board = parse_codes(&parts[0][4..]);
            let att: Vec<u8> = parts[1].split_whitespace().map(|x| x.parse().unwrap()).collect();
            let mob: Vec<u32> = parts[2].split_whitespace().map(|x| x.parse().unwrap()).collect();
            let feats: Vec<f32> = parts[3].split_whitespace().map(|x| x.parse().unwrap()).collect();
            for s in 0..4 {
                for sq in 0..64 {
                    let want = att[s * 64 + sq] == 1;
                    if is_attacked_by(&board, sq, s) != want {
                        bad_att += 1;
                    }
                }
                for t in 0..5 {
                    let mut total = 0u32;
                    for sq in 0..64 {
                        let p = board[sq];
                        if p != EMPTY && owner_of(p) == s && type_of(p) == t {
                            let mut out: Vec<Move> = Vec::new();
                            piece_moves(&board, sq, &mut out);
                            total += out.len() as u32;
                        }
                    }
                    if total != mob[s * 5 + t] {
                        bad_mob += 1;
                    }
                }
            }
            let got = Pst2::features(&Position::from_parts(board, 0, [0; 4], 0, vec![]).unwrap());
            for s in 0..4 {
                for f in 0..NFEAT {
                    worst = worst.max((got[s][f] - feats[s * NFEAT + f]).abs());
                }
            }
            n += 1;
        }
        assert!(n >= 150, "fixture has {} boards", n);
        assert_eq!(bad_att, 0, "attack map mismatches on {} boards", n);
        assert_eq!(bad_mob, 0, "mobility mismatches on {} boards", n);
        assert!(worst < 1e-5, "feature difference {}", worst);
    }

    #[test]
    fn table_file_parses_and_rejects_bad_input() {
        let t = Pst2::parse(TABLE).unwrap();
        assert!(t.c > 0.0);
        assert!(Pst2::parse("pst2 2\n").is_err());
        assert!(Pst2::parse(&TABLE.replace("threat ", "threatx ")).is_err());
        assert!(Pst2::parse(&TABLE.replace("open4 K", "open4 Q")).is_err());
    }

    #[test]
    fn logits_match_python_on_game_positions() {
        let t = Pst2::parse(TABLE).unwrap();
        let mut n = 0;
        for line in PARITY.lines().filter(|l| l.starts_with("pos ")) {
            let parts: Vec<&str> = line.split(" | ").collect();
            assert_eq!(parts.len(), 4, "{}", line);
            let head: Vec<&str> = parts[0].split_whitespace().collect();
            let k: usize = head[2].parse().unwrap();
            let turn: u32 = head[3].parse().unwrap();
            let elim: u8 = head[4].parse().unwrap();
            let scores: [i32; 4] = [head[5], head[6], head[7], head[8]].map(|x| x.parse().unwrap());
            let board = parse_codes(parts[1]);
            let want: Vec<f32> = parts[2].split_whitespace().map(|x| x.parse().unwrap()).collect();
            let want_f: Vec<f32> = parts[3].split_whitespace().map(|x| x.parse().unwrap()).collect();
            let pos = Position::from_parts(board, turn, scores, elim, vec![]).unwrap();
            let got = t.pst_logits(&pos, k);
            for j in 0..6 {
                assert!((got[j] - want[j]).abs() < 1e-4, "{} logit {}: rust {} python {}", head[1], j, got[j], want[j]);
            }
            let f = Pst2::features(&pos);
            for s in 0..4 {
                for j in 0..NFEAT {
                    assert!((f[s][j] - want_f[s * NFEAT + j]).abs() < 1e-4, "{} seat {} feature {}", head[1], s, j);
                }
            }
            n += 1;
        }
        assert!(n >= 15, "fixture has {} positions", n);
    }

    #[test]
    fn dead_pieces_score_nothing_and_attack_nothing() {
        let t = Pst2::parse(TABLE).unwrap();
        let mut board = [0u8; 64];
        board[40] = code(4, 4); // a dead pawn
        let pos = Position::from_parts(board, 0, [0; 4], 0, vec![]).unwrap();
        assert_eq!(t.seat_scores(&pos), [0.0; 4]);
        assert_eq!(Pst2::features(&pos), [[0f32; NFEAT]; 4]);
    }
}

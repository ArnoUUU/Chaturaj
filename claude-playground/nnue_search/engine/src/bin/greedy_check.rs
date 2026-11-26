//! Replays tools/greedy_ref.py games (bot/greedy.py at a fixed depth) with the Rust greedy bot:
//! same move, value (to 0.006) and node count at every searched ply.
//!   greedy_check <ref.txt> <depth> [preset]
use engine_pg::greedy::{GreedyBot, GreedyParams, RESIGN};
use engine_pg::rules::{mv, Move, Position};

fn sq(s: &str) -> usize {
    let b = s.as_bytes();
    (8 - (b[1] - b'0') as usize) * 8 + (b[0] - b'a') as usize
}
fn parse(s: &str) -> Move {
    if s == "resign" { RESIGN } else { mv(sq(&s[0..2]), sq(&s[2..4])) }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let text = std::fs::read_to_string(&a[1]).expect("ref file");
    let depth: u32 = a[2].parse().unwrap();
    let preset = a.get(3).map(|s| s.as_str()).unwrap_or("greedy");
    let mut pos = Position::initial();
    let mut bots: Vec<GreedyBot> = Vec::new();
    let (mut plies, mut searched, mut bm, mut bv, mut bn) = (0, 0, 0, 0, 0);
    let (mut secs, mut nodes) = (0.0, 0u64);
    for line in text.lines() {
        if line.starts_with("game") {
            pos = Position::initial();
            bots = (0..4).map(|_| {
                let mut b = GreedyBot::new(GreedyParams::preset(preset).unwrap());
                b.fixed_depth = Some(depth);
                b
            }).collect();
            continue;
        }
        let t: Vec<&str> = line.split_whitespace().collect();
        let want = parse(t[0]);
        if t[1] != "-" {
            let me = pos.current_player();
            let r = bots[me].best_move(&mut pos);
            searched += 1;
            secs += r.seconds;
            nodes += r.nodes;
            let okm = r.best == want;
            let okv = (r.value - t[1].parse::<f64>().unwrap()).abs() < 0.006;
            let okn = r.nodes == t[2].parse::<u64>().unwrap();
            if (!okm || !okv || !okn) && bm + bv + bn < 8 {
                eprintln!("ply {}: want {} {} n={} got {} {:.3} n={}", plies, t[0], t[1], t[2], r.best, r.value, r.nodes);
            }
            bm += !okm as u32;
            bv += !okv as u32;
            bn += !okn as u32;
        }
        if want == RESIGN { pos.eliminate(pos.current_player()); } else { pos.push(want).unwrap(); }
        plies += 1;
    }
    println!("plies {} searched {} | move mismatches {}, value mismatches {}, node mismatches {} | {:.0} knodes/s",
             plies, searched, bm, bv, bn, nodes as f64 / secs.max(1e-9) / 1000.0);
    if bm > 0 { std::process::exit(1); }
}

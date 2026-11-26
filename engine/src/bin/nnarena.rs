//! Arena for the neural-net engine against the greedy baseline (nn/README.md).
//!
//!   nnarena arena --net nn/runs/r0/net.bin --games 160 --opening-seed 0 --ms 60 --out-games g.jsonl --out-pos p.bin
//!   nnarena greedy-game <opening move codes...>     # greedy x4 from those opening moves, prints the moves
//!   nnarena time                                    # greedy time per move at depths 1..5
//!
//! Openings: opening i is 4 random plies from seed (opening_seed * 100003 + i). Every opening
//! is played 4 times, the test player in each seat once (games 4i .. 4i+3), so colours are
//! balanced exactly. Opponents: one greedy bot per seat (its own transposition table).
//! Score = pairwise points of 3 against the three opponents on final points (ties 0.5), so
//! 1.5 = even with greedy.

use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use chaturaji_engine::features::globals_for;
use chaturaji_engine::greedy::{preset, Greedy};
use chaturaji_engine::nnue::{Layer, Net, NUM_FEATURES, NUM_GLOBALS, NUM_OUTPUTS};
use chaturaji_engine::rules::{is_attacked_by, owner_of, splitmix, type_of, Move, Position};
use chaturaji_engine::search::{Searcher, RESIGN};
use rayon::prelude::*;

fn arg<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn arg_s(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

/// Net file: b"CNN1", u32 h, u32 n_hidden, ft_w (1600*h), ft_b (h), then per layer
/// u32 n_in, u32 n_out, u32 residual, w (n_out*n_in), b (n_out); the output layer last
/// (same header, n_out = 6). Little-endian f32.
fn load_net(path: &str) -> Arc<Net> {
    let mut buf = Vec::new();
    File::open(path).unwrap_or_else(|e| panic!("{path}: {e}")).read_to_end(&mut buf).unwrap();
    assert_eq!(&buf[..4], b"CNN1", "{path}: not a CNN1 net");
    let mut off = 4;
    let u32_ = |off: &mut usize| {
        let v = u32::from_le_bytes(buf[*off..*off + 4].try_into().unwrap());
        *off += 4;
        v as usize
    };
    let h = u32_(&mut off);
    let n_hidden = u32_(&mut off);
    let f32s = |off: &mut usize, n: usize| {
        let v: Vec<f32> = buf[*off..*off + 4 * n].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        *off += 4 * n;
        v
    };
    let ft_w = f32s(&mut off, NUM_FEATURES * h);
    let ft_b = f32s(&mut off, h);
    let mut layers = Vec::new();
    let read_layer = |off: &mut usize| {
        let n_in = u32::from_le_bytes(buf[*off..*off + 4].try_into().unwrap()) as usize;
        let n_out = u32::from_le_bytes(buf[*off + 4..*off + 8].try_into().unwrap()) as usize;
        let res = u32::from_le_bytes(buf[*off + 8..*off + 12].try_into().unwrap()) != 0;
        *off += 12;
        let w: Vec<f32> = buf[*off..*off + 4 * n_in * n_out].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        *off += 4 * n_in * n_out;
        let b: Vec<f32> = buf[*off..*off + 4 * n_out].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        *off += 4 * n_out;
        Layer::new(w, b, n_in, n_out, res).unwrap()
    };
    for _ in 0..n_hidden {
        layers.push(read_layer(&mut off));
    }
    let out = read_layer(&mut off);
    let mut net = Net::new(h, ft_w, ft_b, layers, out).unwrap();
    if net.out.n_out == 4 {
        // points head: beta follows the output layer
        net.points_beta = Some(f32::from_le_bytes(buf[off..off + 4].try_into().unwrap()));
    } else {
        assert_eq!(net.out.n_out, NUM_OUTPUTS);
    }
    Arc::new(net)
}

/// An untrained net: every weight 0, so every leaf is a 3-way tie (only finished games count).
fn zero_net(h: usize) -> Arc<Net> {
    let w0 = 4 * h + NUM_GLOBALS;
    let l1 = Layer::new(vec![0.0; 32 * w0], vec![0.0; 32], w0, 32, false).unwrap();
    let out = Layer::new(vec![0.0; 6 * 32], vec![0.0; 6], 32, 6, false).unwrap();
    Arc::new(Net::new(h, vec![0.0; NUM_FEATURES * h], vec![0.0; h], vec![l1], out).unwrap())
}

/// `plies` random moves; redrawn until no king is attacked at the end (a random opening
/// that hangs a king decides the game before the players move).
fn opening(seed: u64, plies: usize) -> Vec<Move> {
    let mut s = splitmix(seed ^ 0x9E37_79B9_7F4A_7C15);
    loop {
        let mut pos = Position::initial();
        let mut out = vec![];
        for _ in 0..plies {
            let ms = pos.legal_moves();
            s = splitmix(s);
            let m = ms[(s % ms.len() as u64) as usize];
            pos.push(m).unwrap();
            out.push(m);
        }
        let hung = (0..64).any(|sq| {
            let p = pos.board[sq];
            p != 0 && owner_of(p) < 4 && type_of(p) == 0 && (0..4).any(|c| c != owner_of(p) && is_attacked_by(&pos.board, sq, c))
        });
        if !hung {
            return out;
        }
    }
}

enum Player {
    Nn(Searcher),
    Greedy(Greedy),
}

impl Player {
    fn play(&mut self, pos: &mut Position) -> (Move, u32) {
        match self {
            Player::Nn(s) => {
                let r = s.best_move(pos).unwrap();
                (r.best, r.depth)
            }
            Player::Greedy(g) => {
                let r = g.best_move(pos);
                (r.best, r.depth)
            }
        }
    }
}

fn apply(pos: &mut Position, m: Move) {
    if m == RESIGN {
        let c = pos.current_player();
        pos.eliminate(c);
    } else {
        pos.push(m).unwrap();
    }
}

/// One stored position (a record of nn/data.py's dtype, 112 bytes).
fn write_pos(out: &mut Vec<u8>, pos: &Position, game: u32, test_seat: u8) {
    let k = pos.current_player();
    out.extend_from_slice(&pos.board);
    out.push(k as u8);
    out.push(test_seat);
    out.extend_from_slice(&(pos.turn as u16).to_le_bytes());
    out.extend_from_slice(&game.to_le_bytes());
    for s in pos.scores {
        out.extend_from_slice(&(s as i16).to_le_bytes());
    }
    let mut g = Vec::with_capacity(36);
    globals_for(pos, k, 36, &mut g);
    for x in g {
        out.extend_from_slice(&x.to_le_bytes());
    }
    // final scores are patched in at the end of the game (8 bytes)
    out.extend_from_slice(&[0u8; 8]);
}

const REC: usize = 64 + 1 + 1 + 2 + 4 + 8 + 4 * 36 + 8;

struct Cfg {
    net: Option<Arc<Net>>,
    test: String,
    opp: String,
    ms: f64,
    nn_depth: Option<u32>,
    greedy_depth: Option<u32>,
    greedy_ms: Option<f64>,
    qdepth: u32,
    check_ext: u32,
    width: usize,
    full_plies: u32,
    max_plies: u32,
    opening_seed: u64,
    opening_plies: usize,
}

fn make(spec: &str, cfg: &Cfg) -> Player {
    if let Some(p) = preset(spec) {
        let mut g = Greedy::new(p, cfg.greedy_depth, cfg.greedy_ms);
        if cfg.greedy_ms.is_some() {
            g.depth = None;
        } else if g.depth.is_none() {
            g.depth = Some(p.depth);
        }
        return Player::Greedy(g);
    }
    assert_eq!(spec, "nn", "player is nn or a greedy preset");
    let net = cfg.net.clone().expect("--net");
    let time = if cfg.nn_depth.is_some() { None } else { Some(cfg.ms) };
    let mut s = Searcher::new(net, time, cfg.nn_depth, 12, cfg.qdepth, 1 << 20, cfg.check_ext, cfg.width, cfg.full_plies, 1.0, 0.0, 200, true).unwrap();
    s.resign_multi = false; // like greedy: only claim the win with two players left
    Player::Nn(s)
}

fn play_game(g: usize, cfg: &Cfg) -> (String, Vec<u8>) {
    let op = g / 4;
    let seat = g % 4;
    let mut pos = Position::initial();
    for m in opening(cfg.opening_seed * 100_003 + op as u64, cfg.opening_plies) {
        pos.push(m).unwrap();
    }
    let mut players: Vec<Player> = (0..4).map(|s| make(if s == seat { &cfg.test } else { &cfg.opp }, cfg)).collect();
    let mut recs = Vec::new();
    let mut n_rec = 0usize;
    let (mut t_test, mut n_test, mut d_test, mut t_opp, mut n_opp) = (0.0, 0u32, 0u32, 0.0, 0u32);
    while pos.game_over().is_none() && pos.turn < cfg.max_plies {
        write_pos(&mut recs, &pos, g as u32, seat as u8);
        n_rec += 1;
        let who = pos.current_player();
        let t = Instant::now();
        let (m, d) = players[who].play(&mut pos);
        let dt = t.elapsed().as_secs_f64();
        if who == seat {
            t_test += dt;
            n_test += 1;
            d_test += d;
        } else {
            t_opp += dt;
            n_opp += 1;
        }
        apply(&mut pos, m);
    }
    let s = pos.scores;
    for i in 0..n_rec {
        let o = i * REC + REC - 8;
        for c in 0..4 {
            recs[o + 2 * c..o + 2 * c + 2].copy_from_slice(&(s[c] as i16).to_le_bytes());
        }
    }
    let pair: f64 = (0..4).filter(|&c| c != seat).map(|c| if s[seat] > s[c] { 1.0 } else if s[seat] == s[c] { 0.5 } else { 0.0 }).sum();
    let line = format!(
        "{{\"game\":{},\"opening\":{},\"seat\":{},\"scores\":[{},{},{},{}],\"pairwise_of_3\":{},\"plies\":{},\"over\":{},\"test_ms\":{:.1},\"test_depth\":{:.2},\"opp_ms\":{:.1}}}",
        g, op, seat, s[0], s[1], s[2], s[3], pair, pos.turn, pos.game_over().is_some(),
        1000.0 * t_test / n_test.max(1) as f64, d_test as f64 / n_test.max(1) as f64, 1000.0 * t_opp / n_opp.max(1) as f64);
    (line, recs)
}

fn arena(args: &[String]) {
    let net = arg_s(args, "--net").map(|p| if p == "zero" { zero_net(arg(args, "--h", 32)) } else { load_net(&p) });
    let cfg = Cfg {
        net,
        test: arg_s(args, "--test").unwrap_or("nn".into()),
        opp: arg_s(args, "--opp").unwrap_or("greedy".into()),
        ms: arg(args, "--ms", 60.0),
        nn_depth: arg_s(args, "--nn-depth").and_then(|v| v.parse().ok()),
        greedy_depth: arg_s(args, "--greedy-depth").and_then(|v| v.parse().ok()),
        greedy_ms: arg_s(args, "--greedy-ms").and_then(|v| v.parse().ok()),
        qdepth: arg(args, "--qdepth", 2),
        check_ext: arg(args, "--check-ext", 0),
        width: arg(args, "--width", 0),
        full_plies: arg(args, "--full-plies", 2),
        max_plies: arg(args, "--max-plies", 400),
        opening_seed: arg(args, "--opening-seed", 0),
        opening_plies: arg(args, "--opening-plies", 4),
    };
    let games: usize = arg(args, "--games", 160);
    let first: usize = arg(args, "--first", 0);
    let threads: usize = arg(args, "--threads", 4);
    let out_games = arg_s(args, "--out-games");
    let out_pos = arg_s(args, "--out-pos");
    let gw = Mutex::new(out_games.map(|p| BufWriter::new(File::create(p).unwrap())));
    let pw = Mutex::new(out_pos.map(|p| BufWriter::new(File::create(p).unwrap())));
    let total = Mutex::new((0.0f64, 0usize));
    let done = AtomicUsize::new(0);
    let t0 = Instant::now();
    rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap().install(|| {
        (first..first + games).into_par_iter().for_each(|g| {
            let (line, recs) = play_game(g, &cfg);
            let pair: f64 = line.split("\"pairwise_of_3\":").nth(1).unwrap().split(',').next().unwrap().parse().unwrap();
            if let Some(w) = gw.lock().unwrap().as_mut() {
                writeln!(w, "{line}").unwrap();
                w.flush().unwrap();
            }
            if let Some(w) = pw.lock().unwrap().as_mut() {
                w.write_all(&recs).unwrap();
            }
            let mut t = total.lock().unwrap();
            t.0 += pair;
            t.1 += 1;
            let n = done.fetch_add(1, Ordering::SeqCst) + 1;
            if n % 20 == 0 || n == games {
                eprintln!("{n}/{games} games, mean {:.3} of 3, {:.0}s", t.0 / t.1 as f64, t0.elapsed().as_secs_f64());
            }
        });
    });
    let t = total.lock().unwrap();
    println!("{{\"games\":{},\"mean_pairwise_of_3\":{:.4},\"seconds\":{:.1}}}", t.1, t.0 / t.1 as f64, t0.elapsed().as_secs_f64());
}

/// Four greedy bots (each its own table) from the given opening moves, fixed depth.
fn greedy_game(args: &[String]) {
    let depth: u32 = arg(args, "--depth", 4);
    let plies: u32 = arg(args, "--plies", 400);
    let p = preset(&arg_s(args, "--preset").unwrap_or("greedy".into())).unwrap();
    let mut pos = Position::initial();
    let moves: Vec<Move> = args.iter().skip_while(|a| *a != "--moves").skip(1).map(|a| a.parse().unwrap()).collect();
    for &m in &moves {
        pos.push(m).unwrap();
    }
    let mut bots: Vec<Greedy> = (0..4).map(|_| Greedy::new(p, Some(depth), None)).collect();
    let mut out = vec![];
    while pos.game_over().is_none() && pos.turn < plies {
        let who = pos.current_player();
        let r = bots[who].best_move(&mut pos);
        out.push(format!("{}:{:.6}", r.best, r.value));
        apply(&mut pos, r.best);
    }
    println!("{}", out.join(" "));
    eprintln!("scores {:?} turn {}", pos.scores, pos.turn);
}

fn time_cmd(args: &[String]) {
    let depth: u32 = arg(args, "--max", 6);
    let mut pos = Position::initial();
    for (i, m) in opening(7, 24).into_iter().enumerate() {
        let _ = i;
        pos.push(m).unwrap();
    }
    for (name, mut p) in [("opening", Position::initial()), ("round6-random", pos)] {
        for d in 1..=depth {
            let mut g = Greedy::new(preset("greedy").unwrap(), Some(d), None);
            let r = g.best_move(&mut p);
            println!("{name} depth {d}: {:.4}s nodes {} ({:.0} knps)", r.seconds, r.nodes, r.nodes as f64 / r.seconds / 1000.0);
        }
    }
}

/// Logits of the first `--n` records of a position file (checks the export against torch).
fn eval_cmd(args: &[String]) {
    let net = load_net(&arg_s(args, "--net").unwrap());
    let mut buf = Vec::new();
    File::open(arg_s(args, "--pos").unwrap()).unwrap().read_to_end(&mut buf).unwrap();
    let n: usize = arg(args, "--n", 5);
    for r in buf.chunks_exact(REC).take(n) {
        let board: [u8; 64] = r[..64].try_into().unwrap();
        let k = r[64] as usize;
        let g: Vec<f32> = r[80..80 + 4 * net.n_globals].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        let mut acc = chaturaji_engine::nnue::AccStack::new(net.clone(), &board).unwrap();
        let l = acc.logits(k, &g);
        println!("{}", l.iter().map(|x| format!("{x:.5}")).collect::<Vec<_>>().join(" "));
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("arena") => arena(&args),
        Some("greedy-game") => greedy_game(&args),
        Some("time") => time_cmd(&args),
        Some("eval") => eval_cmd(&args),
        _ => eprintln!("usage: nnarena arena|greedy-game|time ..."),
    }
}

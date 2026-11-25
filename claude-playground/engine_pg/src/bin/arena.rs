//! Match runner: a test searcher (one spec) against a base searcher (another spec) on the
//! same network. Openings are paired: opening o = games 4o..4o+3, and in game g the test
//! player holds seat g % 4, so every opening is played from all four seats.
//!
//!   arena --weights <dir> --games N --threads T --seed S --out <file.jsonl>
//!         --test "<spec>" --base "<spec>"
//!
//! <dir> holds net.txt and net.bin (export_weights.py). A spec is a comma list of key=value:
//! depth, nodes, ms, width, full_plies, qdepth, check_ext, discount, elim_bonus, elim_horizon,
//! paranoid (0/1), pmin (0/1), psort (0/1), max_depth, tt_size. Give depth (fixed), nodes (a
//! node limit per move, iterative deepening to max_depth) or ms (a time budget per move in
//! milliseconds). Defaults are the self-play settings. The summary reports the time per move.
//!
//! Games run on T threads, each game single-threaded (rayon is set to one thread).

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use engine_pg::nnue::{Layer, Net, NUM_FEATURES};
use engine_pg::rules::{splitmix, Position};
use engine_pg::search::Searcher;

#[derive(Clone, Debug)]
struct Spec {
    depth: Option<u32>,
    nodes: Option<u64>,
    ms: Option<f64>,
    width: usize,
    full_plies: u32,
    qdepth: u32,
    check_ext: u32,
    discount: f64,
    elim_bonus: f64,
    elim_horizon: u32,
    paranoid: bool,
    paranoid_min_select: bool,
    paranoid_sort: bool,
    max_depth: u32,
    tt_size: usize,
}

impl Spec {
    fn parse(text: &str) -> Result<Spec, String> {
        let mut s = Spec {
            depth: None,
            nodes: None,
            ms: None,
            width: 4,
            full_plies: 1,
            qdepth: 2,
            check_ext: 3,
            discount: 0.9985,
            elim_bonus: 0.008,
            elim_horizon: 200,
            paranoid: false,
            paranoid_min_select: false,
            paranoid_sort: false,
            max_depth: 12,
            tt_size: 1_000_000,
        };
        for item in text.split(',').map(str::trim).filter(|x| !x.is_empty()) {
            let (k, v) = item.split_once('=').ok_or(format!("bad item {:?} (want key=value)", item))?;
            let (k, v) = (k.trim(), v.trim());
            let bad = |e: &dyn std::fmt::Display| format!("spec {:?}: bad value for {}: {} ({})", text, k, v, e);
            match k {
                "depth" => s.depth = Some(v.parse().map_err(|e| bad(&e))?),
                "nodes" => s.nodes = Some(v.parse().map_err(|e| bad(&e))?),
                "ms" => s.ms = Some(v.parse().map_err(|e| bad(&e))?),
                "width" => s.width = v.parse().map_err(|e| bad(&e))?,
                "full_plies" => s.full_plies = v.parse().map_err(|e| bad(&e))?,
                "qdepth" => s.qdepth = v.parse().map_err(|e| bad(&e))?,
                "check_ext" => s.check_ext = v.parse().map_err(|e| bad(&e))?,
                "discount" => s.discount = v.parse().map_err(|e| bad(&e))?,
                "elim_bonus" => s.elim_bonus = v.parse().map_err(|e| bad(&e))?,
                "elim_horizon" => s.elim_horizon = v.parse().map_err(|e| bad(&e))?,
                "max_depth" => s.max_depth = v.parse().map_err(|e| bad(&e))?,
                "tt_size" => s.tt_size = v.parse().map_err(|e| bad(&e))?,
                "paranoid" | "pmin" | "psort" => {
                    let on = match v {
                        "0" => false,
                        "1" => true,
                        _ => return Err(bad(&"want 0 or 1")),
                    };
                    match k {
                        "paranoid" => s.paranoid = on,
                        "pmin" => s.paranoid_min_select = on,
                        _ => s.paranoid_sort = on,
                    }
                }
                _ => return Err(format!("spec {:?}: unknown key {}", text, k)),
            }
        }
        if s.depth.is_none() && s.nodes.is_none() && s.ms.is_none() {
            return Err(format!("spec {:?}: give depth, nodes or ms", text));
        }
        Ok(s)
    }

    fn build(&self, net: &Arc<Net>) -> Result<Searcher, String> {
        // a time budget if given; else Searcher::new wants a depth, and with nodes only the budget is huge
        let time_ms = if self.ms.is_some() {
            self.ms
        } else if self.depth.is_none() {
            Some(1e9)
        } else {
            None
        };
        let mut s = Searcher::new(net.clone(), time_ms, self.depth, self.max_depth, self.qdepth, self.tt_size,
                                  self.check_ext, self.width, self.full_plies, self.discount, self.elim_bonus,
                                  self.elim_horizon)?;
        s.node_limit = self.nodes;
        s.paranoid = self.paranoid;
        s.paranoid_min_select = self.paranoid_min_select;
        s.paranoid_sort = self.paranoid_sort;
        Ok(s)
    }
}

/// net.txt: H, the number of hidden layers, "n_out n_in residual" per layer, then "n_out n_in"
/// for the output layer. net.bin: all arrays as little-endian float32 (ft_w, ft_b, then each
/// layer's w and b, then out_w and out_b).
fn load_net(dir: &Path) -> Result<Arc<Net>, String> {
    let txt = fs::read_to_string(dir.join("net.txt")).map_err(|e| format!("net.txt: {}", e))?;
    let nums: Vec<usize> = txt
        .split_whitespace()
        .map(|t| t.parse::<usize>())
        .collect::<Result<_, _>>()
        .map_err(|e| format!("net.txt: {}", e))?;
    if nums.len() < 2 {
        return Err("net.txt: too short".into());
    }
    let (h, nl) = (nums[0], nums[1]);
    let mut i = 2;
    let mut shapes = Vec::with_capacity(nl);
    for _ in 0..nl {
        if i + 3 > nums.len() {
            return Err("net.txt: truncated layer list".into());
        }
        shapes.push((nums[i], nums[i + 1], nums[i + 2] != 0));
        i += 3;
    }
    if i + 2 != nums.len() {
        return Err("net.txt: bad output layer line".into());
    }
    let out = (nums[i], nums[i + 1]);

    let bytes = fs::read(dir.join("net.bin")).map_err(|e| format!("net.bin: {}", e))?;
    let expected = NUM_FEATURES * h + h
        + shapes.iter().map(|&(o, n, _)| o * n + o).sum::<usize>()
        + out.0 * out.1 + out.0;
    if bytes.len() != 4 * expected {
        return Err(format!("net.bin: {} bytes, expected {}", bytes.len(), 4 * expected));
    }
    let floats: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let mut at = 0usize;
    let mut take = |n: usize| {
        let v = floats[at..at + n].to_vec();
        at += n;
        v
    };
    let ft_w = take(NUM_FEATURES * h);
    let ft_b = take(h);
    let mut layers = Vec::with_capacity(nl);
    for &(n_out, n_in, res) in &shapes {
        let w = take(n_out * n_in);
        let b = take(n_out);
        layers.push(Layer::new(w, b, n_in, n_out, res)?);
    }
    let out_w = take(out.0 * out.1);
    let out_b = take(out.0);
    let out_layer = Layer::new(out_w, out_b, out.1, out.0, false)?;
    Ok(Arc::new(Net::new(h, ft_w, ft_b, layers, out_layer)?))
}

/// The opening of opening index o: 4 random legal plies from the initial position.
fn opening(o: usize, seed: u64) -> Position {
    let mut pos = Position::initial();
    let mut s = splitmix(seed ^ splitmix(0x0A0C_0000 ^ o as u64));
    for _ in 0..4 {
        let moves = pos.legal_moves();
        if moves.is_empty() || pos.game_over().is_some() {
            break;
        }
        s = splitmix(s);
        let m = moves[(s % moves.len() as u64) as usize];
        pos.push(m).expect("legal move");
    }
    pos
}

struct GameResult {
    game: usize,
    opening: usize,
    seat: usize,
    scores: [i32; 4],
    turns: u32,
    score: f64,
    test_nodes: f64,
    base_nodes: f64,
    test_depth: f64,
    base_depth: f64,
    test_ms: f64,
    base_ms: f64,
    secs: f64,
}

impl GameResult {
    fn to_json(&self) -> String {
        format!(
            "{{\"game\":{},\"opening\":{},\"seat\":{},\"scores\":[{},{},{},{}],\"turns\":{},\"score\":{:.4},\
             \"test_nodes\":{:.1},\"base_nodes\":{:.1},\"test_depth\":{:.3},\"base_depth\":{:.3},\
             \"test_ms\":{:.2},\"base_ms\":{:.2},\"secs\":{:.3}}}",
            self.game, self.opening, self.seat, self.scores[0], self.scores[1], self.scores[2], self.scores[3],
            self.turns, self.score, self.test_nodes, self.base_nodes, self.test_depth, self.base_depth,
            self.test_ms, self.base_ms, self.secs
        )
    }
}

fn play(g: usize, net: &Arc<Net>, test: &Spec, base: &Spec, seed: u64) -> Result<GameResult, String> {
    let t0 = Instant::now();
    let (o, s) = (g / 4, g % 4);
    let mut pos = opening(o, seed);
    let mut searchers: Vec<Searcher> = (0..4)
        .map(|seat| if seat == s { test.build(net) } else { base.build(net) })
        .collect::<Result<_, _>>()?;
    // per side: total nodes, moves, total depth
    let (mut tn, mut tm, mut td) = (0u64, 0u64, 0u64);
    let (mut bn, mut bm, mut bd) = (0u64, 0u64, 0u64);
    // per side: total seconds spent in best_move
    let (mut tt_s, mut bt_s) = (0.0f64, 0.0f64);
    while pos.game_over().is_none() && pos.turn < 400 {
        let me = pos.current_player();
        let t_move = Instant::now();
        let r = searchers[me].best_move(&mut pos)?;
        let dt = t_move.elapsed().as_secs_f64();
        if me == s {
            tn += r.nodes;
            tm += 1;
            td += r.depth as u64;
            tt_s += dt;
        } else {
            bn += r.nodes;
            bm += 1;
            bd += r.depth as u64;
            bt_s += dt;
        }
        pos.push(r.best)?;
    }
    let scores = pos.scores;
    let mine = scores[s];
    let mut score = 0.0;
    for (j, &sc) in scores.iter().enumerate() {
        if j != s {
            if sc < mine {
                score += 1.0;
            } else if sc == mine {
                score += 0.5;
            }
        }
    }
    let avg = |sum: u64, n: u64| if n > 0 { sum as f64 / n as f64 } else { 0.0 };
    let ms_per_move = |secs: f64, n: u64| if n > 0 { 1000.0 * secs / n as f64 } else { 0.0 };
    Ok(GameResult {
        game: g,
        opening: o,
        seat: s,
        scores,
        turns: pos.turn,
        score: score / 3.0,
        test_nodes: avg(tn, tm),
        base_nodes: avg(bn, bm),
        test_depth: avg(td, tm),
        base_depth: avg(bd, bm),
        test_ms: ms_per_move(tt_s, tm),
        base_ms: ms_per_move(bt_s, bm),
        secs: t0.elapsed().as_secs_f64(),
    })
}

fn run() -> Result<(), String> {
    // each game is single-threaded; the games themselves run in parallel
    let _ = rayon::ThreadPoolBuilder::new().num_threads(1).build_global();

    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mut weights, mut games, mut threads, mut seed, mut out, mut test, mut base) =
        (None, None, 1usize, 1u64, None, None, None);
    let mut i = 0;
    while i < args.len() {
        let k = args[i].as_str();
        let v = args.get(i + 1).ok_or(format!("{} needs a value", k))?.clone();
        match k {
            "--weights" => weights = Some(v),
            "--games" => games = Some(v.parse::<usize>().map_err(|e| format!("--games: {}", e))?),
            "--threads" => threads = v.parse().map_err(|e| format!("--threads: {}", e))?,
            "--seed" => seed = v.parse().map_err(|e| format!("--seed: {}", e))?,
            "--out" => out = Some(v),
            "--test" => test = Some(v),
            "--base" => base = Some(v),
            _ => return Err(format!("unknown option {}", k)),
        }
        i += 2;
    }
    let weights = weights.ok_or("--weights is required")?;
    let games = games.ok_or("--games is required")?;
    let out = out.ok_or("--out is required")?;
    let test_text = test.ok_or("--test is required")?;
    let base_text = base.ok_or("--base is required")?;
    let test_spec = Spec::parse(&test_text)?;
    let base_spec = Spec::parse(&base_text)?;
    let threads = threads.max(1);

    let net = load_net(Path::new(&weights))?;
    if let Some(parent) = Path::new(&out).parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent).map_err(|e| format!("{}: {}", parent.display(), e))?;
    }
    let file = OpenOptions::new().create(true).append(true).open(&out).map_err(|e| format!("{}: {}", out, e))?;
    let file = Mutex::new(file);

    eprintln!(
        "arena: {} games, {} threads, seed {}, test [{}], base [{}], net h={} ({} params)",
        games, threads, seed, test_text, base_text, net.h, net.num_params()
    );
    let t0 = Instant::now();
    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let results: Mutex<Vec<GameResult>> = Mutex::new(Vec::new());
    let errors: Mutex<Vec<String>> = Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| loop {
                let g = next.fetch_add(1, Ordering::SeqCst);
                if g >= games {
                    break;
                }
                match play(g, &net, &test_spec, &base_spec, seed) {
                    Ok(r) => {
                        {
                            let mut f = file.lock().unwrap();
                            writeln!(f, "{}", r.to_json()).unwrap();
                            f.flush().unwrap();
                        }
                        let n = done.fetch_add(1, Ordering::SeqCst) + 1;
                        eprintln!(
                            "[{}/{}] game {} opening {} seat {}: score {:.3} turns {} test nodes {:.0} depth {:.2} {:.1} ms | base nodes {:.0} depth {:.2} {:.1} ms | {:.1}s",
                            n, games, r.game, r.opening, r.seat, r.score, r.turns, r.test_nodes, r.test_depth,
                            r.test_ms, r.base_nodes, r.base_depth, r.base_ms, r.secs
                        );
                        results.lock().unwrap().push(r);
                    }
                    Err(e) => errors.lock().unwrap().push(format!("game {}: {}", g, e)),
                }
            });
        }
    });

    let mut res = results.into_inner().unwrap();
    res.sort_by_key(|r| r.game);
    let n = res.len();
    if n > 0 {
        let nf = n as f64;
        let mean = res.iter().map(|r| r.score).sum::<f64>() / nf;
        let var = if n > 1 { res.iter().map(|r| (r.score - mean).powi(2)).sum::<f64>() / (nf - 1.0) } else { 0.0 };
        let two_se = 2.0 * (var / nf).sqrt();
        let mean_of = |f: fn(&GameResult) -> f64| res.iter().map(f).sum::<f64>() / nf;
        println!("games {}  secs {:.1}", n, t0.elapsed().as_secs_f64());
        println!("test score {:.4} +- {:.4} (2 SE)   [{}]", mean, two_se, test_text);
        println!(
            "test: nodes/move {:.0}  depth {:.2}    base [{}]: nodes/move {:.0}  depth {:.2}",
            mean_of(|r| r.test_nodes), mean_of(|r| r.test_depth), base_text,
            mean_of(|r| r.base_nodes), mean_of(|r| r.base_depth)
        );
        let (test_ms, base_ms) = (mean_of(|r| r.test_ms), mean_of(|r| r.base_ms));
        let (test_nodes, base_nodes) = (mean_of(|r| r.test_nodes), mean_of(|r| r.base_nodes));
        let us_per_node = |ms: f64, nodes: f64| if nodes > 0.0 { 1000.0 * ms / nodes } else { 0.0 };
        println!(
            "time/move: test {:.1} ms  base {:.1} ms   us/node: test {:.2}  base {:.2}",
            test_ms, base_ms, us_per_node(test_ms, test_nodes), us_per_node(base_ms, base_nodes)
        );
        print!("test score by seat:");
        for s in 0..4 {
            let xs: Vec<f64> = res.iter().filter(|r| r.seat == s).map(|r| r.score).collect();
            if xs.is_empty() {
                print!("  seat {} -", s);
            } else {
                print!("  seat {} {:.3} (n={})", s, xs.iter().sum::<f64>() / xs.len() as f64, xs.len());
            }
        }
        println!();
    }
    let errs = errors.into_inner().unwrap();
    for e in &errs {
        eprintln!("error: {}", e);
    }
    if !errs.is_empty() {
        return Err(format!("{} games failed", errs.len()));
    }
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("arena: {}", e);
        std::process::exit(2);
    }
}

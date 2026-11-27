//! Player specs shared by the arena and the self-play generator: the network/table searcher
//! (engine_pg's options, plus `nn2=<file>` for NNUE-T) or the greedy bot (`greedy=<preset>`,
//! `g_<key>=<value>`, see greedy.rs). Moved here from bin/arena.rs unchanged otherwise.

use std::fs;
use std::path::Path;
use std::sync::Arc;

use crate::greedy::{GreedyBot, GreedyParams};
use crate::nn2::Nn2;
use crate::nnue::{Layer, Net, NUM_FEATURES};
use crate::policy::Policy;
use crate::pst::Pst;
use crate::pst2::Pst2;
use crate::rules::Position;
use crate::search::{Searcher, Table};

#[derive(Clone, Debug)]
pub struct Spec {
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
    table: Option<Table>,
    pst_w: Option<f32>,
    table_only: bool,
    check_pref: f64,
    policy: Option<Arc<Policy>>,
    policy_width: bool,
    policy_order: bool,
    greedy: Option<GreedyParams>,
}

/// A player: the network/table searcher or the greedy bot.
pub enum Player {
    Search(Searcher),
    Greedy(GreedyBot),
}

pub struct Choice {
    pub best: u16,
    pub nodes: u64,
    pub depth: u32,
}

impl Player {
    pub fn choose(&mut self, pos: &mut Position) -> Result<Choice, String> {
        match self {
            Player::Search(s) => {
                let r = s.best_move(pos)?;
                Ok(Choice { best: r.best, nodes: r.nodes, depth: r.depth })
            }
            Player::Greedy(g) => {
                let r = g.best_move(pos);
                Ok(Choice { best: r.best, nodes: r.nodes, depth: r.depth })
            }
        }
    }
}

impl Spec {
    pub fn parse(text: &str) -> Result<Spec, String> {
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
            table: None,
            pst_w: None,
            table_only: false,
            check_pref: 0.0,
            policy: None,
            policy_width: false,
            policy_order: false,
            greedy: None,
        };
        let mut greedy_keys: Vec<(String, String)> = Vec::new();
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
                "nn2" => {
                    if s.table.is_some() {
                        return Err(format!("spec {:?}: one table (pst, pst2 or nn2)", text));
                    }
                    s.table = Some(Table::N2(Arc::new(Nn2::load(Path::new(v))?)));
                }
                "pst" | "pst2" => {
                    if s.table.is_some() {
                        return Err(format!("spec {:?}: give pst or pst2, not both", text));
                    }
                    s.table = Some(if k == "pst" {
                        Table::V1(Arc::new(Pst::load(Path::new(v))?))
                    } else {
                        Table::V2(Arc::new(Pst2::load(Path::new(v))?))
                    });
                }
                "pst_w" => s.pst_w = Some(v.parse().map_err(|e| bad(&e))?),
                "check_pref" => s.check_pref = v.parse().map_err(|e| bad(&e))?,
                "policy" => s.policy = Some(Arc::new(Policy::load(Path::new(v))?)),
                "greedy" => s.greedy = Some(GreedyParams::preset(v)?),
                _ if k.starts_with("g_") => greedy_keys.push((k[2..].to_string(), v.to_string())),
                "paranoid" | "pmin" | "psort" | "pst_only" | "policy_width" | "policy_order" => {
                    let on = match v {
                        "0" => false,
                        "1" => true,
                        _ => return Err(bad(&"want 0 or 1")),
                    };
                    match k {
                        "paranoid" => s.paranoid = on,
                        "pmin" => s.paranoid_min_select = on,
                        "psort" => s.paranoid_sort = on,
                        "pst_only" => s.table_only = on,
                        "policy_width" => s.policy_width = on,
                        _ => s.policy_order = on,
                    }
                }
                _ => return Err(format!("spec {:?}: unknown key {}", text, k)),
            }
        }
        if let Some(g) = s.greedy.as_mut() {
            for (k, v) in &greedy_keys {
                g.set(k, v)?;
            }
            if let Some(Table::V2(t)) = &s.table {
                g.pst2 = Some(t.clone());
            }
            if g.tiebreak != 0.0 && g.pst2.is_none() {
                return Err(format!("spec {:?}: g_tiebreak needs pst2", text));
            }
            return Ok(s);
        } else if !greedy_keys.is_empty() {
            return Err(format!("spec {:?}: g_ keys need greedy=<preset>", text));
        }
        if s.depth.is_none() && s.nodes.is_none() && s.ms.is_none() {
            return Err(format!("spec {:?}: give depth, nodes or ms", text));
        }
        if s.table.is_none() && (s.pst_w.is_some() || s.table_only) {
            return Err(format!("spec {:?}: pst_w and pst_only need pst, pst2 or nn2", text));
        }
        if s.policy.is_none() && (s.policy_width || s.policy_order) {
            return Err(format!("spec {:?}: policy_width and policy_order need policy", text));
        }
        Ok(s)
    }

    pub fn build(&self, net: &Arc<Net>) -> Result<Player, String> {
        if let Some(g) = &self.greedy {
            let mut b = GreedyBot::new(g.clone());
            b.fixed_depth = self.depth;
            b.time_ms = self.ms;
            b.node_limit = self.nodes;
            b.max_depth = self.max_depth;
            return Ok(Player::Greedy(b));
        }
        self.build_searcher(net).map(Player::Search)
    }

    fn build_searcher(&self, net: &Arc<Net>) -> Result<Searcher, String> {
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
        s.table = self.table.clone();
        s.table_w = self.pst_w.unwrap_or(1.0);
        s.table_only = self.table_only;
        s.check_pref = self.check_pref;
        s.policy = self.policy.clone();
        s.policy_width = self.policy_width;
        s.policy_order = self.policy_order;
        Ok(s)
    }
}

/// net.txt: H, the number of hidden layers, "n_out n_in residual" per layer, then "n_out n_in"
/// for the output layer. net.bin: all arrays as little-endian float32 (ft_w, ft_b, then each
/// layer's w and b, then out_w and out_b).
pub fn load_net(dir: &Path) -> Result<Arc<Net>, String> {
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


//! `chaturaji_engine`: the compiled core of the Chaturaji bot.
//!
//! - `nnue.rs`: NNUE inference with incremental accumulators, exposed to Python as
//!   `chaturaji_engine.Nnue` and `chaturaji_engine.Accumulator`. `bot/nnue.py` uses it
//!   automatically when the module is installed.
//! - `rules.rs`: the game rules (move generation, make/unmake, scoring, elimination, game
//!   end, the network's global features), exposed as `chaturaji_engine.Position`.
//!   `bot/rules_core.py` converts to and from the Python rules engine and checks the two
//!   agree on every scraped game.
//! - `search.rs`: the max^n search over both (a port of bot/maxn.py), exposed as
//!   `chaturaji_engine.MaxN`. bot/maxn.py uses it automatically.

pub mod nnue;
pub mod rules;
pub mod search;

#[cfg(feature = "python")]
mod python {
    use std::sync::Arc;

    use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
    use pyo3::exceptions::{PyIndexError, PyValueError};
    use pyo3::prelude::*;

    use crate::nnue::{AccStack, Layer, Net, NUM_GLOBALS};
    use crate::rules::{self, GameOver, Position};
    use crate::search::{Searcher, RESIGN};

    fn vec1(a: &PyReadonlyArray1<f32>) -> Vec<f32> {
        a.as_array().iter().copied().collect()
    }

    fn vec2(a: &PyReadonlyArray2<f32>) -> (Vec<f32>, usize, usize) {
        let s = a.shape();
        (a.as_array().iter().copied().collect(), s[0], s[1])
    }

    fn board(a: &PyReadonlyArray1<u8>) -> PyResult<[u8; 64]> {
        let v = a.as_array();
        if v.len() != 64 {
            return Err(PyValueError::new_err(format!("board must have 64 squares, got {}", v.len())));
        }
        let mut b = [0u8; 64];
        for (d, s) in b.iter_mut().zip(v.iter()) {
            *d = *s;
        }
        Ok(b)
    }

    fn globals(a: &PyReadonlyArray1<f32>) -> PyResult<Vec<f32>> {
        let g = vec1(a);
        if g.len() != NUM_GLOBALS {
            return Err(PyValueError::new_err(format!("expected {} globals, got {}", NUM_GLOBALS, g.len())));
        }
        Ok(g)
    }

    fn err(e: String) -> PyErr {
        PyValueError::new_err(e)
    }

    /// The network's weights (shapes as in bot/nnue.py's .npz files).
    #[pyclass(name = "Nnue", module = "chaturaji_engine", frozen)]
    pub struct PyNnue {
        net: Arc<Net>,
    }

    #[pymethods]
    impl PyNnue {
        #[new]
        #[allow(clippy::too_many_arguments)]
        fn new(
            ft_w: PyReadonlyArray2<f32>,
            ft_b: PyReadonlyArray1<f32>,
            layer_w: Vec<PyReadonlyArray2<f32>>,
            layer_b: Vec<PyReadonlyArray1<f32>>,
            residual: Vec<bool>,
            out_w: PyReadonlyArray2<f32>,
            out_b: PyReadonlyArray1<f32>,
        ) -> PyResult<Self> {
            if layer_w.len() != layer_b.len() || layer_w.len() != residual.len() {
                return Err(PyValueError::new_err("layer_w, layer_b and residual must have the same length"));
            }
            let (fw, rows, h) = vec2(&ft_w);
            if rows != crate::nnue::NUM_FEATURES {
                return Err(PyValueError::new_err(format!("ft_w must have 1600 rows, got {}", rows)));
            }
            let mut layers = Vec::with_capacity(layer_w.len());
            for ((w, b), r) in layer_w.iter().zip(&layer_b).zip(&residual) {
                let (wv, n_out, n_in) = vec2(w);
                layers.push(Layer::new(wv, vec1(b), n_in, n_out, *r).map_err(err)?);
            }
            let (ow, o_out, o_in) = vec2(&out_w);
            let out = Layer::new(ow, vec1(&out_b), o_in, o_out, false).map_err(err)?;
            Ok(PyNnue { net: Arc::new(Net::new(h, fw, vec1(&ft_b), layers, out).map_err(err)?) })
        }

        #[getter]
        fn acc_size(&self) -> usize {
            self.net.h
        }

        #[getter]
        fn num_params(&self) -> usize {
            self.net.num_params()
        }

        /// A new accumulator stack rooted at this board (64 piece codes).
        fn accumulator(&self, board_codes: PyReadonlyArray1<u8>) -> PyResult<PyAccumulator> {
            let b = board(&board_codes)?;
            Ok(PyAccumulator { stack: AccStack::new(self.net.clone(), &b).map_err(err)? })
        }

        /// (4, H) accumulators of a board, from scratch (absolute seats).
        fn refresh<'py>(&self, py: Python<'py>, board_codes: PyReadonlyArray1<u8>) -> PyResult<Bound<'py, PyArray2<f32>>> {
            let b = board(&board_codes)?;
            let mut acc = vec![0.0f32; 4 * self.net.h];
            self.net.refresh(&b, &mut acc).map_err(err)?;
            to_array(py, &acc, self.net.h)
        }

        /// 6 pairwise logits from (4, H) accumulators, mover k and 16 globals.
        fn forward(&self, acc: PyReadonlyArray2<f32>, k: usize, globals_: PyReadonlyArray1<f32>) -> PyResult<Vec<f32>> {
            let (a, rows, h) = vec2(&acc);
            if rows != 4 || h != self.net.h || k > 3 {
                return Err(PyValueError::new_err("acc must be (4, H) and k in 0..4"));
            }
            let g = globals(&globals_)?;
            let mut scratch = Vec::new();
            Ok(self.net.forward(&a, k, &g, &mut scratch).to_vec())
        }
    }

    fn to_array<'py>(py: Python<'py>, acc: &[f32], h: usize) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let rows: Vec<Vec<f32>> = (0..4).map(|p| acc[p * h..(p + 1) * h].to_vec()).collect();
        PyArray2::from_vec2(py, &rows).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Accumulators that follow a search: push(board) after a move, pop() to undo it.
    #[pyclass(name = "Accumulator", module = "chaturaji_engine")]
    pub struct PyAccumulator {
        stack: AccStack,
    }

    #[pymethods]
    impl PyAccumulator {
        fn push(&mut self, board_codes: PyReadonlyArray1<u8>) -> PyResult<()> {
            let b = board(&board_codes)?;
            self.stack.push(&b).map_err(err)
        }

        fn pop(&mut self) -> PyResult<()> {
            self.stack.pop().map_err(PyIndexError::new_err)
        }

        #[getter]
        fn depth(&self) -> usize {
            self.stack.depth()
        }

        /// The current (4, H) accumulators, absolute seats.
        fn acc<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
            to_array(py, self.stack.acc(), self.stack.net.h)
        }

        /// 6 pairwise logits for mover k (0..4, absolute) with its 16 globals.
        fn logits(&mut self, k: usize, globals_: PyReadonlyArray1<f32>) -> PyResult<Vec<f32>> {
            if k > 3 {
                return Err(PyValueError::new_err("k must be in 0..4"));
            }
            let g = globals(&globals_)?;
            Ok(self.stack.logits(k, &g).to_vec())
        }

        /// (n, 6) logits of n child positions of the current one, in one batched pass.
        /// boards: (n, 64) piece codes, ks: the mover of each child, globals_: (n, 16).
        fn child_logits<'py>(
            &self,
            py: Python<'py>,
            boards: PyReadonlyArray2<u8>,
            ks: Vec<usize>,
            globals_: PyReadonlyArray2<f32>,
        ) -> PyResult<Bound<'py, PyArray2<f32>>> {
            let bs = boards.shape();
            let gs = globals_.shape();
            if bs[1] != 64 || bs[0] != ks.len() || gs[0] != ks.len() || gs[1] != NUM_GLOBALS || ks.iter().any(|&k| k > 3) {
                return Err(PyValueError::new_err("boards (n, 64), ks (n, each 0..4) and globals (n, 16) must match"));
            }
            let flat: Vec<u8> = boards.as_array().iter().copied().collect();
            let bv: Vec<[u8; 64]> = flat.chunks_exact(64).map(|c| c.try_into().unwrap()).collect();
            let g: Vec<f32> = globals_.as_array().iter().copied().collect();
            let out = self.stack.child_logits(&bv, &ks, &g).map_err(err)?;
            let mut flat_out = Vec::with_capacity(out.len() * 6);
            for o in &out {
                flat_out.extend_from_slice(o);
            }
            numpy::ndarray::Array2::from_shape_vec((out.len(), 6), flat_out)
                .map(|a| numpy::PyArray2::from_owned_array(py, a))
                .map_err(|e| PyValueError::new_err(e.to_string()))
        }
    }

    /// Parse a chaturaji board_key ("/"-joined rows of 2-character squares, ".." = empty).
    fn parse_board_key(key: &str) -> PyResult<[u8; 64]> {
        let rows: Vec<&str> = key.split('/').collect();
        let bad = || PyValueError::new_err(format!("bad board key {:?}", key));
        if rows.len() != 8 {
            return Err(bad());
        }
        let mut b = [0u8; 64];
        for (r, row) in rows.iter().enumerate() {
            let bytes = row.as_bytes();
            if bytes.len() != 16 {
                return Err(bad());
            }
            for c in 0..8 {
                let (o, t) = (bytes[2 * c], bytes[2 * c + 1]);
                if o == b'.' {
                    continue;
                }
                let owner = b"RBYGX".iter().position(|&x| x == o).ok_or_else(bad)?;
                let ty = b"KBNRP".iter().position(|&x| x == t).ok_or_else(bad)?;
                b[r * 8 + c] = rules::code(owner, ty);
            }
        }
        Ok(b)
    }

    fn over_name(o: Option<GameOver>) -> Option<&'static str> {
        match o {
            Some(GameOver::LastKing) => Some("last_king"),
            Some(GameOver::Repetition) => Some("repetition"),
            None => None,
        }
    }

    /// A game position with make/unmake (the rules in engine/src/rules.rs). Seats are
    /// 0..4 = R, B, Y, G; moves are from_square * 64 + to_square (square = row * 8 + col).
    #[pyclass(name = "Position", module = "chaturaji_engine")]
    #[derive(Clone)]
    pub struct PyPosition {
        pos: Position,
    }

    #[pymethods]
    impl PyPosition {
        #[staticmethod]
        fn initial() -> Self {
            PyPosition { pos: Position::initial() }
        }

        /// From a chaturaji state's parts: 64 piece codes, turn, 4 scores (R B Y G),
        /// eliminated seats and position_counts as (board_key, count) pairs.
        #[staticmethod]
        fn from_parts(board_codes: PyReadonlyArray1<u8>, turn: u32, scores: Vec<i32>, eliminated: Vec<usize>,
                      counts: Vec<(String, u32)>) -> PyResult<Self> {
            let b = board(&board_codes)?;
            if scores.len() != 4 || eliminated.iter().any(|&c| c > 3) {
                return Err(PyValueError::new_err("scores must have 4 entries and seats be 0..4"));
            }
            let mut parsed = Vec::with_capacity(counts.len());
            for (k, n) in &counts {
                parsed.push((parse_board_key(k)?, *n));
            }
            let mask = eliminated.iter().fold(0u8, |m, &c| m | (1 << c));
            let pos = Position::from_parts(b, turn, [scores[0], scores[1], scores[2], scores[3]], mask, parsed).map_err(err)?;
            Ok(PyPosition { pos })
        }

        fn copy(&self) -> Self {
            self.clone()
        }

        fn legal_moves(&self) -> Vec<u16> {
            self.pos.legal_moves()
        }

        /// Make a move: returns (points gained incl. end bonus, captured code, check bonus,
        /// eliminated seat or None).
        fn push(&mut self, m: u16) -> PyResult<(i32, u8, i32, Option<usize>)> {
            let i = self.pos.push(m).map_err(err)?;
            Ok((i.points, i.captured, i.check_bonus, i.eliminated))
        }

        /// Resignation or timeout of a seat (undo with pop).
        fn eliminate(&mut self, color: usize) -> PyResult<()> {
            if color > 3 {
                return Err(PyValueError::new_err("seat must be 0..4"));
            }
            self.pos.eliminate(color);
            Ok(())
        }

        fn pop(&mut self) -> PyResult<()> {
            self.pos.pop().map_err(PyIndexError::new_err)
        }

        fn capture_points(&self, m: u16) -> i32 {
            self.pos.capture_points(m)
        }

        /// Whether a move would newly attack an active king (the position is unchanged).
        fn gives_check(&mut self, m: u16) -> PyResult<bool> {
            let c = self.pos.push(m).map_err(err)?.gives_check;
            self.pos.pop().map_err(PyIndexError::new_err)?;
            Ok(c)
        }

        /// The 16 network globals from seat k's view (bot.mlp.global_features).
        fn globals<'py>(&self, py: Python<'py>, k: usize) -> Bound<'py, numpy::PyArray1<f32>> {
            numpy::PyArray1::from_slice(py, &self.pos.globals(k % 4))
        }

        #[getter]
        fn board<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<u8>> {
            numpy::PyArray1::from_slice(py, &self.pos.board)
        }

        #[getter]
        fn turn(&self) -> u32 {
            self.pos.turn
        }

        #[getter]
        fn scores(&self) -> Vec<i32> {
            self.pos.scores.to_vec()
        }

        #[getter]
        fn eliminated(&self) -> Vec<usize> {
            (0..4).filter(|&c| !self.pos.is_active(c)).collect()
        }

        #[getter]
        fn current_player(&self) -> usize {
            self.pos.current_player()
        }

        #[getter]
        fn game_over(&self) -> Option<&'static str> {
            over_name(self.pos.game_over())
        }

        #[getter]
        fn repetitions(&self) -> u32 {
            self.pos.repetitions()
        }

        #[getter]
        fn depth(&self) -> usize {
            self.pos.depth()
        }

        #[getter]
        fn hash(&self) -> u64 {
            self.pos.hash
        }
    }

    /// max^n search (engine/src/search.rs). Give `time_ms` (iterative deepening until the
    /// time runs out) or a fixed `depth`. The transposition table persists across moves.
    #[pyclass(name = "MaxN", module = "chaturaji_engine")]
    pub struct PyMaxN {
        s: Searcher,
    }

    #[pymethods]
    impl PyMaxN {
        #[new]
        #[pyo3(signature = (net, time_ms=None, depth=None, max_depth=12, qdepth=2, tt_size=200_000, check_ext=0, width=0, full_plies=2, discount=1.0, elim_bonus=0.0, elim_horizon=200, resign=true))]
        fn new(net: &PyNnue, time_ms: Option<f64>, depth: Option<u32>, max_depth: u32, qdepth: u32, tt_size: usize,
               check_ext: u32, width: usize, full_plies: u32, discount: f64, elim_bonus: f64,
               elim_horizon: u32, resign: bool) -> PyResult<Self> {
            Ok(PyMaxN { s: Searcher::new(net.net.clone(), time_ms, depth, max_depth, qdepth, tt_size, check_ext, width, full_plies, discount, elim_bonus, elim_horizon, resign).map_err(err)? })
        }

        /// Search a position: returns (move, depth reached, nodes, root utilities R B Y G or
        /// None, forced, seconds). The move is `RESIGN` (65535) when resigning is best. The
        /// position is left unchanged.
        #[allow(clippy::type_complexity)]
        fn best_move(&mut self, py: Python<'_>, position: &PyPosition)
                     -> PyResult<(u16, u32, u64, Option<Vec<f64>>, bool, f64)> {
            let mut pos = position.pos.clone();
            let s = &mut self.s;
            let r = py.detach(move || s.best_move(&mut pos)).map_err(err)?;
            Ok((r.best, r.depth, r.nodes, r.value.map(|v| v.to_vec()), r.forced, r.seconds))
        }

        fn clear(&mut self) {
            self.s.clear();
        }
    }

    #[pymodule]
    fn chaturaji_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<PyNnue>()?;
        m.add_class::<PyAccumulator>()?;
        m.add_class::<PyPosition>()?;
        m.add_class::<PyMaxN>()?;
        m.add("RESIGN", RESIGN)?;
        m.add("__version__", env!("CARGO_PKG_VERSION"))?;
        Ok(())
    }
}

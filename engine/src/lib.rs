//! `chaturaji_engine`: the compiled core of the Chaturaji bot.
//!
//! So far: NNUE inference (`nnue.rs`) with incremental accumulators, exposed to Python as
//! `chaturaji_engine.Nnue` and `chaturaji_engine.Accumulator`. `bot/nnue.py` uses it
//! automatically when the module is installed. Next: move generation and the max^n search.

pub mod nnue;

#[cfg(feature = "python")]
mod python {
    use std::sync::Arc;

    use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
    use pyo3::exceptions::{PyIndexError, PyValueError};
    use pyo3::prelude::*;

    use crate::nnue::{AccStack, Layer, Net, NUM_GLOBALS};

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

    #[pymodule]
    fn chaturaji_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<PyNnue>()?;
        m.add_class::<PyAccumulator>()?;
        m.add("__version__", env!("CARGO_PKG_VERSION"))?;
        Ok(())
    }
}

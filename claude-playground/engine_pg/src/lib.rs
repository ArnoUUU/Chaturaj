//! `engine_pg`: a playground copy of the Chaturaji engine core (NNUE inference, the rules,
//! max^n search) with a paranoid-search option and a match runner (`src/bin/arena.rs`).
//! No Python bindings.

pub mod nnue;
pub mod rules;
pub mod search;

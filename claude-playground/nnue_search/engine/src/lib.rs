//! `engine_pg`: a playground copy of the Chaturaji engine core (NNUE inference, the rules,
//! max^n search) with a paranoid-search option and a match runner (`src/bin/arena.rs`).
//! No Python bindings.

pub mod nnue;
pub mod policy;
pub mod pst;
pub mod pst2;
pub mod rules;
pub mod search;
pub mod greedy;
pub mod nn2;
pub mod players;

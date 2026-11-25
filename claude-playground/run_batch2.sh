#!/usr/bin/env bash
# Round 2: is the paranoid result real, and can it be improved? Same arena settings as round 1
# (48 games, 4 game threads, 1 rayon thread each, seed 7 = the same openings), round-2 binary.
cd "$(dirname "$0")/engine_pg"
export RAYON_NUM_THREADS=1
A=./target/b2/release/arena
W=../weights/nnue-v1
run() { name=$1; shift; echo "== $name" >&2; $A --weights $W --games 48 --threads 4 --seed 7 --out ../results/$name.jsonl "$@" 2> ../results/$name.log; tail -3 ../results/$name.log; }
run f1_paranoid_ms500   --test "ms=500,paranoid=1"               --base "ms=500"
run f2_psort_n10k       --test "nodes=10000,paranoid=1,psort=1"  --base "nodes=10000"
run f3_pmin_n10k        --test "nodes=10000,paranoid=1,pmin=1"   --base "nodes=10000"
run f4_paranoid_w0_n10k --test "nodes=10000,paranoid=1,width=0"  --base "nodes=10000"
run f5_paranoid_n20k    --test "nodes=20000,paranoid=1"          --base "nodes=20000"

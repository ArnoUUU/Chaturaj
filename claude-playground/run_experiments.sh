#!/usr/bin/env bash
# Runs the playground experiments one after another (4 game threads, 1 rayon thread each).
cd "$(dirname "$0")/engine_pg"
export RAYON_NUM_THREADS=1
A=./target/release/arena
W=../weights/nnue-v1
run() { name=$1; shift; echo "== $name" >&2; $A --weights $W --games 48 --threads 4 --seed 7 --out ../results/$name.jsonl "$@" 2> ../results/$name.log; tail -5 ../results/$name.log; }
run e1_paranoid_n10k   --test "nodes=10000,paranoid=1" --base "nodes=10000"
run e2_width6_n10k     --test "nodes=10000,width=6"    --base "nodes=10000"
run e3_width3_n10k     --test "nodes=10000,width=3"    --base "nodes=10000"
run e4_qdepth4_n10k    --test "nodes=10000,qdepth=4"   --base "nodes=10000"

#!/bin/bash
# Greedy ideas (DESIGN.md section 6): each variant as 1 test vs 3 copies of full greedy,
# 96 games (24 openings x 4 seats), 100 ms a move, 12 threads. Skips finished matches.
#   ./scripts/run_greedy_ideas.sh [ms] [games] [threads]
cd "$(dirname "$0")/.."
MS=${1:-100}; GAMES=${2:-96}; THREADS=${3:-12}
A=./engine/target/b1/release/arena.exe
W=../weights/nnue-v1
T2=../best_models/1_paranoid_table_v2/pst2.txt
G="greedy=greedy,ms=$MS"
TAB="ms=$MS,paranoid=1,pst2=$T2,pst_only=1"
run() {  # name test base
  out=results/greedy/$1
  if [ -f "$out.summary" ]; then echo "skip $1"; return; fi
  rm -f "$out.jsonl"
  RAYON_NUM_THREADS=1 $A --weights $W --games $GAMES --threads $THREADS --seed 7 --out "$out.jsonl" --test "$2" --base "$3" > "$out.summary.tmp" 2> "$out.log" \
    && mv "$out.summary.tmp" "$out.summary" && echo "$1: $(grep 'test score' $out.summary)"
}
run g02_no_threat      "$G,g_threat=0"            "$G"
run g02_loss_w1        "$G,g_loss_weight=1.0"     "$G"
run g02_no_king_bonus  "$G,g_king_bonus=0"        "$G"
run g02_no_pawn_loss   "$G,g_pawn_loss=0"         "$G"
run g02_no_endgame     "$G,g_endgame_loss=0"      "$G"
run g02_raw_points     "$G,g_placement=0"         "$G"
run g05_one_capture    "$G,g_one_capture=1"       "$G"
run g07_king_safety03  "$G,g_king_safety=0.3"     "$G"
run g07_king_safety01  "$G,g_king_safety=0.1"     "$G"
run g09_king_floor10   "$G,g_king_floor=10"       "$G"
run g06_offense04      "$G,g_offense=0.4"         "$G"
run g10_material01     "$G,g_material=0.1"        "$G"
run g14_tiebreak1      "$G,pst2=$T2,g_tiebreak=1" "$G"
run g01_greedy_vs_table "$G"                      "$TAB"
run g01_table_vs_greedy "$TAB"                    "$G"
echo GREEDY_IDEAS_DONE

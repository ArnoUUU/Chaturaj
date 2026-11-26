// Greedy bot: a port of bot/greedy.py (hand-written "value to go" eval under paranoid
// alpha-beta). Same terms, same move ordering, same transposition-table policy, so at a
// fixed depth it plays the same moves with the same values and node counts as the Python
// bot (checked by tools/greedy_ref.py + `arena greedycheck`). This is idea G15.
#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <unordered_map>
#include "rules.h"

namespace cr {

struct GreedyParams {
    double loss_weight = 1.2;
    double king_bonus = 4.0;
    int king_bonus_rounds = 10;
    double pawn_loss = 2.0;
    int pawn_loss_rounds = 15;
    double endgame_loss = 1.0;
    int endgame_pawn_dist = 2;
    int king_loss_rounds = 30;
    double king_floor = 0.0;
    double trade_bonus = 0.0;
    int trade_rounds = 1000;
    bool threat = true;
    double push_bonus = 0.0;
    int push_from = 15;
    int depth = 4;
    int qdepth = 4;
    double threat_p[3] = {0.9, 0.6, 0.4};
    double placement[4] = {150.0, -20.0, -40.0, -100.0};
};

inline GreedyParams greedy_preset(const std::string& name) {
    GreedyParams p;
    if (name == "greedy-trade") p.trade_bonus = 1.0;
    else if (name == "naive") {
        p.loss_weight = 1.0; p.king_bonus = 0; p.pawn_loss = 0; p.endgame_loss = 0;
        p.trade_bonus = 3.0; p.trade_rounds = 15; p.threat = false; p.push_bonus = 0.6; p.depth = 2; p.qdepth = 0;
    }
    return p;
}

inline int pawn_to_go(int color, int sq) {
    int r = sq / 8, c = sq % 8, dr = PAWN_DIR[color][0], dc = PAWN_DIR[color][1];
    if (dr) return dr > 0 ? 7 - r : r;
    return dc > 0 ? 7 - c : c;
}

inline bool endgame_now(const Board& b) {
    for (int s = 0; s < 64; s++) {
        uint8_t p = b[s];
        if (p && owner_of(p) != DEAD && type_of(p) == PAWN && pawn_to_go(owner_of(p), s) <= 2) return true;
    }
    return false;
}

inline double placement_value(const int sc[4], int me, const double placement[4]) {
    int better = 0, tied = 0;
    for (int c = 0; c < 4; c++)
        if (c != me) { better += sc[c] > sc[me]; tied += sc[c] == sc[me]; }
    double s = 0;
    for (int i = better; i <= better + tied; i++) s += placement[i];
    return s / (tied + 1);
}

inline int live_kings(const Board& b) {
    int n = 0;
    for (int s = 0; s < 64; s++) n += b[s] && owner_of(b[s]) != DEAD && type_of(b[s]) == KING;
    return n;
}

struct GFlags {
    bool endgame = false;
    int trade_v = -1;  // -1 = no pending trade
    int trade_plies = 0;
    uint64_t key() const { return (uint64_t)endgame | ((uint64_t)(trade_v + 1) << 1) | ((uint64_t)trade_plies << 8); }
};

struct SearchStats {
    int depth = 0;
    uint64_t nodes = 0;
    double seconds = 0, value = 0;
};

struct Timeout {};

class GreedyBot {
public:
    GreedyParams p;
    int fixed_depth = 0;  // 0 = p.depth (or iterate to max_depth with a time budget)
    double time_ms = 0;
    int max_depth = 12;
    size_t tt_size = 400000;
    SearchStats stats;

    explicit GreedyBot(GreedyParams params = GreedyParams()) : p(params) { tt.reserve(tt_size); }

    Move best_move(Position& pos) {
        me = pos.current_player();
        root_depth = pos.depth();
        Move moves[256];
        int n = pos.legal_moves(moves);
        if (n == 0) return RESIGN;
        GFlags f0;
        f0.endgame = endgame_now(pos.board);
        nodes = 0;
        auto t0 = std::chrono::steady_clock::now();
        has_deadline = time_ms > 0;
        deadline = t0 + std::chrono::microseconds((long long)(time_ms * 1000));
        if (tt.size() > tt_size / 2) tt.clear();
        double claim;
        bool has_claim = claim_value(pos, claim);
        Move best = moves[0];
        double bv = -INF;
        int reached = 0;
        int limit = fixed_depth ? fixed_depth : p.depth;
        if (time_ms > 0 && !fixed_depth) limit = max_depth;
        std::vector<Move> order = ordered(pos, moves, n, NO_MOVE, false);
        for (int d = 1; d <= limit; d++) {
            try {
                double alpha = -INF, cur_v = -INF;
                Move cur_best = NO_MOVE;
                std::vector<std::pair<double, Move>> scores;
                for (Move m : order) {
                    double dl; GFlags nf; bool term;
                    child(pos, m, f0, dl, nf, term);
                    double v = term ? dl : dl + search(pos, d - 1, alpha - dl, INF, nf);
                    pos.pop();
                    scores.push_back({v, m});
                    if (v > cur_v) { cur_v = v; cur_best = m; alpha = std::max(alpha, v); }
                }
                best = cur_best; bv = cur_v; reached = d;
                std::stable_sort(scores.begin(), scores.end(), [](auto& a, auto& b) { return a.first > b.first; });
                for (size_t i = 0; i < scores.size(); i++) order[i] = scores[i].second;
            } catch (Timeout&) {
                while (pos.depth() > root_depth) pos.pop();
                break;
            }
        }
        if (has_claim && claim > bv) { best = RESIGN; bv = claim; }
        stats.depth = reached;
        stats.nodes = nodes;
        stats.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        stats.value = bv;
        return best;
    }

    // Root move values at a fixed depth (for analysis / teacher labels).
    size_t root_depth = 0;

    // Expected loss at a leaf from my pieces the opponents attack right now (public for tests).
    double threat(const Position& pos) const {
        const Board& b = pos.board;
        int r = pos.round();
        int order[3], no = 0;
        int s = pos.current_player();
        for (int k = 0; k < 4; k++) {
            int c = (s + k) % 4;
            if (c == me) break;
            if (pos.is_active(c)) order[no++] = c;
        }
        bool mine_to_move = false;
        if (no == 0) {
            for (int k = 1; k < 4; k++) {
                int c = (me + k) % 4;
                if (pos.is_active(c)) order[no++] = c;
            }
            mine_to_move = true;
        }
        int kings = -1;
        double losses[32];
        int nl = 0;
        for (int sq = 0; sq < 64; sq++) {
            uint8_t pc = b[sq];
            if (!pc || owner_of(pc) != me) continue;
            int t = type_of(pc);
            double v = PIECE_POINTS[t], worst = 0;
            for (int k = 0; k < no; k++) {
                int a = cheapest_attacker(b, sq, order[k]);
                if (!a) continue;
                double loss;
                if (t == KING) {
                    if (kings < 0) kings = live_kings(b);
                    loss = std::max(0, p.king_loss_rounds - r) * kings + p.loss_weight * v;
                } else {
                    bool defended = cheapest_attacker(b, sq, me) > 0;
                    double net = defended ? std::max(0.0, v - a) : v;
                    loss = p.loss_weight * net;
                }
                worst = std::max(worst, loss * p.threat_p[std::min(k + (mine_to_move ? 1 : 0), 2)]);
            }
            if (worst) losses[nl++] = worst;
        }
        if (!nl) return 0;
        int skip = -1;  // Python: losses.remove(max(losses)) drops the first maximum, then sums in order
        if (mine_to_move) {
            skip = 0;
            for (int i = 1; i < nl; i++) if (losses[i] > losses[skip]) skip = i;
        }
        double sum = 0;
        for (int i = 0; i < nl; i++) if (i != skip) sum += losses[i];
        return sum;
    }
    int me = 0;

private:
    static constexpr double INF = 1e9;
    struct TTE { int depth; int flag; double v; Move m; };
    std::unordered_map<uint64_t, TTE> tt;
    uint64_t nodes = 0;
    bool has_deadline = false;
    std::chrono::steady_clock::time_point deadline;

    void tick() {
        nodes++;
        if (has_deadline && (nodes & 1023) == 0 && std::chrono::steady_clock::now() > deadline) throw Timeout();
    }

    uint64_t key(const Position& pos, const GFlags& f) const {
        uint64_t h = pos.hash;
        h ^= splitmix(0x1234 + pos.turn % 4);
        h ^= splitmix(0x5678 + std::min(pos.round(), 30));
        for (int i = 0; i < 4; i++) h ^= splitmix(0x9abc + i * 1000003ULL + (uint64_t)(pos.scores[i] + 500));
        h ^= splitmix(0xdef0 + pos.eliminated);
        h ^= splitmix(0x4242 + f.key());
        return h;
    }

    double leaf(const Position& pos) const { return p.threat ? -threat(pos) : 0.0; }

    bool claim_value(Position& pos, double& out) {
        if (__popcnt(pos.eliminated) != 2) return false;
        pos.eliminate(pos.current_player());
        out = placement_value(pos.scores, me, p.placement);
        pos.pop();
        return true;
    }

    std::vector<Move> ordered(const Position& pos, const Move* moves, int n, Move tt_move, bool only_captures) const {
        std::vector<std::pair<int, Move>> sc;
        sc.reserve(n);
        for (int i = 0; i < n; i++) {
            Move m = moves[i];
            uint8_t victim = pos.board[to_sq(m)];
            int gain = 0;
            bool has_victim = victim != EMPTY;
            if (has_victim) {
                gain = victim == code(DEAD, KING) ? 3 : (owner_of(victim) == DEAD ? 0 : PIECE_POINTS[type_of(victim)]);
                if (owner_of(victim) == DEAD && type_of(victim) != KING) has_victim = false;
            }
            if (only_captures && (!has_victim || gain == 0)) continue;
            int att = PIECE_POINTS[type_of(pos.board[from_sq(m)])];
            int k = m == tt_move ? 1000 : (has_victim ? gain * 10 - att + 100 : 0);
            sc.push_back({k, m});
        }
        std::stable_sort(sc.begin(), sc.end(), [](auto& a, auto& b) { return a.first > b.first; });
        std::vector<Move> out(sc.size());
        for (size_t i = 0; i < sc.size(); i++) out[i] = sc[i].second;
        return out;
    }

    // Pushes m. d = value change for me, nf = new flags, term = line over (game over or I'm out).
    void child(Position& pos, Move m, const GFlags& flags, double& d, GFlags& nf, bool& term) {
        int r = pos.round();
        int kings_before = -1;
        uint8_t cap = pos.board[to_sq(m)];
        if (cap && owner_of(cap) == me && type_of(cap) == KING) kings_before = live_kings(pos.board);
        MoveInfo info = pos.push(m);
        nf = flags;
        d = 0;
        if (nf.trade_v >= 0) {
            if (nf.trade_plies < 3) nf.trade_plies++;
            else nf.trade_v = -1, nf.trade_plies = 0;
        }
        int mover = info.mover;
        if (mover == me) {
            d += info.points;
            if (cap && owner_of(cap) != DEAD) {
                if (type_of(cap) == KING && r < p.king_bonus_rounds) d += p.king_bonus;
                if (owner_of(cap) != me) { nf.trade_v = PIECE_POINTS[type_of(cap)]; nf.trade_plies = 0; }
            }
            if (p.push_bonus && r >= p.push_from && type_of(info.piece) == PAWN) d += p.push_bonus;
        } else if (cap && owner_of(cap) == me) {
            double v = PIECE_POINTS[type_of(cap)];
            if (type_of(cap) == KING) {
                if (r < p.king_loss_rounds) d -= (double)(p.king_loss_rounds - r) * kings_before;
                d -= p.king_floor;
                d -= p.loss_weight * v;
            } else {
                d -= p.loss_weight * v;
                if (type_of(cap) == PAWN && r < p.pawn_loss_rounds) d -= p.pawn_loss;
                if (flags.endgame) d -= p.endgame_loss;
                if (nf.trade_v >= 0 && v <= nf.trade_v && r < p.trade_rounds) {
                    d += p.trade_bonus;
                    nf.trade_v = -1; nf.trade_plies = 0;
                }
            }
        }
        if (!flags.endgame) {
            if (info.eliminated >= 0 && r >= 10) nf.endgame = true;
            else if (type_of(info.piece) == PAWN && pawn_to_go(mover, to_sq(m)) <= p.endgame_pawn_dist) nf.endgame = true;
        }
        term = false;
        if (!pos.is_active(me)) { term = true; return; }
        if (pos.game_over()) { d += placement_value(pos.scores, me, p.placement); term = true; }
    }

    double quiesce(Position& pos, double alpha, double beta, const GFlags& flags, int q) {
        tick();
        bool maxing = pos.current_player() == me;
        double stand = leaf(pos), claim;
        if (claim_value(pos, claim)) stand = maxing ? std::max(stand, claim) : std::min(stand, claim);
        if (q <= 0) return stand;
        if (maxing) {
            if (stand >= beta) return stand;
            alpha = std::max(alpha, stand);
        } else {
            if (stand <= alpha) return stand;
            beta = std::min(beta, stand);
        }
        double best = stand;
        Move moves[256];
        int n = pos.legal_moves(moves);
        for (Move m : ordered(pos, moves, n, NO_MOVE, true)) {
            double d; GFlags nf; bool term;
            child(pos, m, flags, d, nf, term);
            double v = term ? d : d + quiesce(pos, alpha - d, beta - d, nf, q - 1);
            pos.pop();
            if (maxing) { if (v > best) { best = v; alpha = std::max(alpha, v); } }
            else if (v < best) { best = v; beta = std::min(beta, v); }
            if (alpha >= beta) break;
        }
        return best;
    }

    double search(Position& pos, int depth, double alpha, double beta, const GFlags& flags) {
        if (depth <= 0) return quiesce(pos, alpha, beta, flags, p.qdepth);
        tick();
        uint64_t k = key(pos, flags);
        Move tt_move = NO_MOVE;
        double a0 = alpha, b0 = beta;
        auto it = tt.find(k);
        if (it != tt.end()) {
            const TTE& e = it->second;
            tt_move = e.m;
            if (e.depth >= depth) {
                if (e.flag == 0) return e.v;
                if (e.flag == 1) alpha = std::max(alpha, e.v);
                else beta = std::min(beta, e.v);
                if (alpha >= beta) return e.v;
            }
        }
        bool maxing = pos.current_player() == me;
        double best = maxing ? -INF : INF, claim;
        Move best_move = NO_MOVE;
        if (claim_value(pos, claim)) {
            best = claim;
            if (maxing) alpha = std::max(alpha, best);
            else beta = std::min(beta, best);
        }
        if (alpha < beta) {
            Move moves[256];
            int n = pos.legal_moves(moves);
            for (Move m : ordered(pos, moves, n, tt_move, false)) {
                double d; GFlags nf; bool term;
                child(pos, m, flags, d, nf, term);
                double v = term ? d : d + search(pos, depth - 1, alpha - d, beta - d, nf);
                pos.pop();
                if (maxing) { if (v > best) { best = v; best_move = m; alpha = std::max(alpha, v); } }
                else if (v < best) { best = v; best_move = m; beta = std::min(beta, v); }
                if (alpha >= beta) break;
            }
        }
        int flag = (a0 < best && best < b0) ? 0 : (best >= b0 ? 1 : 2);
        if (tt.size() < tt_size) tt[k] = TTE{depth, flag, best, best_move};  // Python: a full table is never written
        return best;
    }
};

}  // namespace cr

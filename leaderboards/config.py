"""Leaderboard settings. TOP_N is the board size; the rest tune the pool probe.

TOP_N = 100 is the target board. The top 20 come from the server's board list
(get.leaderboard). Ranks 21-100 come from a pool of players probed one by one with the
profile-card call (player-popup -> user_popup, compositeRatings[board].lbRank). The pool
only covers players we have seen, so a strong player nobody has seen can be missed. If the
probe fails, the daily job publishes the top 20 only and says so (coverage "top20").
"""

TOP_N = 100
BOARD = "rt:chaturaji:blitz"
RAW_BASE = "https://raw.githubusercontent.com/ArnoUUU/Chaturaj/main/leaderboards/data/"

LEADER_ROWS = 20          # rows the get.leaderboard reply carries
POOL_MAX_RANK = 250       # players ranked above this are dropped from the pool
SEED_MIN_RATING = 2000    # watch-list and saved-game players below this are not added
WATCH_FILTER = "all"      # subscribe-watch value; "all" lists live games of every kind
REPROBE_DAYS = 7          # non-pool candidates are probed again only after this many days
MAX_NEW_CANDIDATES = 150  # cap on non-pool candidates per run (saved games are the big source)
MAX_PROBES = 400          # hard cap on profile probes per daily run
PROBE_DELAY = 1.0         # seconds between profile probes (be polite)

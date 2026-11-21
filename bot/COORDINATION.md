# Coordination Between Colors — Quick Plan

In 4-player free-for-all Chaturaji, players often work together without saying so. One
color makes a **sacrifice or setup**, and another color **follows through**: capturing a
king, winning a rook, or breaking the leader's position. The player who set it up gains
by weakening a rival or opening a follow-up for themselves. The bot should both
**start** these plans and **recognize and join** ones other colors start.

## What it looks like
- **Sacrifice for a follow-up:** Red gives up a bishop to pull away the defender of
  Yellow's king, and Blue (moving next) captures Yellow's king. Red loses 5 points but a
  rival is eliminated, and their pieces become dead blockers.
- **Shared attack:** two colors aim at the same king. Neither can take it alone, but the
  second attacker finishes it after the first forces a weakness.
- **Turn order matters:** Red → Blue → Yellow → Green. A setup is most reliable when the
  color that follows through moves *right after* you (your "next" seat). The
  "previous" seat moves just before you, so its threats are already on the board.
- **Leader pressure:** players gang up on whoever is ahead on points. Helping trim the
  leader can be worth more than your own capture.

## How the engine learns it

### 1. Search sees it (already in the design)
Max^n MCTS assumes every color plays for itself. If Red's sacrifice gives Blue a move
that is good *for Blue* and also good for Red, the search finds that line, as long as it
looks 2–4 plies deep.
- **Do:** give sacrifices enough prior. Don't let the heuristic prior punish "moves onto
  an attacked square" so hard that sacrifices are never explored. Cap that penalty, or
  remove it for moves that create a capture threat.
- **Do:** spend enough simulations to see "my move → their capture → result" (about 3
  plies at 4 players).

### 2. The model can see it (encoding)
Add a few dense global features that make cooperative threats visible:
- Number of attackers on each relative seat's king, and from how many different colors.
- Which seat is the points leader (one-hot), and their lead.
- Pieces of mine that are attacked by more than one color.

The value head already predicts a 4-vector (one per seat), so the network can learn
"this position is good for me *because* the next player is about to take the leader's
king".

### 3. Learn it from real games (labels + extra weight)
Add an **assisted-sacrifice detector** that runs over scraped chess.com games:
1. Color **A** makes a move that gives up material: the moved piece is captured within
   the next 3 plies, or it moves onto a square attacked by another color.
2. Within the next 4 plies, a different color **B** captures a king or a piece worth
   5 or more from a third color **C**, using the line A opened. Check this by testing
   whether B's capture was illegal or blocked before A's move.
3. A's final score share, or the gap between A and the leader, ends up better than in
   similar positions without the sacrifice.

Positions that pass are tagged `coordination=True`. Uses:
- **Weight them 2–3×** in supervised training, so the policy learns these moves.
- Build a **coordination puzzle set** for evaluation: the share of setups the bot
  starts, and the share of follow-ups it completes.

### 4. Practice it in self-play
- **Copies of itself** at every seat will start and complete each other's setups when
  that is good for both.
- Add **opponent variety** so it learns more than one style:
  - *Leader-hunter* players with a shaped goal: `own points − 0.5 × leader's points`.
  - *Greedy* players that always take the biggest capture, which reliably "follow
    through" on baits.
  - *Human-like* players: copies of the supervised model (Phase 1), which follow through
    the way real players do.
- Track how often the bot's sacrifices get followed through, and how often it
  completes an opening another color creates.

## Small first steps
1. Write the detector and run it on scraped games. Count how common these patterns are.
2. Add the king-attacker and leader features (around 10 numbers) to `sparse_features`.
3. Relax the sacrifice penalty in the heuristic priors.
4. Build a puzzle set of about 50 positions from the detector's hits, and track it in the arena report.

## Open questions for you
- Which patterns do strong players actually use most often (bait for the next player,
  ganging up on the leader, other)? Real examples from your games would make the
  detector more precise.
- Should the bot sometimes sacrifice purely to stop the leader, even if its own score
  drops? That's a choice of goal, not something the code can work out on its own.

# TigerNet: Model Review and Plan for Reinforcement Learning with Self-Play

This document has two parts:

1. **A review of the current model and training setup**: what it does today, what is
   broken or limiting, and the improvements to make.
2. **A phased plan** to train both the Tiger and the Goat with reinforcement learning
   and self-play, starting with policy-gradient self-play and moving to AlphaZero-style
   MCTS, while keeping the ONNX/browser deployment working.

---

## Part 1: Review of the current model

### 1.1 What exists today

| Component | File | Summary |
|---|---|---|
| Rules engine | `src/baghchal.py` | 5x5 numpy board (`1` = tiger, `-1` = goat, `0` = empty), `validmove`/`validjump`, `checkwinner` |
| Network | `src/tiger.py` `TigerNet` | `Conv2d(3→50,k2)` → ReLU → `Conv2d(50→100,k3)` → `ConvTranspose2d(100→4,k4)` → sigmoid → 100 outputs (4 tigers × 25 destinations) |
| Move selection | `src/tiger.py` `tigernetPlayer` | Clip outputs, zero illegal moves, renormalise, sample or argmax |
| Training | `src/board.py` `TrainThread` | Plays 50 games vs. a random goat, then takes one Adam step on a capture-based loss |
| Opponent | `src/randomplayers.py` | Uniform random legal goat or tiger |
| Deployment | `docs/js/ai.js` + `docs/bg_trained_400_4.onnx` | Browser inference with onnxruntime-web |

### 1.2 Correctness issues (fix these before any RL work)

1. **Output indices depend on tiger identity, which the input doesn't contain.**
   Output `k` means "move tiger `1 + k//25` to square `k%25`". The tiger IDs come from
   the `game.tigers` dict, but the network only sees a board with anonymous tiger planes.
   It can't tell which tiger is "tiger 1", so it has to learn a mapping that the input
   doesn't determine. **Fix:** use a from-square/to-square action encoding (§2.2).

2. **The training loss is not a real RL objective.**
   ```python
   if geaten != 1: prob = 1 - prob
   loss = -prob.log() - total_goats_eaten / games
   ```
   - The second term is a constant, so it adds no gradient. The game outcome and the
     total number of captures never affect learning.
   - So the loss is per-move supervised learning: push up sigmoid outputs for moves
     that captured right away, push down all others. The tiger learns to be greedy
     about the next capture, which is the strategy goats exploit.
   - Each output is an independent sigmoid, not a softmax over legal moves. Pushing
     one move down doesn't push the alternatives up, and the probabilities aren't a
     distribution until they are renormalised outside the graph.
   - `prob + 0.01` when `prob < 0.01` is a hand-made floor. Use `log_softmax` instead.

3. **Games never reach the movement phase.** `MAX_STEPS = 25` counts plies for both
   players, so a training game lasts about 12 goat placements and 12 tiger moves. The
   goat places 20 goats before it can move, so the network never sees the movement
   phase, where most of the strategy is. Almost every game ends as "no winner"
   (`winners.count(0)` dominates the printout).

4. **`goat_eaten` is never set.** `Baghchal.movetiger` updates `goats_eaten` but never
   sets the `goat_eaten` flag. As a result, `current_game_goats_eaten_or_not` in the GUI
   path is always 0, and *Train with played games* treats every human game as having no
   captures.

5. **Bugs in *Train with played games*:**
   - Its loss divides by `len(self.saved_move_probs)`, unlike the other path.
   - `self.save_move_probs = []` is a typo, so `saved_move_probs` is never cleared and
     the same games are replayed forever.
   - It calls `loss.backward(retain_graph=True)` on graphs that were built during play
     under the GUI's own forward passes.

6. **Thread safety.** `TrainThread` runs on a `QThreadPool` thread and mutates the
   global `net` and `optimizer` while the GUI thread runs `net(...)` for moves.
   Inference also runs without `torch.no_grad()`/`net.eval()`, so it builds autograd
   graphs for no reason.

7. **Inconsistent goat encoding.** Python uses `b == -1` for the goat plane, while
   `savemodel()` exports with a dummy input built from `b == 2`. The JS side stores goats
   as `2`, and the network sees the same planes in both. This works, but only by
   accident. Use one shared `encode_state()` everywhere.

8. **Fragile ONNX contract.** `ai.js` reads `output[13]`, an auto-generated tensor name,
   and the input is `"input.1"`. Re-exporting a different architecture will silently
   break the web page. Export with `input_names=["board"]` and `output_names=["policy", "value"]`.

9. **The web page uses different rules from Python.** `checkWinner()` in
   `docs/js/game.js` only gives the tiger the win at 6 captures (`goatsEaten > 5`),
   while Python uses 5. `ai.js` also creates a new `InferenceSession` on every tiger
   move. The new app in `docs/play/` fixes both (see Phase 5).

10. **Rules gaps:**
   - Nothing detects draws or repetitions. In the movement phase both sides can
     shuffle forever.
   - When the goat has no legal move, the game is decided ad hoc in the training loop
     (`winner = 1`), not in the engine.
   - The `checkwinner` docstring says the tiger wins at 6 captures, but the code
     uses 5 (`goats_eaten > 4`). Five is the common rule, so fix the docstring.

### 1.3 Modelling limitations

| Limitation | Effect | Improvement |
|---|---|---|
| The input is only 3 planes (empty, tiger, goat) | The net can't see the phase (goats left to place), the number of captures, or whose turn it is | Add constant planes for `goats_outside/20`, `goats_eaten/5`, and side to move; add a "legal destinations" plane |
| Unpadded convs shrink 5x5 → 4x4 → 2x2 | Spatial information is lost early and rebuilt by the transposed conv | Use `padding=1` 3x3 convs that keep the board 5x5, in a small residual tower |
| Policy only | No position evaluation, so no baseline for variance reduction and nothing for search to use | Add a **value head** (tanh scalar, expected outcome in [-1, 1]) |
| Sigmoid per action | Not a probability distribution | Masked **softmax** over legal actions |
| Tiger-only network | Self-play is impossible without a learned goat | One shared policy-value net for **both sides**, conditioned on side to move (or two nets that share code) |
| Only the random goat as opponent and benchmark | Beating a random goat says little about strength | Add heuristic baselines, an opponent pool, and Elo tracking |
| No use of board symmetry | Wastes data | The board's line pattern (orthogonal lines everywhere, diagonals on even-parity points) is preserved by all 8 rotations and reflections of the square, so augment ×8 |

### 1.4 Recommended network (replacement for `TigerNet`)

```
input  : (B, C_in, 5, 5)   C_in ≈ 7  [empty, tiger, goat, side_to_move,
                                       goats_outside/20, goats_eaten/5, legal_dest]
trunk  : Conv3x3(C_in→64, pad=1) + BN + ReLU
         4–6 × ResidualBlock(64)        (Conv3x3-BN-ReLU-Conv3x3-BN + skip, ReLU)
policy : Conv1x1(64→32) + BN + ReLU → flatten(800) → Linear(800→A)   (A = 650, see §2.2)
value  : Conv1x1(64→8)  + BN + ReLU → flatten(200) → Linear(200→64) → ReLU → Linear(64→1) → tanh
```

- About 300–500k parameters, which is still fast on CPU and small enough for
  onnxruntime-web.
- Apply the legal-move mask outside the net as logits plus `-inf` on illegal actions,
  so the ONNX export stays simple. JS applies the same mask.

---

## Part 2: Plan for RL and self-play

### Guiding decisions

- **Learn both sides.** Self-play needs a learned goat, and a strong tiger needs strong
  goats to train against. The site can then also let people play as the tiger.
- **Reward = game outcome only:** +1 win, −1 loss, 0 draw, from the mover's
  perspective. Captures may be added as a small shaped reward early in training, but
  it must be annealed to zero so it doesn't bring back the greedy-tiger bias.
- **Two learning stages:**
  - **Stage A**, actor-critic (PPO) self-play: cheap, reuses the existing
    "policy outputs a move" design, and its output is a plain policy that runs in the
    browser without search.
  - **Stage B**, AlphaZero-style MCTS plus a policy-value net: much stronger. Its
    result can be deployed as a raw policy (distilled) or with light MCTS in JS.
- **Training runs headless.** Move training out of the Qt GUI into a command-line
  program that writes checkpoints. The GUI and the web page only load models.
- **Ground truth.** Baghchal is reported as weakly solved as a draw with perfect play
  (Lim & Nievergelt, 2004, *Computing Tigers and Goats*). Expect strong play to draw
  often, and measure progress by win rates against fixed opponents, not by "always wins".

### Proposed layout

```
src/
  rules.py         # fast engine: precomputed adjacency/jump tables, legal moves,
                   # apply/undo, terminal + draw detection, state hashing
  encoding.py      # encode_state(), action <-> move mapping, 8 board symmetries
  model.py         # PolicyValueNet (§1.4)
  agents.py        # RandomAgent, GreedyTiger, HeuristicGoat, NetAgent, MCTSAgent
  ppo.py           # Stage A trainer
  mcts.py          # Stage B PUCT search (batched leaf evaluation)
  selfplay.py      # self-play workers → replay buffer
  replay.py        # ring buffer of (state, policy target, value target)
  evaluate.py      # arena matches, Elo / win-rate tables, gating
  export_onnx.py   # stable named inputs/outputs, writes docs/*.onnx
train.py           # CLI: python train.py --stage ppo|az --config configs/*.yaml
tests/             # pytest: rules, encoding round-trips, symmetry, MCTS sanity
```

`src/baghchal.py`, `src/tiger.py` and the GUI stay working during the migration.
`Board` will later switch to `NetAgent` and load a checkpoint.

---

### Phase 0: Engine fixes and tests (prerequisite)

1. Write `rules.py`, using the 25-point board graph. `docs/play/js/engine.js` is a
   tested reference implementation of the same rules and action encoding, so port
   it rather than starting from scratch:
   - Precompute `NEIGHBORS[sq]` and `JUMPS[sq] = [(over, to), ...]` once.
   - Legal move generation then costs O(pieces × 8) instead of scanning every pair of
     squares. MCTS needs this speed.
2. Make game state explicit and immutable or copyable:
   board, side to move, goats to place, goats captured, ply count, and a position
   history for repetition checks.
3. Terminal rules:
   - The tiger wins on its 5th capture.
   - The goat wins when the tiger has no legal move.
   - A goat with no legal move on its turn loses. Write the chosen rule down and
     enforce it in the engine.
   - **Draw** on threefold repetition, or when the ply limit (e.g., 200) is reached.
4. Fix the issues in §1.2: the `goat_eaten` flag, the `save_move_probs` typo, the
   `MAX_STEPS` semantics, and the docstrings.
5. Tests:
   - Known move counts from the start position and a few hand-made positions
     (perft-style).
   - Every jump is legal only over a goat onto an empty square.
   - Terminal detection.
   - Python engine and `docs/js/ai.js` agree on legal moves for a set of random
     positions (export fixtures as JSON).

**Done when:** pytest is green and 10k random-vs-random games finish without errors,
with win/draw rates reported.

### Phase 1: Encoding, model, baselines

1. **Action space shared by both sides (A = 650):**
   - `0..24`: goat *placement* on square `s`.
   - `25 + from*25 + to`: *move or jump* from `from` to `to`. The rules decide whether
     it is a step or a jump.
   - Simple to decode in JS, and it doesn't depend on tiger identity.
   - A more compact option is from × 16 directions (8 steps + 8 jumps), about 425
     actions. It's an option, but not needed at this board size.
2. **State encoding** as in §1.4, always from the side-to-move's perspective. Write
   `encode_state()` once in Python and port it to JS exactly. Test with shared
   fixtures.
3. **Symmetries:** apply the 8 D4 transforms to planes and actions (permute
   squares). Test that legal-move sets map correctly under each transform.
4. **Baseline agents** used for evaluation:
   - Random.
   - Greedy tiger: capture if possible, else maximise its own mobility.
   - Heuristic goat: avoid squares that can be jumped, fill edges and corners first,
     minimise tiger mobility.
   - The existing `bg_trained_400_4.onnx`, wrapped as an agent, as a historical
     reference.
5. `PolicyValueNet` with a masked-softmax helper.

**Done when:** the old ONNX model and all baselines play each other in `evaluate.py`,
which prints a win/draw/loss table.

### Phase 2 (Stage A): PPO self-play

1. **Data collection:**
   - Run N parallel games (vectorised in one process with batched forward passes;
     use multiprocessing later if needed).
   - Both sides are played by the current net, with a configurable fraction of games
     against the opponent pool.
2. **Opponent pool (league):**
   - Snapshot the policy every K updates.
   - Sample opponents as about 50% latest self, 30% past snapshots, 20% baselines.
   - This prevents strategy cycling, where tiger and goat chase each other's
     latest weakness.
3. **Learning:**
   - Generalised advantage estimation with the value head as critic
     (γ = 1, λ ≈ 0.95).
   - Clipped PPO objective, value loss, and an entropy bonus that decays over
     training.
   - Use the masked softmax for both log-probabilities and entropy.
   - Augment each sample with a random symmetry.
4. **Curriculum:**
   - (a) Train both sides against the random and heuristic baselines first.
   - (b) Switch to self-play and the league.
   - (c) Optionally add a small capture reward (`±0.1` per capture), annealed to 0
     within the first ~20% of training.
5. **Logging:**
   - Use TensorBoard or CSV.
   - Track win rate per side against each baseline, game length, draw rate, policy
     entropy, value loss, and the explained variance of the value head.

**Done when:** as tiger, the net beats the random goat in > 95% of games; as goat, it
beats or draws the greedy tiger in > 80% of games. It must also clearly beat the old
ONNX model from both sides.

### Phase 3 (Stage B): AlphaZero-style MCTS self-play

1. **PUCT MCTS** (`mcts.py`):
   - Priors come from the policy head and leaf values from the value head.
   - Add Dirichlet noise at the root (α ≈ 0.3 for ~10–30 legal moves, ε = 0.25).
   - Use virtual loss so leaf evaluations can be batched.
   - Start with 100–200 simulations per move.
   - Store values from the side-to-move's perspective and flip the sign when the
     side changes (the game alternates strictly, so this is simple).
2. **Self-play loop:**
   - Play games with temperature τ = 1 for the first ~10–15 plies, then τ → 0.
   - Record `(state, π_visits, z)` for every position, with `z` the final outcome
     from that position's mover.
   - Draws give `z = 0`. For data efficiency, cap the repetition draw sooner during
     self-play.
3. **Training:**
   - Loss: `L = (z − v)² − π·log p + c‖θ‖²`.
   - Sample minibatches from a replay buffer holding the last ~50–200k positions,
     with symmetry augmentation.
4. **Gating:**
   - A new checkpoint plays the current best for ~200 games (MCTS on both sides,
     alternating colours).
   - It's promoted if it scores ≥ 55%. Alternatively skip gating (AlphaZero style)
     and track Elo only.
5. **Warm start:** initialise from the Stage A network to save compute.
6. **Throughput:**
   - The precomputed-table engine plus batched inference across 64–256 games
     running at once should be enough on CPU.
   - If Python MCTS becomes the bottleneck, move the tree to numba, or move the
     engine and tree to a small C++/Rust extension. The interface stays the same.

**Done when:** MCTS(200) with the Stage B net beats the best Stage A policy from both
sides, and Elo rises steadily across generations. A good secondary check is that the
draw rate in self-play goes up, as expected for a game solved as a draw.

### Phase 4: Evaluation and analysis

- `evaluate.py` runs a round-robin over: random, greedy, heuristic, the old ONNX
  model, Stage A, and Stage B (raw policy and with MCTS at 50/200/800 simulations).
  It outputs an Elo table (e.g., BayesElo-style fit) and a per-side W/D/L matrix.
- Run fixed test positions with known tactical answers, such as "tiger must capture
  now" or "goat must block a double threat", as regression tests on the policy.
- Optional: compare against a small endgame or late-placement database built by
  retrograde analysis, for exact value checks.

### Phase 5: Deployment (GUI and web)

A new web app, separate from the original page, lives in **`docs/play/`** and is
served at `/TigerNet/play/` on GitHub Pages. It already covers most of the web side
of this phase:

- **Rules engine:** precomputed board graph, the shared 650-action encoding,
  repetition and quiet-move draws.
- **Players:**
  - Human, Random, Greedy, Minimax (alpha-beta) and MCTS, all running in a Web Worker.
  - The legacy TigerNet ONNX model.
  - A **policy-value model loader** that implements the contract below, for either side.
- **Analysis tools:** live MCTS analysis with move arrows and an evaluation bar,
  hints, warnings for threatened goats and unsafe points, undo/redo, a clickable
  move list, and shareable game links.
- **Arena tab:** runs batches of engine-vs-engine games and shows W/D/L results.
  These are useful Phase 1 and Phase 4 baselines.
- **Tests:** `node --test tests/web/*.test.mjs`.

Remaining work for this phase:

1. **ONNX export (`export_onnx.py`):**
   - Named `board → policy, value`, with a fixed opset.
   - Input `board`, float32 `[1, 7, 5, 5]`, with these planes:
     - 0 empty, 1 tiger, 2 goat;
     - 3 side to move (ones when the tiger moves);
     - 4 goats to place / 20, 5 goats captured / 5;
     - 6 legal destination squares.
   - Output `policy`, float32 `[1, 650]`: logits.
   - Output `value`, float32 `[1, 1]`: in [-1, 1], for the side to move.
   - The reference encoder is `encodeState()` in `docs/play/js/neural.js`. The Python
     `encode_state()` must match it, so test both against shared JSON fixtures.
   - Include a JSON sidecar with the action-space version, input-plane spec and
     model version.
2. **Web:**
   - Ship a trained model under `docs/play/models/` and preload it, so it doesn't
     need to be loaded from a file.
   - Add PUCT search that uses the network's priors and values (AlphaZero inference)
     as a "Hard" neural level.
   - Retire or redirect the original `docs/index.html` once the new app is the default.
3. **GUI:**
   - Replace the *Train* buttons with *Load checkpoint* plus side selection
     (human plays goat or tiger).
   - Optionally show the value-head evaluation and the top-k policy moves as an
     overlay.
   - Keep training out of the GUI process. If interactive training is still wanted,
     run it in a separate process that writes checkpoints.
4. Update `README.md` and `requirements.txt`:
   - Replace `onnx-pytorch` with `onnx`, plus `onnxruntime` for tests.
   - Add `pytest` and `tensorboard`.

---

### Risks and mitigations

| Risk | Mitigation |
|---|---|
| Endless shuffling games / degenerate draws | Repetition plus ply-limit draws in the engine; shorter limits during self-play |
| Strategy cycling between tiger and goat | Opponent league in Stage A; replay buffer and gating in Stage B |
| Asymmetric difficulty (one side learns much faster) | Track per-side metrics; if the value head collapses toward one side, re-balance the opponent mix or sample more positions for the weaker side |
| JS and Python rule/encoding drift | Shared JSON fixtures tested in both languages; versioned action-space spec in the ONNX sidecar |
| Python MCTS too slow | Precomputed tables and batched inference first; numba or a native extension only if profiling shows it's needed |

### Suggested order of pull requests

1. Engine rewrite plus tests, and §1.2 bug fixes (no behaviour change for the GUI
   beyond the fixes).
2. Encoding, action space, symmetries, baselines, and the arena.
3. `PolicyValueNet` and headless PPO self-play training (`train.py --stage ppo`).
4. ONNX export with a stable contract, plus the web/GUI update to the new model.
5. MCTS plus AlphaZero training loop (`train.py --stage az`) and gating.
6. Ship the trained model in `docs/play/` and add network-guided PUCT search there.
   (Search engines, play-as-tiger, analysis overlays and the arena are already done.)

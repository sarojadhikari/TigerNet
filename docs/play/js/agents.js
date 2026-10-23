/**
 * Search-based and heuristic players. All agents are synchronous functions
 * `(state, opts) => { move, info }` so they can run inside a Web Worker or
 * under Node for tests and arena matches.
 */
import {
  TIGER, GOAT, CAPTURES_TO_WIN, QUIET_LIMIT, JUMP_OVER,
  initialState, legalMoves, applyMove, status, repetitionCount, tigerCanCapture,
  threatenedGoats, unsafeSquares, tigerMobility, decode,
} from './engine.js';

const WIN = 100000;

const pick = (arr) => arr[Math.floor(Math.random() * arr.length)];

function shuffle(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

const isCapture = (a) => a >= 25 && JUMP_OVER[a - 25] >= 0;

/**
 * Static evaluation from the tiger's point of view (positive favours tiger).
 * Captures dominate; then immediate threats, tiger mobility and trapped tigers.
 */
export function evaluate(s) {
  let mobility = 0, trapped = 0;
  for (const m of tigerMobility(s)) {
    mobility += m;
    if (m === 0) trapped++;
  }
  const threats = threatenedGoats(s).size;
  let threatScore;
  if (s.turn === TIGER) {
    // The tiger will take one goat right away.
    threatScore = threats > 0 ? 70 + 15 * (threats - 1) : 0;
  } else {
    // The goat can usually parry a single threat, but not two.
    threatScore = threats >= 2 ? 50 + 15 * (threats - 2) : 8 * threats;
  }
  // Mobility is centred on 10 moves so that a quiet, balanced position scores near 0.
  return 100 * s.captured + threatScore + 4 * (mobility - 10) - 30 * trapped;
}

/** Uniformly random legal move. */
export function randomAgent(s) {
  return { move: pick(legalMoves(s)), info: {} };
}

/** One-ply greedy: best static evaluation after the move, random tie-break. */
export function greedyAgent(s) {
  const moves = shuffle(legalMoves(s));
  const sign = s.turn === TIGER ? 1 : -1;
  let best = moves[0], bestScore = -Infinity;
  for (const a of moves) {
    const t = applyMove(s, a);
    const score = t.captured >= CAPTURES_TO_WIN ? WIN : sign * evaluate(t);
    if (score > bestScore) { bestScore = score; best = a; }
  }
  return { move: best, info: { score: bestScore } };
}

// ---------------------------------------------------------------- minimax

class Timeout extends Error {}

/** Terminal score from the side to move, or null if the game goes on. */
function terminalScore(s, moves, dist) {
  if (s.captured >= CAPTURES_TO_WIN) return s.turn === TIGER ? WIN - dist : -(WIN - dist);
  if (moves.length === 0) return -(WIN - dist);
  // Inside the search a single repetition already counts as a draw.
  if (dist > 0 && !s.irreversible && repetitionCount(s) >= 2) return 0;
  if (s.quiet >= QUIET_LIMIT) return 0;
  return null;
}

function orderMoves(s, moves, ttMove) {
  let unsafe = null;
  if (s.turn === GOAT) unsafe = unsafeSquares(s);
  const scored = moves.map((a) => {
    let k = Math.random();
    if (a === ttMove) k += 1000;
    if (s.turn === TIGER && isCapture(a)) k += 100;
    if (unsafe && unsafe.has(decode(a).to)) k -= 100;
    return [k, a];
  });
  scored.sort((x, y) => y[0] - x[0]);
  return scored.map((x) => x[1]);
}

function negamax(s, depth, alpha, beta, dist, ctx) {
  if ((++ctx.nodes & 1023) === 0 && performance.now() > ctx.deadline) throw new Timeout();
  const moves = legalMoves(s);
  const term = terminalScore(s, moves, dist);
  if (term !== null) return term;
  if (depth === 0) return (s.turn === TIGER ? 1 : -1) * evaluate(s);

  const ttKey = s.key + 2541865828329 * s.toPlace; // 3^26 keeps keys exact
  const entry = ctx.tt.get(ttKey);
  if (entry && entry.depth >= depth && dist > 0) {
    if (entry.flag === 0) return entry.score;
    if (entry.flag < 0 && entry.score <= alpha) return entry.score;
    if (entry.flag > 0 && entry.score >= beta) return entry.score;
  }

  const alpha0 = alpha;
  let best = -Infinity, bestMove = moves[0];
  for (const a of orderMoves(s, moves, entry?.move)) {
    const score = -negamax(applyMove(s, a), depth - 1, -beta, -alpha, dist + 1, ctx);
    if (score > best) { best = score; bestMove = a; }
    if (best > alpha) alpha = best;
    if (alpha >= beta) break;
  }
  const flag = best <= alpha0 ? -1 : best >= beta ? 1 : 0;
  ctx.tt.set(ttKey, { depth, score: best, flag, move: bestMove });
  return best;
}

/**
 * Iterative-deepening alpha-beta with a transposition table.
 * opts: { timeMs = 800, maxDepth = 20 }
 */
export function minimaxAgent(s, opts = {}) {
  const timeMs = opts.timeMs ?? 800, maxDepth = opts.maxDepth ?? 20;
  const ctx = { nodes: 0, deadline: performance.now() + timeMs, tt: new Map() };
  const rootMoves = legalMoves(s);
  let bestMove = pick(rootMoves), bestScore = 0, reached = 0;

  for (let depth = 1; depth <= maxDepth; depth++) {
    try {
      let alpha = -Infinity, iterBest = null, iterScore = -Infinity;
      for (const a of orderMoves(s, rootMoves, bestMove)) {
        const score = -negamax(applyMove(s, a), depth - 1, -Infinity, -alpha, 1, ctx);
        if (score > iterScore) { iterScore = score; iterBest = a; }
        if (score > alpha) alpha = score;
      }
      bestMove = iterBest;
      bestScore = iterScore;
      reached = depth;
      if (Math.abs(bestScore) > WIN - 1000) break; // forced result found
    } catch (e) {
      if (!(e instanceof Timeout)) throw e;
      break;
    }
  }
  return { move: bestMove, info: { depth: reached, score: bestScore, nodes: ctx.nodes, mate: Math.abs(bestScore) > WIN - 1000 } };
}

// ------------------------------------------------------------------- MCTS

const ROLLOUT_PLIES = 6;

/** Light rollout policy: tigers grab captures, goats avoid hanging pieces. */
function rolloutMove(s, moves) {
  if (s.turn === TIGER) {
    if (Math.random() < 0.9) {
      const caps = moves.filter(isCapture);
      if (caps.length) return pick(caps);
    }
    return pick(moves);
  }
  // Goats pick at random among moves that leave nothing en prise, when any exist.
  const safe = moves.filter((a) => !tigerCanCapture(applyMove(s, a)));
  return pick(safe.length ? safe : moves);
}

/** Play out to the end (or a ply cap) and return the value for the tiger in [-1, 1]. */
function rollout(s) {
  for (let i = 0; i < ROLLOUT_PLIES; i++) {
    const moves = legalMoves(s);
    const st = status(s, moves);
    if (st.over) return st.winner === TIGER ? 1 : st.winner === GOAT ? -1 : 0;
    s = applyMove(s, rolloutMove(s, moves));
  }
  return Math.tanh(evaluate(s) / 150);
}

class Node {
  constructor(state, move, parent) {
    this.state = state;
    this.move = move;
    this.parent = parent;
    this.children = [];
    this.moves = legalMoves(state);
    this.status = status(state, this.moves);
    this.untried = this.status.over ? [] : shuffle(this.moves.slice());
    this.n = 0;
    this.w = 0; // summed value for the player who made `move`
  }
}

/**
 * UCT Monte Carlo tree search with heuristic rollouts.
 * opts: { timeMs = 800, iterations = Infinity, c = 1.2 }
 * info.top lists root moves by visits; info.value is the root value for the side to move.
 */
export function mctsAgent(s, opts = {}) {
  const timeMs = opts.timeMs ?? 800, maxIter = opts.iterations ?? Infinity, c = opts.c ?? 1.2;
  const root = new Node(s, -1, null);
  const deadline = performance.now() + timeMs;
  let iter = 0;

  while (iter < maxIter && ((iter & 31) !== 0 || performance.now() < deadline)) {
    iter++;
    let node = root;
    // Selection
    while (node.untried.length === 0 && node.children.length > 0) {
      const logN = Math.log(node.n);
      let best = null, bestU = -Infinity;
      for (const ch of node.children) {
        const u = ch.w / ch.n + c * Math.sqrt(logN / ch.n);
        if (u > bestU) { bestU = u; best = ch; }
      }
      node = best;
    }
    // Expansion
    if (node.untried.length > 0) {
      const a = node.untried.pop();
      const child = new Node(applyMove(node.state, a), a, node);
      node.children.push(child);
      node = child;
    }
    // Simulation (value from the tiger's perspective)
    let v;
    if (node.status.over) v = node.status.winner === TIGER ? 1 : node.status.winner === GOAT ? -1 : 0;
    else v = rollout(node.state);
    // Backpropagation
    for (let p = node; p; p = p.parent) {
      p.n++;
      if (p.parent) p.w += p.parent.state.turn === TIGER ? v : -v;
    }
  }

  const ranked = root.children.slice().sort((x, y) => y.n - x.n);
  const best = ranked[0];
  const top = ranked.slice(0, 5).map((ch) => ({ move: ch.move, visits: ch.n, value: ch.w / ch.n }));
  return {
    move: best ? best.move : pick(root.moves),
    info: { iterations: iter, value: best ? best.w / best.n : 0, top },
  };
}

// --------------------------------------------------------------- registry

export const AGENTS = {
  random: { label: 'Random', run: randomAgent },
  greedy: { label: 'Greedy (1-ply)', run: greedyAgent },
  minimax: { label: 'Minimax (alpha-beta)', run: minimaxAgent },
  mcts: { label: 'MCTS (UCT)', run: mctsAgent },
};

export function runAgent(spec, s) {
  const agent = AGENTS[spec.kind];
  if (!agent) throw new Error(`unknown agent ${spec.kind}`);
  return agent.run(s, spec);
}

/** Play one game between two agent specs; returns { winner, reason, plies }. */
export function playGame(goatSpec, tigerSpec) {
  let s = initialState();
  for (;;) {
    const st = status(s);
    if (st.over) return { winner: st.winner, reason: st.reason, plies: s.ply };
    s = applyMove(s, runAgent(s.turn === GOAT ? goatSpec : tigerSpec, s).move);
  }
}

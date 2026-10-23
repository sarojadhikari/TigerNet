/**
 * Baghchal rules engine.
 *
 * Pure functions on immutable states, shared by the UI, the search worker and
 * the Node tests. The board is 25 points indexed sq = row * 5 + col (row 0 at
 * the top). Lines run orthogonally everywhere and diagonally through points
 * whose row + col is even, so the whole board graph is precomputed once.
 *
 * Actions use the shared 650-entry encoding from PLAN_RL_SELFPLAY.md:
 *   0..24                 goat placement on square `to`
 *   25 + from * 25 + to   step or jump from `from` to `to`
 */

export const EMPTY = 0;
export const TIGER = 1;
export const GOAT = 2;

export const SIZE = 5;
export const N = 25;
export const NUM_ACTIONS = 650;
export const GOATS_TOTAL = 20;
export const CAPTURES_TO_WIN = 5;
/** Plies without a placement or capture before the game is drawn. */
export const QUIET_LIMIT = 100;

export const row = (sq) => Math.floor(sq / SIZE);
export const col = (sq) => sq % SIZE;

const DIRS = [[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [-1, 1], [1, -1], [1, 1]];

/** NEIGHBORS[sq]: points one step away along a line. */
export const NEIGHBORS = [];
/** JUMPS[sq]: [over, to] pairs for a straight two-step jump. */
export const JUMPS = [];
/** JUMP_OVER[from * 25 + to]: the jumped square, or -1 if not a jump. */
export const JUMP_OVER = new Int8Array(N * N).fill(-1);

for (let sq = 0; sq < N; sq++) {
  const r = row(sq), c = col(sq);
  const diagonalPoint = (r + c) % 2 === 0;
  NEIGHBORS.push([]);
  JUMPS.push([]);
  for (const [dr, dc] of DIRS) {
    if (dr !== 0 && dc !== 0 && !diagonalPoint) continue;
    const r1 = r + dr, c1 = c + dc;
    if (r1 < 0 || r1 >= SIZE || c1 < 0 || c1 >= SIZE) continue;
    NEIGHBORS[sq].push(r1 * SIZE + c1);
    const r2 = r + 2 * dr, c2 = c + 2 * dc;
    if (r2 < 0 || r2 >= SIZE || c2 < 0 || c2 >= SIZE) continue;
    const over = r1 * SIZE + c1, to = r2 * SIZE + c2;
    JUMPS[sq].push([over, to]);
    JUMP_OVER[sq * N + to] = over;
  }
}

// Exact numeric position key: base-3 board digits plus the side to move.
const POW3 = [];
for (let i = 0, p = 1; i <= N + 1; i++, p *= 3) POW3.push(p);
const TURN_KEY = POW3[N];

/** Tiger start squares, in the identity order the legacy TigerNet model uses. */
export const TIGER_START = [0, 20, 4, 24];

export function initialState() {
  const board = new Int8Array(N);
  let key = 0;
  for (const sq of TIGER_START) {
    board[sq] = TIGER;
    key += TIGER * POW3[sq];
  }
  return {
    board,
    turn: GOAT,
    toPlace: GOATS_TOTAL,
    captured: 0,
    ply: 0,
    quiet: 0,
    tigers: TIGER_START.slice(),
    key,
    irreversible: true,
    prev: null,
    lastMove: -1,
  };
}

export const placeAction = (to) => to;
export const moveAction = (from, to) => 25 + from * N + to;
export const isPlacement = (a) => a < 25;

/** Decode an action into { from, to, over } (from = -1 for a placement). */
export function decode(a) {
  if (a < 25) return { from: -1, to: a, over: -1 };
  const from = Math.floor((a - 25) / N), to = (a - 25) % N;
  return { from, to, over: JUMP_OVER[from * N + to] };
}

export function legalMoves(s) {
  const b = s.board, moves = [];
  if (s.turn === GOAT) {
    if (s.toPlace > 0) {
      for (let sq = 0; sq < N; sq++) if (b[sq] === EMPTY) moves.push(sq);
    } else {
      for (let sq = 0; sq < N; sq++) {
        if (b[sq] !== GOAT) continue;
        for (const to of NEIGHBORS[sq]) if (b[to] === EMPTY) moves.push(moveAction(sq, to));
      }
    }
  } else {
    for (const sq of s.tigers) {
      for (const to of NEIGHBORS[sq]) if (b[to] === EMPTY) moves.push(moveAction(sq, to));
      for (const [over, to] of JUMPS[sq]) {
        if (b[over] === GOAT && b[to] === EMPTY) moves.push(moveAction(sq, to));
      }
    }
  }
  return moves;
}

/** Return the state after action `a`; `a` must be legal. */
export function applyMove(s, a) {
  const board = s.board.slice();
  let { toPlace, captured, tigers, key } = s;
  let irreversible = false;
  if (a < 25) {
    board[a] = GOAT;
    key += GOAT * POW3[a];
    toPlace -= 1;
    irreversible = true;
  } else {
    const from = Math.floor((a - 25) / N), to = (a - 25) % N;
    const piece = board[from];
    board[from] = EMPTY;
    board[to] = piece;
    key += piece * (POW3[to] - POW3[from]);
    if (piece === TIGER) {
      tigers = tigers.slice();
      tigers[tigers.indexOf(from)] = to;
      const over = JUMP_OVER[from * N + to];
      if (over >= 0) {
        board[over] = EMPTY;
        key -= GOAT * POW3[over];
        captured += 1;
        irreversible = true;
      }
    }
  }
  const turn = s.turn === GOAT ? TIGER : GOAT;
  key += turn === TIGER ? TURN_KEY : -TURN_KEY;
  return {
    board,
    turn,
    toPlace,
    captured,
    ply: s.ply + 1,
    quiet: irreversible ? 0 : s.quiet + 1,
    tigers,
    key,
    irreversible,
    prev: s,
    lastMove: a,
  };
}

/** Number of times the current position occurred since the last irreversible move. */
export function repetitionCount(s) {
  let count = 1;
  for (let p = s; !p.irreversible && p.prev; ) {
    p = p.prev;
    if (p.key === s.key) count++;
  }
  return count;
}

/**
 * Game status: { over, winner (TIGER | GOAT | null for draw), reason }.
 * Pass `moves` when the caller already generated them.
 */
export function status(s, moves = null) {
  if (s.captured >= CAPTURES_TO_WIN) return { over: true, winner: TIGER, reason: `${CAPTURES_TO_WIN} goats captured` };
  const legal = moves ?? legalMoves(s);
  if (legal.length === 0) {
    return s.turn === TIGER
      ? { over: true, winner: GOAT, reason: 'all tigers trapped' }
      : { over: true, winner: TIGER, reason: 'goats cannot move' };
  }
  if (repetitionCount(s) >= 3) return { over: true, winner: null, reason: 'threefold repetition' };
  if (s.quiet >= QUIET_LIMIT) return { over: true, winner: null, reason: `${QUIET_LIMIT} plies without a capture` };
  return { over: false, winner: null, reason: '' };
}

/** True if the tiger, were it to move, could capture something. */
export function tigerCanCapture(s) {
  const b = s.board;
  for (const sq of s.tigers) {
    for (const [over, to] of JUMPS[sq]) if (b[over] === GOAT && b[to] === EMPTY) return true;
  }
  return false;
}

/** Distinct goats that a tiger could jump right now, regardless of turn. */
export function threatenedGoats(s) {
  const b = s.board, out = new Set();
  for (const sq of s.tigers) {
    for (const [over, to] of JUMPS[sq]) if (b[over] === GOAT && b[to] === EMPTY) out.add(over);
  }
  return out;
}

/**
 * Empty squares where a goat placed or moved now could be jumped at once:
 * a tiger sits on one side along a line and the other side is empty.
 * (`vacated` is treated as empty, for goat moves.)
 */
export function unsafeSquares(s, vacated = -1) {
  const b = s.board, out = new Set();
  for (const sq of s.tigers) {
    for (const [over, to] of JUMPS[sq]) {
      if ((b[over] === EMPTY || over === vacated) && (b[to] === EMPTY || to === vacated)) out.add(over);
    }
  }
  return out;
}

/** Per-tiger legal move counts (as if the tiger were to move). */
export function tigerMobility(s) {
  const b = s.board;
  return s.tigers.map((sq) => {
    let n = 0;
    for (const to of NEIGHBORS[sq]) if (b[to] === EMPTY) n++;
    for (const [over, to] of JUMPS[sq]) if (b[over] === GOAT && b[to] === EMPTY) n++;
    return n;
  });
}

export function goatsOnBoard(s) {
  let n = 0;
  for (let sq = 0; sq < N; sq++) if (s.board[sq] === GOAT) n++;
  return n;
}

/** Square name: files a–e left to right, ranks 5–1 top to bottom. */
export const squareName = (sq) => 'abcde'[col(sq)] + (SIZE - row(sq));

/** Human-readable move: "G c3", "G b2-b3", "T a5-b4", "T a5xc3". */
export function moveName(s, a) {
  const { from, to } = decode(a);
  if (from < 0) return `G ${squareName(to)}`;
  const piece = s.board[from] === TIGER ? 'T' : 'G';
  const jump = JUMP_OVER[from * N + to] >= 0;
  return `${piece} ${squareName(from)}${jump ? 'x' : '-'}${squareName(to)}`;
}

/** Replay a list of actions from the start position (used for undo, import and the worker). */
export function replay(actions) {
  let s = initialState();
  for (const a of actions) s = applyMove(s, a);
  return s;
}

/** The actions that led to `s`, oldest first. */
export function actionsOf(s) {
  const out = [];
  for (let p = s; p.prev; p = p.prev) out.push(p.lastMove);
  return out.reverse();
}

// Run with: node --test tests/web/
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  TIGER, GOAT, EMPTY, N, NEIGHBORS, JUMPS, initialState, legalMoves, applyMove,
  status, decode, moveAction, placeAction, moveName, replay, actionsOf, goatsOnBoard,
  threatenedGoats, unsafeSquares,
} from '../../docs/play/js/engine.js';
import { greedyAgent, minimaxAgent, mctsAgent, randomAgent, playGame } from '../../docs/play/js/agents.js';

const sq = (r, c) => r * 5 + c;

/** Build a state from a picture: T = tiger, G = goat, . = empty. */
function position(rows, { turn = TIGER, toPlace = 0, captured = null } = {}) {
  const s = initialState();
  s.board = new Int8Array(N);
  s.tigers = [];
  rows.join('').split('').forEach((ch, i) => {
    if (ch === 'T') { s.board[i] = TIGER; s.tigers.push(i); }
    if (ch === 'G') s.board[i] = GOAT;
  });
  s.turn = turn;
  s.toPlace = toPlace;
  s.captured = captured ?? GOAT_TOTAL_MINUS(s, toPlace);
  s.key = Math.random(); // unique; repetition is not under test here
  return s;
}
const GOAT_TOTAL_MINUS = (s, toPlace) => 20 - toPlace - goatsOnBoard(s);

test('board graph has the standard 56 lines', () => {
  const degree = NEIGHBORS.map((n) => n.length);
  assert.equal(degree.reduce((a, b) => a + b) / 2, 56);
  assert.equal(degree[sq(0, 0)], 3); // corner
  assert.equal(degree[sq(0, 1)], 3); // odd edge point: no diagonals
  assert.equal(degree[sq(0, 2)], 5); // even edge point
  assert.equal(degree[sq(1, 1)], 8);
  assert.equal(degree[sq(1, 2)], 4);
  assert.equal(degree[sq(2, 2)], 8);
  assert.deepEqual(JUMPS[sq(0, 0)].map(([, to]) => to).sort((a, b) => a - b), [sq(0, 2), sq(2, 0), sq(2, 2)]);
});

test('opening: 21 placements, then the tiger has 12 moves', () => {
  const s = initialState();
  assert.equal(legalMoves(s).length, 21);
  const t = applyMove(s, placeAction(sq(2, 2)));
  assert.equal(t.turn, TIGER);
  assert.equal(legalMoves(t).length, 12);
});

test('tiger captures by jumping over a goat onto an empty point', () => {
  let s = initialState();
  s = applyMove(s, placeAction(sq(0, 1)));
  const jump = moveAction(sq(0, 0), sq(0, 2));
  assert.ok(legalMoves(s).includes(jump));
  assert.deepEqual(decode(jump), { from: 0, to: 2, over: 1 });
  assert.equal(moveName(s, jump), 'T a5xc5');
  const t = applyMove(s, jump);
  assert.equal(t.captured, 1);
  assert.equal(t.board[sq(0, 1)], EMPTY);
  assert.equal(t.board[sq(0, 2)], TIGER);
  assert.ok(t.tigers.includes(sq(0, 2)));
  assert.deepEqual([...threatenedGoats(s)], [sq(0, 1)]);
});

test('no diagonal moves from odd points', () => {
  const s = position(['.....', '..T..', '.....', '.....', '.....']);
  const tos = legalMoves(s).map((a) => decode(a).to).sort((a, b) => a - b);
  assert.deepEqual(tos, [sq(0, 2), sq(1, 1), sq(1, 3), sq(2, 2)]);
});

test('trapped tigers lose; five captures win', () => {
  const trapped = position(['TGGGT', 'GG.GG', 'G.G.G', 'GG.GG', 'TGGGT'], { turn: TIGER });
  assert.deepEqual(status(trapped), { over: true, winner: GOAT, reason: 'all tigers trapped' });
  const s = position(['T...T', '.....', '.....', '.....', 'T...T'], { captured: 5 });
  assert.equal(status(s).winner, TIGER);
});

test('threefold repetition is a draw', () => {
  let s = position(['T...T', '.....', '..G..', '.....', 'T...T'], { turn: GOAT, captured: 0 });
  s.key = 0;
  const cycle = [moveAction(sq(2, 2), sq(2, 1)), moveAction(0, 1), moveAction(sq(2, 1), sq(2, 2)), moveAction(1, 0)];
  for (let i = 0; i < 2; i++) for (const a of cycle) { assert.equal(status(s).over, false); s = applyMove(s, a); }
  assert.equal(status(s).reason, 'threefold repetition');
});

test('unsafe squares for goat placement', () => {
  const s = initialState();
  // Any point next to a corner tiger with an empty point behind it is unsafe.
  const unsafe = unsafeSquares(s);
  assert.ok(unsafe.has(sq(0, 1)) && unsafe.has(sq(1, 1)) && !unsafe.has(sq(2, 2)));
});

test('random games keep invariants and always terminate', () => {
  for (let g = 0; g < 300; g++) {
    let s = initialState();
    for (;;) {
      const goats = goatsOnBoard(s);
      assert.equal(goats + s.captured + s.toPlace, 20);
      assert.equal(s.tigers.length, 4);
      for (const t of s.tigers) assert.equal(s.board[t], TIGER);
      const st = status(s);
      if (st.over) break;
      s = applyMove(s, randomAgent(s).move);
    }
    assert.deepEqual(replay(actionsOf(s)).board, s.board);
  }
});

test('greedy and minimax tigers take a winning capture', () => {
  const s = position(['TG..T', '.....', '.....', '.....', 'T...T'], { captured: 4 });
  const win = moveAction(sq(0, 0), sq(0, 2));
  assert.equal(greedyAgent(s).move, win);
  assert.equal(minimaxAgent(s, { timeMs: 200 }).move, win);
  assert.equal(mctsAgent(s, { timeMs: 200 }).move, win);
});

test('minimax goat traps the last tiger when it can', () => {
  // Only c2-c1 closes the gap on the bottom edge; any other move frees a tiger.
  const s = position(['TGGGT', 'GG.GG', 'G.G.G', 'GGGGG', 'TG.GT'], { turn: GOAT });
  const { move } = minimaxAgent(s, { timeMs: 300 });
  assert.equal(move, moveAction(sq(3, 2), sq(4, 2)));
  assert.equal(status(applyMove(s, move)).winner, GOAT);
});

test('minimax tiger beats a random goat', () => {
  let tigerWins = 0;
  for (let g = 0; g < 4; g++) {
    if (playGame({ kind: 'random' }, { kind: 'minimax', timeMs: 30, maxDepth: 3 }).winner === TIGER) tigerWins++;
  }
  assert.ok(tigerWins >= 3, `tiger won ${tigerWins}/4`);
});

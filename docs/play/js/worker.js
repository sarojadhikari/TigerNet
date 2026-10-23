/**
 * Search worker: runs agents off the main thread so the board stays responsive.
 *
 * Messages in:
 *   { id, type: 'move',  actions, spec }              -> { id, type: 'result', move, info }
 *   { id, type: 'arena', goat, tiger, games }          -> progress messages, then a final 'result'
 */
import { replay } from './engine.js';
import { runAgent, playGame } from './agents.js';

self.onmessage = (e) => {
  const msg = e.data;
  try {
    if (msg.type === 'move') {
      const { move, info } = runAgent(msg.spec, replay(msg.actions));
      self.postMessage({ id: msg.id, type: 'result', move, info });
    } else if (msg.type === 'arena') {
      const tally = { tiger: 0, goat: 0, draw: 0, plies: 0, reasons: {}, done: 0 };
      for (let g = 0; g < msg.games; g++) {
        const r = playGame(msg.goat, msg.tiger);
        tally[r.winner === 1 ? 'tiger' : r.winner === 2 ? 'goat' : 'draw']++;
        tally.plies += r.plies;
        tally.reasons[r.reason] = (tally.reasons[r.reason] ?? 0) + 1;
        tally.done = g + 1;
        self.postMessage({ id: msg.id, type: 'progress', tally });
      }
      self.postMessage({ id: msg.id, type: 'result', tally });
    }
  } catch (err) {
    self.postMessage({ id: msg.id, type: 'error', message: String(err?.stack ?? err) });
  }
};

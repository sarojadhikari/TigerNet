/**
 * Neural-network players running in the browser with onnxruntime-web.
 *
 * Two model contracts are supported:
 *
 * 1. Legacy TigerNet (docs/bg_trained_400_4.onnx). Input [1, 3, 5, 5] planes
 *    (empty, tiger, goat); output 100 sigmoid scores indexed by
 *    tiger_id * 25 + destination, where tiger_id follows the tigers' starting
 *    corners a5, a1, e5, e1. Tiger only.
 *
 * 2. Policy-value model from PLAN_RL_SELFPLAY.md, for either side.
 *    Input  "board"  float32 [1, 7, 5, 5]:
 *      0 empty, 1 tiger, 2 goat,
 *      3 side to move (all ones when the tiger moves, zeros for the goat),
 *      4 goats still to place / 20, 5 goats captured / 5,
 *      6 legal destination squares for the side to move.
 *    Output "policy" float32 [1, 650]: logits over the shared action space
 *           (0..24 placement, 25 + from * 25 + to for steps and jumps).
 *    Output "value"  float32 [1, 1]: expected result in [-1, 1] for the side to move.
 */
import { N, NUM_ACTIONS, TIGER, GOAT, EMPTY, GOATS_TOTAL, CAPTURES_TO_WIN, legalMoves, decode } from './engine.js';

const ORT_VERSION = '1.19.2';
const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
let ortPromise = null;

/** Load onnxruntime-web on first use, so the page stays light without neural players. */
function loadOrt() {
  if (!ortPromise) {
    ortPromise = new Promise((resolve, reject) => {
      if (globalThis.ort) return resolve(globalThis.ort);
      const script = document.createElement('script');
      script.src = `${ORT_BASE}ort.min.js`;
      script.onload = () => {
        globalThis.ort.env.wasm.wasmPaths = ORT_BASE;
        resolve(globalThis.ort);
      };
      script.onerror = () => {
        ortPromise = null;
        reject(new Error('Could not load onnxruntime-web'));
      };
      document.head.appendChild(script);
    });
  }
  return ortPromise;
}

/** Encode a state with the 7-plane policy-value input layout described above. */
export function encodeState(s) {
  const x = new Float32Array(7 * N);
  for (let sq = 0; sq < N; sq++) {
    x[sq] = s.board[sq] === EMPTY ? 1 : 0;
    x[N + sq] = s.board[sq] === TIGER ? 1 : 0;
    x[2 * N + sq] = s.board[sq] === GOAT ? 1 : 0;
    x[3 * N + sq] = s.turn === TIGER ? 1 : 0;
    x[4 * N + sq] = s.toPlace / GOATS_TOTAL;
    x[5 * N + sq] = s.captured / CAPTURES_TO_WIN;
  }
  for (const a of legalMoves(s)) x[6 * N + decode(a).to] = 1;
  return x;
}

/** Choose from (move, weight) pairs: argmax when greedy, else proportional sampling. */
function choose(moves, weights, greedy) {
  let best = 0;
  for (let i = 1; i < moves.length; i++) if (weights[i] > weights[best]) best = i;
  if (!greedy) {
    const total = weights.reduce((a, b) => a + b, 0);
    let r = Math.random() * total;
    for (let i = 0; i < moves.length; i++) {
      r -= weights[i];
      if (r <= 0) { best = i; break; }
    }
  }
  const total = weights.reduce((a, b) => a + b, 0) || 1;
  const top = moves
    .map((move, i) => ({ move, prob: weights[i] / total }))
    .sort((a, b) => b.prob - a.prob)
    .slice(0, 5);
  return { move: moves[best], top };
}

export class LegacyTigerNet {
  static async load(url) {
    const ort = await loadOrt();
    const net = new LegacyTigerNet();
    net.session = await ort.InferenceSession.create(url);
    net.ort = ort;
    return net;
  }

  get sides() { return [TIGER]; }

  async move(s, { greedy = false } = {}) {
    const x = new Float32Array(3 * N);
    for (let sq = 0; sq < N; sq++) {
      x[sq] = s.board[sq] === EMPTY ? 1 : 0;
      x[N + sq] = s.board[sq] === TIGER ? 1 : 0;
      x[2 * N + sq] = s.board[sq] === GOAT ? 1 : 0;
    }
    const input = new this.ort.Tensor('float32', x, [1, 3, 5, 5]);
    const out = await this.session.run({ [this.session.inputNames[0]]: input });
    const scores = out[this.session.outputNames[0]].data;
    const moves = legalMoves(s);
    const weights = moves.map((a) => {
      const { from, to } = decode(a);
      const v = scores[s.tigers.indexOf(from) * N + to];
      return Math.min(Math.max(v, 1e-4), 0.9999);
    });
    const { move, top } = choose(moves, weights, greedy);
    return { move, info: { top } };
  }
}

export class PolicyValueNet {
  static async load(source) {
    const ort = await loadOrt();
    const net = new PolicyValueNet();
    net.session = await ort.InferenceSession.create(source);
    net.ort = ort;
    for (const name of ['board']) {
      if (!net.session.inputNames.includes(name)) throw new Error(`model has no "${name}" input`);
    }
    for (const name of ['policy', 'value']) {
      if (!net.session.outputNames.includes(name)) throw new Error(`model has no "${name}" output`);
    }
    return net;
  }

  get sides() { return [GOAT, TIGER]; }

  async evaluate(s) {
    const input = new this.ort.Tensor('float32', encodeState(s), [1, 7, 5, 5]);
    const out = await this.session.run({ board: input });
    const logits = out.policy.data;
    if (logits.length !== NUM_ACTIONS) throw new Error(`policy has ${logits.length} entries, expected ${NUM_ACTIONS}`);
    return { logits, value: out.value.data[0] };
  }

  async move(s, { greedy = false, temperature = 1 } = {}) {
    const { logits, value } = await this.evaluate(s);
    const moves = legalMoves(s);
    const max = Math.max(...moves.map((a) => logits[a]));
    const weights = moves.map((a) => Math.exp((logits[a] - max) / temperature));
    const { move, top } = choose(moves, weights, greedy);
    return { move, info: { top, value } };
  }
}

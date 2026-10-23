import {
  N, TIGER, GOAT, EMPTY, NEIGHBORS, JUMP_OVER, row, col, initialState, legalMoves, applyMove,
  status, decode, moveAction, moveName, actionsOf, threatenedGoats, unsafeSquares,
} from './engine.js';
import { LegacyTigerNet, PolicyValueNet } from './neural.js';

// ------------------------------------------------------------ configuration

const KINDS = [
  { id: 'human', label: 'Human' },
  { id: 'random', label: 'Random', search: true, noLevel: true },
  { id: 'greedy', label: 'Greedy', search: true, noLevel: true },
  { id: 'minimax', label: 'Minimax', search: true },
  { id: 'mcts', label: 'MCTS', search: true },
  { id: 'legacy', label: 'TigerNet (legacy)', sides: [TIGER] },
  { id: 'neural', label: 'Neural (loaded model)', needsModel: true },
];
const LEVELS = {
  easy: { label: 'Easy', timeMs: 150, maxDepth: 2, greedy: false, temperature: 1.5 },
  medium: { label: 'Medium', timeMs: 700, greedy: false, temperature: 0.5 },
  hard: { label: 'Hard', timeMs: 2000, greedy: true, temperature: 0.1 },
};
const LEGACY_MODEL_URL = '../bg_trained_400_4.onnx';
const ANALYSIS_MS = 1200;

const SIDE_NAME = { [GOAT]: 'Goat', [TIGER]: 'Tiger' };
const kindById = (id) => KINDS.find((k) => k.id === id);

// -------------------------------------------------------------- app state

const app = {
  history: [initialState()],
  index: 0,
  selected: -1,
  hint: null,         // { move }
  analysis: null,     // { top, value, iterations, forIndex }
  paused: false,
  thinking: null,     // side currently thinking
  gen: 0,             // bumps whenever pending AI results become stale
  lastCaptureAt: -1,
  legacyNet: null,
  model: null,
};
const current = () => app.history[app.index];
const atEnd = () => app.index === app.history.length - 1;

const $ = (id) => document.getElementById(id);

// ------------------------------------------------------------ worker pool

class Cancelled extends Error {}

/** A restartable worker: cancel() terminates in-flight searches immediately. */
class EngineWorker {
  constructor() { this.nextId = 1; this.pending = new Map(); this.spawn(); }

  spawn() {
    this.worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
    this.worker.onmessage = (e) => {
      const msg = e.data, p = this.pending.get(msg.id);
      if (!p) return;
      if (msg.type === 'progress') { p.onProgress?.(msg.tally); return; }
      this.pending.delete(msg.id);
      if (msg.type === 'error') p.reject(new Error(msg.message));
      else p.resolve(msg);
    };
  }

  request(msg, onProgress) {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject, onProgress });
      this.worker.postMessage({ ...msg, id });
    });
  }

  cancel() {
    if (this.pending.size === 0) return;
    this.worker.terminate();
    for (const p of this.pending.values()) p.reject(new Cancelled());
    this.pending.clear();
    this.spawn();
  }
}

const playWorker = new EngineWorker();
const analysisWorker = new EngineWorker();
const arenaWorker = new EngineWorker();

// ------------------------------------------------------------ board (SVG)

const SVG_NS = 'http://www.w3.org/2000/svg';
const X = (sq) => 60 + col(sq) * 100;
const Y = (sq) => 60 + row(sq) * 100;

function el(tag, attrs = {}, parent = null) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (parent) parent.appendChild(node);
  return node;
}

const svg = $('board');
let layerMarks, layerPieces, layerOverlay;

function buildBoard() {
  const defs = el('defs', {}, svg);
  const marker = el('marker', {
    id: 'arrowhead', viewBox: '0 0 10 10', refX: 6, refY: 5, markerWidth: 4, markerHeight: 4, orient: 'auto-start-reverse',
  }, defs);
  el('path', { d: 'M0,0 L10,5 L0,10 z', class: 'arrow-head' }, marker);

  const lines = el('g', {}, svg);
  for (let a = 0; a < N; a++) {
    for (const b of NEIGHBORS[a]) {
      if (b > a) el('line', { x1: X(a), y1: Y(a), x2: X(b), y2: Y(b), class: 'line' }, lines);
    }
  }
  const coords = el('g', {}, svg);
  for (let i = 0; i < 5; i++) {
    el('text', { x: 60 + i * 100, y: 503, class: 'coord' }, coords).textContent = 'abcde'[i];
    el('text', { x: 20, y: 60 + i * 100, class: 'coord' }, coords).textContent = String(5 - i);
  }
  layerMarks = el('g', {}, svg);
  layerPieces = el('g', {}, svg);
  layerOverlay = el('g', {}, svg);
  const hits = el('g', {}, svg);
  for (let sq = 0; sq < N; sq++) {
    const hit = el('circle', { cx: X(sq), cy: Y(sq), r: 44, class: 'hit', 'data-sq': sq }, hits);
    hit.addEventListener('click', () => onPoint(sq));
  }
}

function drawTiger(sq, parent) {
  const g = el('g', { class: 'piece', transform: `translate(${X(sq)} ${Y(sq)})` }, parent);
  el('circle', { r: 30, class: 'tiger-body' }, g);
  el('path', { d: 'M-16,-14 q6,8 0,16 M0,-20 q6,10 0,20 M16,-14 q6,8 0,16', class: 'tiger-stripe' }, g);
}

function drawGoat(sq, parent) {
  const g = el('g', { class: 'piece', transform: `translate(${X(sq)} ${Y(sq)})` }, parent);
  el('circle', { r: 23, class: 'goat-body' }, g);
  el('circle', { r: 3.5, cx: -7, cy: -3, class: 'goat-mark' }, g);
  el('circle', { r: 3.5, cx: 7, cy: -3, class: 'goat-mark' }, g);
}

function drawArrow(move, width, label, parent) {
  const { from, to } = decode(move);
  if (from < 0) {
    el('circle', { cx: X(to), cy: Y(to), r: 32, class: 'place-ring', 'stroke-width': width }, parent);
  } else {
    const x1 = X(from), y1 = Y(from), x2 = X(to), y2 = Y(to);
    const len = Math.hypot(x2 - x1, y2 - y1), ux = (x2 - x1) / len, uy = (y2 - y1) / len;
    el('line', {
      x1: x1 + ux * 30, y1: y1 + uy * 30, x2: x2 - ux * 34, y2: y2 - uy * 34,
      class: 'arrow', 'stroke-width': width, 'marker-end': 'url(#arrowhead)',
    }, parent);
  }
  if (label) {
    const t = el('text', { x: X(to) + 26, y: Y(to) - 26, class: 'arrow-label' }, parent);
    t.textContent = label;
  }
}

function renderBoard() {
  const s = current();
  layerMarks.replaceChildren();
  layerPieces.replaceChildren();
  layerOverlay.replaceChildren();

  // Last move highlight
  if (s.lastMove >= 0) {
    const { from, to } = decode(s.lastMove);
    for (const sq of from >= 0 ? [from, to] : [to]) el('circle', { cx: X(sq), cy: Y(sq), r: 40, class: 'last' }, layerMarks);
  }

  const human = isHuman(s.turn) && !status(s).over && atEnd();
  const showDanger = $('opt-danger').checked;
  if (showDanger) {
    for (const sq of threatenedGoats(s)) el('circle', { cx: X(sq), cy: Y(sq), r: 30, class: 'threat' }, layerMarks);
  }

  for (let sq = 0; sq < N; sq++) {
    if (s.board[sq] === TIGER) drawTiger(sq, layerPieces);
    else if (s.board[sq] === GOAT) drawGoat(sq, layerPieces);
  }

  // Selection and legal destinations. Goat destinations that could be jumped at once are red.
  if (human) {
    let unsafe = new Set();
    if (showDanger && s.turn === GOAT && (s.toPlace > 0 || app.selected >= 0)) {
      unsafe = unsafeSquares(s, s.toPlace > 0 ? -1 : app.selected);
    }
    if (app.selected >= 0) {
      el('circle', { cx: X(app.selected), cy: Y(app.selected), r: 36, class: 'selected' }, layerOverlay);
    }
    for (const a of legalMoves(s)) {
      const { from, to } = decode(a);
      if (from >= 0 && from !== app.selected) continue;
      const isJump = from >= 0 && JUMP_OVER[from * N + to] >= 0;
      if (isJump) el('circle', { cx: X(to), cy: Y(to), r: 16, class: 'legal-jump' }, layerOverlay);
      else if (unsafe.has(to)) el('circle', { cx: X(to), cy: Y(to), r: 9, class: 'unsafe' }, layerOverlay);
      else if ($('opt-legal').checked) el('circle', { cx: X(to), cy: Y(to), r: from < 0 ? 6 : 9, class: 'legal' }, layerOverlay);
    }
  }

  // Capture flash
  if (app.lastCaptureAt >= 0 && s.lastMove >= 0 && decode(s.lastMove).over === app.lastCaptureAt && s.board[app.lastCaptureAt] === EMPTY) {
    const x = X(app.lastCaptureAt), y = Y(app.lastCaptureAt);
    el('path', { d: `M${x - 14},${y - 14} L${x + 14},${y + 14} M${x + 14},${y - 14} L${x - 14},${y + 14}`, class: 'captured-x' }, layerOverlay);
  }

  // Analysis arrows
  if (app.analysis && app.analysis.forIndex === app.index && $('opt-analysis').checked) {
    const total = app.analysis.top.reduce((a, t) => a + t.visits, 0) || 1;
    app.analysis.top.slice(0, 3).forEach((t, i) => {
      drawArrow(t.move, i === 0 ? 9 : 5, `${Math.round((100 * t.visits) / total)}%`, layerOverlay);
    });
  }
  if (app.hint && app.hint.forIndex === app.index) drawArrow(app.hint.move, 10, '', layerOverlay);
}

// --------------------------------------------------------------- panels

function renderStatus() {
  const s = current(), st = status(s);
  $('to-place').textContent = s.toPlace;
  $('pips').replaceChildren(...Array.from({ length: 5 }, (_, i) => {
    const pip = document.createElement('span');
    pip.className = `pip${i < s.captured ? ' on' : ''}`;
    return pip;
  }));

  const turn = $('turn');
  turn.replaceChildren();
  const sw = document.createElement('span');
  sw.className = `swatch ${s.turn === TIGER ? 'tiger' : 'goat'}`;
  turn.append(sw, st.over ? 'Game over' : `${SIDE_NAME[s.turn]} to move${s.turn === GOAT && s.toPlace > 0 ? ' (place)' : ''}`);

  const banner = $('banner');
  if (st.over) {
    banner.hidden = false;
    banner.innerHTML = '';
    banner.append(st.winner ? `${SIDE_NAME[st.winner]}s win` : 'Draw');
    const small = document.createElement('small');
    small.textContent = st.reason;
    banner.append(small);
  } else {
    banner.hidden = true;
  }

  $('thinking').hidden = app.thinking === null;
  if (app.thinking !== null) {
    $('thinking-text').textContent = `${SIDE_NAME[app.thinking]} (${kindById(playerSpec(app.thinking).kind).label}) is thinking…`;
  }

  $('btn-undo').disabled = app.index === 0;
  $('btn-redo').disabled = atEnd();
  $('btn-hint').disabled = st.over;
  $('btn-pause').textContent = app.paused ? 'Resume' : 'Pause';
  $('btn-pause').disabled = isHuman(GOAT) && isHuman(TIGER);
}

function renderMoves() {
  const list = $('move-list');
  const final = app.history[app.history.length - 1];
  const actions = actionsOf(final);
  const items = actions.map((a, i) => {
    const li = document.createElement('li');
    li.textContent = moveName(app.history[i], a);
    if (i === app.index - 1) li.className = 'current';
    li.addEventListener('click', () => jumpTo(i + 1));
    return li;
  });
  list.replaceChildren(...items);
  list.querySelector('.current')?.scrollIntoView({ block: 'nearest' });
}

function renderAnalysis() {
  const on = $('opt-analysis').checked;
  $('analysis').hidden = !on;
  $('evalbar').hidden = !on;
  if (!on) return;
  const s = current(), a = app.analysis;
  const list = $('analysis-list');
  if (status(s).over) {
    list.replaceChildren();
    $('analysis-meta').textContent = 'Game over.';
    setEval(status(s).winner === TIGER ? 1 : status(s).winner === GOAT ? -1 : 0);
    return;
  }
  if (!a || a.forIndex !== app.index) {
    $('analysis-meta').textContent = 'Analysing…';
    return;
  }
  const total = a.top.reduce((x, t) => x + t.visits, 0) || 1;
  list.replaceChildren(...a.top.map((t) => {
    const li = document.createElement('li');
    const pct = Math.round((100 * t.visits) / total);
    li.textContent = `${moveName(s, t.move)} · ${pct}% visits · score ${t.value >= 0 ? '+' : ''}${t.value.toFixed(2)}`;
    return li;
  }));
  const tigerValue = s.turn === TIGER ? a.value : -a.value;
  setEval(tigerValue);
  let meta = `${a.iterations.toLocaleString()} playouts. Score is for the side to move (−1 to +1).`;
  if (a.modelValue !== undefined) meta += ` Loaded model value: ${a.modelValue >= 0 ? '+' : ''}${a.modelValue.toFixed(2)}.`;
  $('analysis-meta').textContent = meta;
}

function setEval(tigerValue) {
  const pct = Math.round(((tigerValue + 1) / 2) * 100);
  $('evalbar-fill').style.height = `${pct}%`;
  $('evalbar-label').textContent = tigerValue >= 0 ? `T +${tigerValue.toFixed(1)}` : `G +${(-tigerValue).toFixed(1)}`;
}

function render() {
  renderBoard();
  renderStatus();
  renderMoves();
  renderAnalysis();
}

// ----------------------------------------------------------- players

function playerSpec(side) {
  const prefix = side === GOAT ? 'goat' : 'tiger';
  const kind = $(`${prefix}-kind`).value;
  return { kind, ...LEVELS[$(`${prefix}-level`).value] };
}
const isHuman = (side) => playerSpec(side).kind === 'human';

function fillKindSelect(select, side, { searchOnly = false } = {}) {
  const value = select.value;
  select.replaceChildren(...KINDS.filter((k) => {
    if (searchOnly) return k.search;
    if (k.sides && !k.sides.includes(side)) return false;
    return true;
  }).map((k) => {
    const opt = document.createElement('option');
    opt.value = k.id;
    opt.textContent = k.label;
    opt.disabled = Boolean(k.needsModel && !app.model);
    return opt;
  }));
  if (value && [...select.options].some((o) => o.value === value && !o.disabled)) select.value = value;
}

function fillLevelSelect(select, value) {
  select.replaceChildren(...Object.entries(LEVELS).map(([id, l]) => {
    const opt = document.createElement('option');
    opt.value = id;
    opt.textContent = l.label;
    return opt;
  }));
  select.value = value;
}

function syncLevelEnabled() {
  for (const p of ['goat', 'tiger', 'arena-goat', 'arena-tiger']) {
    const kind = kindById($(p === 'goat' || p === 'tiger' ? `${p}-kind` : p).value);
    $(`${p}-level`).disabled = !kind || kind.id === 'human' || Boolean(kind.noLevel);
  }
}

async function getMove(spec, s) {
  const kind = kindById(spec.kind);
  if (kind.search) {
    const res = await playWorker.request({ type: 'move', actions: actionsOf(s), spec });
    return res;
  }
  if (spec.kind === 'legacy') {
    app.legacyNet ??= await LegacyTigerNet.load(LEGACY_MODEL_URL);
    return app.legacyNet.move(s, spec);
  }
  if (spec.kind === 'neural') return app.model.move(s, spec);
  throw new Error(`no engine for ${spec.kind}`);
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** Let the AI move if it is an AI's turn at the end of the history. */
async function maybeAIMove() {
  const s = current();
  if (app.paused || !atEnd() || status(s).over || isHuman(s.turn) || app.thinking !== null) return;
  const gen = ++app.gen;
  const side = s.turn;
  app.thinking = side;
  renderStatus();
  try {
    const started = performance.now();
    const res = await getMove(playerSpec(side), s);
    const wait = Number($('opt-delay').value) - (performance.now() - started);
    if (wait > 0) await sleep(wait);
    if (gen !== app.gen) return;
    app.thinking = null;
    commitMove(res.move);
  } catch (err) {
    if (gen === app.gen) {
      app.thinking = null;
      app.paused = true;
      renderStatus();
      if (!(err instanceof Cancelled)) toast(`AI error: ${err.message}`);
    }
  }
}

/** Invalidate pending AI work (moves, hints, analysis). */
function cancelPending() {
  app.gen++;
  app.thinking = null;
  playWorker.cancel();
}

// ------------------------------------------------------------- moves

function commitMove(move) {
  const s = current();
  app.history = app.history.slice(0, app.index + 1);
  const next = applyMove(s, move);
  app.history.push(next);
  app.index++;
  app.selected = -1;
  app.hint = null;
  app.lastCaptureAt = next.captured > s.captured ? decode(move).over : -1;
  afterPositionChange();
}

function afterPositionChange() {
  render();
  updateHash();
  requestAnalysis();
  maybeAIMove();
}

function onPoint(sq) {
  const s = current();
  if (!atEnd() || status(s).over || !isHuman(s.turn) || app.thinking !== null) return;
  const legal = legalMoves(s);
  if (s.turn === GOAT && s.toPlace > 0) {
    if (legal.includes(sq)) commitMove(sq);
    return;
  }
  const own = s.turn === TIGER ? TIGER : GOAT;
  if (app.selected >= 0 && legal.includes(moveAction(app.selected, sq))) {
    commitMove(moveAction(app.selected, sq));
    return;
  }
  app.selected = s.board[sq] === own && app.selected !== sq ? sq : -1;
  renderBoard();
}

function jumpTo(index) {
  cancelPending();
  app.index = Math.max(0, Math.min(index, app.history.length - 1));
  app.selected = -1;
  app.hint = null;
  app.lastCaptureAt = -1;
  // Reviewing an earlier position pauses AI-vs-AI play so it doesn't overwrite the game.
  if (!atEnd() && !isHuman(GOAT) && !isHuman(TIGER)) app.paused = true;
  afterPositionChange();
}

function undo() {
  if (app.index === 0) return;
  let i = app.index - 1;
  // Step back to the last position where a human is to move.
  while (i > 0 && !isHuman(app.history[i].turn) && (isHuman(GOAT) || isHuman(TIGER))) i--;
  jumpTo(i);
}

function redo() {
  if (atEnd()) return;
  let i = app.index + 1;
  while (i < app.history.length - 1 && !isHuman(app.history[i].turn) && (isHuman(GOAT) || isHuman(TIGER))) i++;
  jumpTo(i);
}

function newGame() {
  cancelPending();
  app.history = [initialState()];
  app.index = 0;
  app.selected = -1;
  app.hint = null;
  app.analysis = null;
  app.lastCaptureAt = -1;
  app.paused = false;
  afterPositionChange();
}

// ----------------------------------------------------- hint & analysis

async function hint() {
  const s = current();
  if (status(s).over) return;
  const index = app.index;
  try {
    const res = await analysisWorker.request({ type: 'move', actions: actionsOf(s), spec: { kind: 'minimax', timeMs: 1200 } });
    if (index !== app.index) return;
    app.hint = { move: res.move, forIndex: index };
    toast(`Hint: ${moveName(s, res.move)}${res.info.mate ? ' (forced result found)' : ''}`);
    renderBoard();
  } catch (err) {
    if (!(err instanceof Cancelled)) toast(err.message);
  }
}

async function requestAnalysis() {
  analysisWorker.cancel();
  if (!$('opt-analysis').checked) { renderAnalysis(); return; }
  const s = current(), index = app.index;
  if (status(s).over) { renderAnalysis(); return; }
  try {
    const res = await analysisWorker.request({ type: 'move', actions: actionsOf(s), spec: { kind: 'mcts', timeMs: ANALYSIS_MS } });
    if (index !== app.index) return;
    app.analysis = { ...res.info, forIndex: index };
    if (app.model) {
      const { value } = await app.model.evaluate(s);
      if (index === app.index) app.analysis.modelValue = value;
    }
    renderBoard();
    renderAnalysis();
  } catch (err) {
    if (!(err instanceof Cancelled)) toast(err.message);
  }
}

// ------------------------------------------------------------- arena

async function runArena() {
  const spec = (p) => ({ kind: $(p).value, ...LEVELS[$(`${p}-level`).value] });
  const games = Math.max(1, Math.min(500, Number($('arena-games').value) || 1));
  $('arena-run').disabled = true;
  $('arena-stop').disabled = false;
  $('arena-results').hidden = false;
  showTally({ tiger: 0, goat: 0, draw: 0, plies: 0, reasons: {}, done: 0 }, games);
  try {
    const res = await arenaWorker.request(
      { type: 'arena', goat: spec('arena-goat'), tiger: spec('arena-tiger'), games },
      (tally) => showTally(tally, games),
    );
    showTally(res.tally, games);
  } catch (err) {
    if (!(err instanceof Cancelled)) toast(err.message);
  } finally {
    $('arena-run').disabled = false;
    $('arena-stop').disabled = true;
  }
}

function showTally(t, games) {
  const [g, d, ti] = $('arena-bar').children;
  const pct = (n) => `${(100 * n) / games}%`;
  g.style.width = pct(t.goat);
  d.style.width = pct(t.draw);
  ti.style.width = pct(t.tiger);
  $('arena-goat-wins').textContent = t.goat;
  $('arena-draws').textContent = t.draw;
  $('arena-tiger-wins').textContent = t.tiger;
  $('arena-plies').textContent = t.done ? Math.round(t.plies / t.done) : 0;
  $('arena-reasons').replaceChildren(...Object.entries(t.reasons).map(([r, n]) => {
    const li = document.createElement('li');
    li.textContent = `${r}: ${n}`;
    return li;
  }), Object.assign(document.createElement('li'), { textContent: `${t.done} / ${games} games played` }));
}

// ------------------------------------------------------ model loading

async function loadModelFile(file) {
  $('model-status').textContent = 'Loading…';
  try {
    app.model = await PolicyValueNet.load(new Uint8Array(await file.arrayBuffer()));
    $('model-status').textContent = `Loaded ${file.name}`;
  } catch (err) {
    app.model = null;
    $('model-status').textContent = `Could not load: ${err.message}`;
  }
  fillKindSelect($('goat-kind'), GOAT);
  fillKindSelect($('tiger-kind'), TIGER);
  syncLevelEnabled();
}

// ---------------------------------------------------- sharing & misc

function updateHash() {
  const actions = actionsOf(app.history[app.history.length - 1]).slice(0, app.index);
  const hash = actions.length ? `#m=${actions.join('.')}` : '';
  history.replaceState(null, '', `${location.pathname}${location.search}${hash}`);
}

function loadFromHash() {
  const m = /^#m=([\d.]+)$/.exec(location.hash);
  if (!m) return;
  let s = initialState();
  const states = [s];
  for (const a of m[1].split('.').map(Number)) {
    if (status(s).over || !legalMoves(s).includes(a)) break;
    s = applyMove(s, a);
    states.push(s);
  }
  app.history = states;
  app.index = states.length - 1;
}

let toastTimer = null;
function toast(text) {
  const t = $('toast');
  t.textContent = text;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, 2600);
}

function setTab(name) {
  for (const b of document.querySelectorAll('.tab')) {
    const on = b.dataset.tab === name;
    b.classList.toggle('active', on);
    b.setAttribute('aria-selected', String(on));
  }
  for (const pane of document.querySelectorAll('.tabpane')) pane.hidden = pane.id !== `tab-${name}`;
}

// ------------------------------------------------------------- wiring

function init() {
  buildBoard();
  fillKindSelect($('goat-kind'), GOAT);
  fillKindSelect($('tiger-kind'), TIGER);
  fillKindSelect($('arena-goat'), GOAT, { searchOnly: true });
  fillKindSelect($('arena-tiger'), TIGER, { searchOnly: true });
  $('goat-kind').value = 'human';
  $('tiger-kind').value = 'minimax';
  $('arena-goat').value = 'minimax';
  $('arena-tiger').value = 'mcts';
  fillLevelSelect($('goat-level'), 'medium');
  fillLevelSelect($('tiger-level'), 'medium');
  fillLevelSelect($('arena-goat-level'), 'easy');
  fillLevelSelect($('arena-tiger-level'), 'easy');
  syncLevelEnabled();

  for (const id of ['goat-kind', 'goat-level', 'tiger-kind', 'tiger-level']) {
    $(id).addEventListener('change', () => {
      syncLevelEnabled();
      cancelPending();
      app.selected = -1;
      render();
      maybeAIMove();
    });
  }
  for (const id of ['arena-goat', 'arena-tiger']) $(id).addEventListener('change', syncLevelEnabled);

  $('btn-new').addEventListener('click', newGame);
  $('btn-undo').addEventListener('click', undo);
  $('btn-redo').addEventListener('click', redo);
  $('btn-hint').addEventListener('click', hint);
  $('btn-pause').addEventListener('click', () => {
    app.paused = !app.paused;
    if (app.paused) cancelPending();
    renderStatus();
    maybeAIMove();
  });
  $('btn-share').addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(location.href);
      toast('Link to this position copied');
    } catch {
      toast(location.href);
    }
  });
  $('opt-legal').addEventListener('change', renderBoard);
  $('opt-danger').addEventListener('change', renderBoard);
  $('opt-analysis').addEventListener('change', () => { render(); requestAnalysis(); });
  $('model-file').addEventListener('change', (e) => { if (e.target.files[0]) loadModelFile(e.target.files[0]); });
  $('arena-run').addEventListener('click', runArena);
  $('arena-stop').addEventListener('click', () => arenaWorker.cancel());
  for (const b of document.querySelectorAll('.tab')) b.addEventListener('click', () => setTab(b.dataset.tab));

  document.addEventListener('keydown', (e) => {
    if (e.target.closest('input, select')) return;
    const mod = e.ctrlKey || e.metaKey;
    if (mod && e.key.toLowerCase() === 'z' && !e.shiftKey) { e.preventDefault(); undo(); }
    else if (mod && (e.key.toLowerCase() === 'y' || (e.key.toLowerCase() === 'z' && e.shiftKey))) { e.preventDefault(); redo(); }
    else if (e.key === 'Escape') { app.selected = -1; renderBoard(); }
  });

  loadFromHash();
  afterPositionChange();
}

init();

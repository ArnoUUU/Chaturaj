"use strict";
// Browser side of gui/play.py: setup, board, clocks, pause, seat changes and review.

const ORDER = "RBYG";
const NAMES = { R: "Red", B: "Blue", Y: "Yellow", G: "Green" };
const COLORS = { R: "#BF3B43", B: "#4185BF", Y: "#C09526", G: "#4E9161" };
const CUSTOM = "custom";

const $ = (id) => document.getElementById(id);
const ESC = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ESC[c]);

const app = {
  assets: null,       // {sprites, views, clocks}
  bots: null,         // {options, default}
  spriteURL: {},      // piece code -> data URL
  squares: [],        // 64 board cells in display order
  S: null,            // latest snapshot from the server (null when there is no game)
  view: null,         // null = live game; a number = reviewing the position after that many moves
  got: 0,             // performance.now() when S arrived (for the running clock)
  sel: null,          // selected board index (a person to move)
  busy: false,
  lastSeats: null,
  lastTime: "1|5",
  lastView: "R",      // which seat's point of view the board shows
};

// --- helpers ----------------------------------------------------------------------------

async function get(path) {
  const r = await fetch(path, { cache: "no-store" });
  return r.json();
}

async function post(path, body) {
  const r = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  return r.json();
}

function fmt(t) {
  t = Math.max(0, t);
  if (t < 10) return t.toFixed(1);
  const m = Math.floor(t / 60), s = Math.floor(t % 60);
  return m + ":" + String(s).padStart(2, "0");
}

// Time left for seat c, counting down the move in progress.
function clockLeft(c) {
  let t = app.S.clock[c];
  if (app.S.running === c && !app.S.paused) t -= (performance.now() - app.got) / 1000;
  return t;
}

function botLabel(ref) {
  const opt = app.bots.options.find((o) => o.value === ref);
  return opt ? opt.label : ref;
}

function optionsFor(select) {
  for (const o of app.bots.options) {
    const op = document.createElement("option");
    op.value = o.value;
    op.textContent = o.label;
    select.appendChild(op);
  }
  return select;
}

// --- setup screen -----------------------------------------------------------------------

function buildSetup() {
  const box = $("seats");
  box.innerHTML = "";
  for (const c of ORDER) {
    const row = document.createElement("div");
    row.className = "seatrow";
    row.innerHTML = `<span class="swatch" style="background:${COLORS[c]}"></span><span class="sname">${NAMES[c]}</span>`;
    const sel = optionsFor(document.createElement("select"));
    sel.id = "seat-" + c;
    const want = app.lastSeats ? app.lastSeats[c] : (c === "R" ? "human" : app.bots.default);
    sel.value = [...sel.options].some((o) => o.value === want) ? want : app.bots.default;
    row.appendChild(sel);
    box.appendChild(row);
  }
  syncTimeControl(app.lastTime);
  $("setup-err").textContent = "";
}

// The time control: a preset from the dropdown, or a custom value typed in.
function syncTimeControl(value) {
  // the controls the scraper has seen, most played first; each is {value, label, games}
  const presets = app.assets.clocks;
  const preset = $("time-preset");
  preset.innerHTML = "";
  for (const p of presets) {
    const o = document.createElement("option");
    o.value = p.value;
    o.textContent = p.games ? `${p.label}  (${p.games.toLocaleString()} games)` : p.label;
    preset.appendChild(o);
  }
  const custom = document.createElement("option");
  custom.value = CUSTOM;
  custom.textContent = "Custom…";
  preset.appendChild(custom);
  if (presets.some((p) => p.value === value)) {
    preset.value = value;
    $("time-custom-row").hidden = true;
  } else {
    preset.value = CUSTOM;
    $("time-custom").value = value;
    $("time-custom-row").hidden = false;
  }
}

function chosenTime() {
  const v = $("time-preset").value;
  return v === CUSTOM ? $("time-custom").value.trim() : v;
}

async function startGame() {
  const seats = {};
  for (const c of ORDER) seats[c] = $("seat-" + c).value;
  const time = chosenTime() || "1|5";
  const humanClock = $("human-clock").checked;
  $("start").disabled = true;
  const r = await post("/api/new", { seats, time, human_clock: humanClock });
  $("start").disabled = false;
  if (r.error) {
    $("setup-err").textContent = r.error;
    return;
  }
  app.lastSeats = seats;
  app.lastTime = time;
  const people = ORDER.split("").filter((c) => seats[c] === "human").length;
  $("flip").checked = people > 1;
  app.view = null;
  app.sel = null;
  await poll();
}

// --- board and game view ----------------------------------------------------------------

// The seat whose point of view the board shows.
function viewSeat() {
  const S = app.S;
  if ($("flip").checked && S.t && S.humans.includes(S.t)) {
    app.lastView = S.t;
  } else if (!S.humans.includes(app.lastView)) {
    app.lastView = S.humans[0] || "R";
  }
  return app.lastView;
}

function renderBoard() {
  const S = app.S;
  const view = app.assets.views[viewSeat()];
  const targets = app.sel === null ? [] : S.legal.filter((m) => m[0] === app.sel).map((m) => m[1]);
  for (let d = 0; d < 64; d++) {
    const i = view[d];
    const p = S.b.substr(i * 2, 2);
    const cell = app.squares[d];
    cell.style.backgroundImage = p === ".." ? "" : `url(${app.spriteURL[p]})`;
    cell.classList.toggle("last", !!S.last && (S.last[0] === i || S.last[1] === i));
    cell.classList.toggle("sel", app.sel === i);
    const hit = targets.includes(i);
    cell.classList.toggle("dot", hit && p === "..");
    cell.classList.toggle("cap", hit && p !== "..");
  }
}

function renderClocks() {
  const S = app.S;
  $("clocks").innerHTML = ORDER.split("").map((c, i) => {
    const seat = S.seats[i];
    const cls = "clock" + (S.e.includes(c) ? " out" : "") + (S.t === c && !app.view ? " turn" : "");
    const label = seat.human ? seat.name : `${seat.name} · bot`;
    const untimed = S.clock[c] === null;
    const time = untimed
      ? `<div class="ck none">no clock</div>`
      : `<div class="ck${clockLeft(c) < 10 ? " low" : ""}">${fmt(clockLeft(c))}</div>`;
    return `<div class="${cls}" style="background:${COLORS[c]}">
      <div class="who" title="${esc(label)}">${esc(label)}</div>
      <span class="score">${S.s[i]}</span>${time}</div>`;
  }).join("");
}

function reviewing() {
  return app.view !== null;
}

function renderStatus() {
  const S = app.S;
  let text = "";
  if (reviewing()) {
    text = `Reviewing: after move ${app.view} of ${S.ply}`;
  } else if (S.result) {
    const winners = S.result.winners.map((c) => NAMES[c]).join(" & ");
    text = `Game over (${S.result.reason}). Winner: ${winners}.` + (S.saved ? " Saved to games/played." : "");
  } else if (S.paused) {
    text = "Paused";
  } else if (S.t && S.humans.includes(S.t)) {
    text = `Your move: ${NAMES[S.t]}`;
  } else if (S.t) {
    text = `${NAMES[S.t]} is thinking…`;
  }
  $("status").textContent = text;
  const people = S.humans.length ? "people: " + S.humans.map((c) => NAMES[c]).join(", ") : "bots only";
  const clockNote = S.human_clock ? "" : " · no clock for people";
  $("tc").textContent = `Time control ${S.tc}${clockNote} · ${people}`;
}

// Dropdown per side: who plays it. Changing it takes effect at once.
function renderPlayers() {
  const S = app.S;
  if (app.playersBuilt !== S.ply + ":" + S.seats.map((s) => s.bot || "h").join(",")) {
    app.playersBuilt = S.ply + ":" + S.seats.map((s) => s.bot || "h").join(",");
    $("players").innerHTML = "";
    for (const seat of S.seats) {
      const c = seat.color;
      const row = document.createElement("div");
      row.className = "player";
      row.innerHTML = `<span class="swatch" style="background:${COLORS[c]}"></span><span class="sname">${NAMES[c]}</span>`;
      const sel = optionsFor(document.createElement("select"));
      sel.dataset.color = c;
      sel.value = seat.bot || "human";
      sel.addEventListener("change", () => changeSeat(c, sel.value));
      row.appendChild(sel);
      $("players").appendChild(row);
    }
  }
}

async function changeSeat(color, value) {
  const r = await post("/api/seat", { color, value });
  if (r.error) $("status").textContent = r.error;
  app.sel = null;
  await poll();
}

function renderMoves() {
  const S = app.S;
  const shown = reviewing() ? app.view : S.ply;
  const rows = S.moves.map((m) => {
    const cur = m.n === shown ? " cur" : "";
    const side = `<span class="dot" style="background:${COLORS[m.color]}"></span>${esc(m.san || "")}`;
    const pts = m.points ? `+${m.points}` : "";
    const clk = m.clock === null || m.clock === undefined ? "" : fmt(m.clock);
    return `<div class="mrow${cur}" data-ply="${m.n}" role="listitem">
      <span class="num">${m.n}.</span><span class="san">${side}</span>
      <span class="sub">${[pts, clk].filter(Boolean).join(" · ")}</span></div>`;
  });
  const list = $("moves");
  const atBottom = list.scrollTop + list.clientHeight >= list.scrollHeight - 8;
  list.innerHTML = rows.length ? rows.join("") : `<div class="empty">No moves yet.</div>`;
  const cur = list.querySelector(".cur");
  if (cur) {
    if (reviewing() || atBottom) cur.scrollIntoView({ block: "nearest" });
  }
}

function renderControls() {
  const S = app.S;
  const total = S.ply;
  const at = reviewing() ? app.view : total;
  $("first").disabled = at === 0;
  $("prev").disabled = at === 0;
  $("next").disabled = at >= total;
  $("last").disabled = at >= total;
  $("live").disabled = !reviewing();
  const canPause = !S.result;
  $("pause").disabled = !canPause;
  $("pause").textContent = S.paused ? "Resume" : "Pause";
  $("resign").disabled = !!S.result || reviewing() || !S.humans.some((c) => !S.e.includes(c));
  $("bots").innerHTML = S.engine && Object.keys(S.engine).length
    ? Object.entries(S.engine).map(([c, e]) =>
        `<div><b style="color:${COLORS[c]}">${NAMES[c]}</b> · ${esc(e.bot)}: ${e.sims} sims in ${e.seconds}s, ` +
        e.values.map((v, i) => `<span style="color:${COLORS[ORDER[i]]}">${Math.round(v * 100)}%</span>`).join(" ") +
        `</div>`).join("")
    : "No bot has moved yet.";
}

function render() {
  if (!app.S) return;
  renderBoard();
  renderClocks();
  renderStatus();
  renderPlayers();
  renderMoves();
  renderControls();
}

// The screen follows the server: a game is running or it is not.
function syncView() {
  const inGame = !!app.S;
  $("setup").hidden = inGame;
  $("game").hidden = !inGame;
  if (!inGame) buildSetup();
}

// Fetch the game at the position being viewed (live, or an earlier ply when reviewing).
async function poll() {
  try {
    const q = app.view === null ? "" : `?ply=${app.view}`;
    const s = await get("/api/state" + q);
    if (s.idle) {
      app.S = null;
      app.view = null;
    } else {
      const prev = app.S;
      if (!prev || s.ply !== prev.ply || s.running !== prev.running || s.paused !== prev.paused ||
          !!s.result !== !!prev.result) {
        app.sel = null;
      }
      // a review stays on its position; back in the live game the view follows the last move
      if (app.view !== null && app.view >= s.ply) app.view = null;
      app.S = s;
      app.got = performance.now();
    }
    syncView();
  } catch (e) {
    $("status").textContent = "Lost connection to the game server.";
  }
}

// Move the review to ply p (null = live).
function goTo(p) {
  if (!app.S) return;
  app.view = p === null || p >= app.S.ply ? null : Math.max(0, p);
  app.sel = null;
  poll().then(render);
}

function stepBy(delta) {
  if (!app.S) return;
  const at = reviewing() ? app.view : app.S.ply;
  goTo(at + delta);
}

async function clickSquare(d) {
  const S = app.S;
  if (!S || app.busy || reviewing() || S.paused || !S.humans.includes(S.t)) return;
  const i = app.assets.views[viewSeat()][d];
  if (app.sel !== null && S.legal.some((m) => m[0] === app.sel && m[1] === i)) {
    app.busy = true;
    const r = await post("/api/move", { color: S.t, from: app.sel, to: i });
    app.busy = false;
    app.sel = null;
    if (r.error) $("status").textContent = r.error;
    await poll();
    render();
    return;
  }
  app.sel = S.legal.some((m) => m[0] === i) ? i : null;
  render();
}

// --- buttons and keys -------------------------------------------------------------------

$("start").onclick = startGame;
$("time-preset").onchange = () => {
  const custom = $("time-preset").value === CUSTOM;
  $("time-custom-row").hidden = !custom;
  if (custom) $("time-custom").focus();
};

$("pause").onclick = async () => {
  if (!app.S || app.S.result) return;
  const r = await post("/api/pause", { paused: !app.S.paused });
  if (r.error) $("status").textContent = r.error;
  await poll();
  render();
};

$("resign").onclick = async () => {
  const S = app.S;
  if (!S || S.result || reviewing()) return;
  const alive = S.humans.filter((c) => !S.e.includes(c));
  const c = alive.includes(S.t) ? S.t : alive[0];
  if (!c || !confirm(`Resign for ${NAMES[c]}?`)) return;
  const r = await post("/api/resign", { color: c });
  if (r.error) $("status").textContent = r.error;
  await poll();
  render();
};

$("new").onclick = async () => {
  if (app.S && !app.S.result && !confirm("Abandon this game and set up a new one?")) return;
  if (app.S) await post("/api/close", {});
  app.S = null;
  app.view = null;
  app.sel = null;
  syncView();
};

$("flip").onchange = () => render();
$("first").onclick = () => goTo(0);
$("prev").onclick = () => stepBy(-1);
$("next").onclick = () => stepBy(1);
$("last").onclick = () => goTo(null);
$("live").onclick = () => goTo(null);

$("moves").addEventListener("click", (e) => {
  const row = e.target.closest(".mrow");
  if (row && app.S) goTo(Number(row.dataset.ply));  // a move's number = the position after it
});

document.addEventListener("keydown", (e) => {
  if (!app.S || e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (e.key === "ArrowLeft") { e.preventDefault(); stepBy(-1); }
  else if (e.key === "ArrowRight") { e.preventDefault(); stepBy(1); }
  else if (e.key === "Home") { e.preventDefault(); goTo(0); }
  else if (e.key === "End") { e.preventDefault(); goTo(null); }
  else if (e.key === " " && !app.S.result) { e.preventDefault(); $("pause").click(); }
});

// --- start ------------------------------------------------------------------------------

async function init() {
  app.assets = await get("/api/assets");
  app.bots = await get("/api/bots");
  for (const k of Object.keys(app.assets.sprites)) {
    app.spriteURL[k] = "data:image/svg+xml;base64," + btoa(app.assets.sprites[k]);
  }
  for (let d = 0; d < 64; d++) {
    const cell = document.createElement("div");
    cell.className = "sq " + ((((d >> 3) + (d & 7)) % 2) ? "d" : "l");
    cell.addEventListener("click", () => clickSquare(d));
    $("board").appendChild(cell);
    app.squares.push(cell);
  }
  await poll();
  setInterval(poll, 250);
  setInterval(render, 100);
}

init().catch((e) => {
  document.body.textContent = "Could not reach the game server: " + e;
});

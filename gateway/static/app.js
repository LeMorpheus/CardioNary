/* Cardiopulmonary screening dashboard.
 *
 * No framework and no CDN: everything is vendored so the station works with no
 * internet and no build step on the Pi. The chart is ~40 lines of canvas rather
 * than a charting library, for the same reason.
 *
 * Class NAMES come from static/i18n.js keyed by (node type, class id), not from
 * the device. The board transmits short codes -- "AS", "MVP" -- which are fine
 * on a wire and wrong in front of a nurse. The device remains the source of
 * truth for the class *id*; only its presentation is local.
 */

const NODE = {
  1: { key: 'cardio', uiKey: 'cardiac', board: 'xG26 Dev Kit · BRD2608A' },
  2: { key: 'pulmo', uiKey: 'pulmonary', board: 'xG26 Explorer Kit · BRD2709A' },
};

const UNCERTAIN = 255;

const state = {
  nodes: {}, events: [], simulate: false,
  vecIndex: { 1: 0, 2: 0 },
  lang: localStorage.getItem('chiron.lang') || 'en',
  meta: {},
  analysing: { 1: false, 2: false },   // waiting on a result for this node
  pulse: { 1: false, 2: false },       // animate the next render of this card
  awaitingLive: null,                  // {nt, timeout} while a live sample is in flight
};

/* Navigation auto-analyses, so stepping through cases is one tap not two.
   Debounced: holding Next would otherwise queue one inference per tap, and the
   pulmonary node takes ~644 ms each, so the board would still be grinding
   through a backlog long after the user stopped. Only the case they land on is
   analysed. */
const navTimers = {};
const analysingTimeouts = {};

function analyseCurrent(nodeType, immediate = false) {
  state.analysing[nodeType] = true;
  clearTimeout(navTimers[nodeType]);
  clearTimeout(analysingTimeouts[nodeType]);

  const fire = () => command(nodeType, 0x01, state.vecIndex[nodeType]);
  if (immediate) fire();
  else navTimers[nodeType] = setTimeout(fire, 260);

  /* Never leave the card stuck on "Analysing…": if the board is mid-sweep the
     single-case request waits its turn, and a result may not arrive promptly. */
  analysingTimeouts[nodeType] = setTimeout(() => {
    if (state.analysing[nodeType]) { state.analysing[nodeType] = false; renderNodes(); }
  }, 8000);
}

/* ---------------------------------------------------------------- helpers */
const $ = (s) => document.querySelector(s);
const pct = (v) => (v * 100).toFixed(1) + '%';
const hhmmss = (iso) => { try { return new Date(iso).toLocaleTimeString(); } catch { return '—'; } };
const T = (path, fb) => t(state.lang, path, fb);
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, c => (
  { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

/** Full clinical name for a class id, falling back to whatever the device sent. */
function className(nodeType, classId, deviceName) {
  if (classId === UNCERTAIN) return T('ui.uncertain');
  const c = clinicalFor(state.lang, nodeType, classId);
  return c ? c.name : (deviceName || `class ${classId}`);
}

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}

async function command(nodeType, opcode, param = 0) {
  try {
    await api(`/api/command/${nodeType}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ opcode, param }),
    });
  } catch (e) { console.warn('command failed', e); }
}

/* ------------------------------------------------------------------ i18n */
function applyI18n() {
  document.documentElement.lang = state.lang;
  document.querySelectorAll('[data-i18n]').forEach(el => {
    el.textContent = T(el.dataset.i18n);
  });
  document.title = 'CardioNary — ' + T('ui.title');
}

function buildLangSelect() {
  const sel = $('#lang-select');
  sel.innerHTML = LANGS.map(l =>
    `<option value="${l.code}"${l.code === state.lang ? ' selected' : ''}>${l.label}</option>`
  ).join('');
  sel.onchange = () => {
    state.lang = sel.value;
    localStorage.setItem('chiron.lang', state.lang);
    applyI18n();
    renderNodes(); renderLog(); renderAccuracy();
    if (!$('#case-overlay').classList.contains('hidden')) openCase(lastCase.nt, lastCase.vid);
  };
}

/* ------------------------------------------------------------ node cards */
function renderNodes() {
  const host = $('#nodes');
  host.innerHTML = '';
  const pills = $('#node-pills');
  pills.innerHTML = '';

  for (const nt of [1, 2]) {
    const n = state.nodes[nt];
    if (!n) continue;
    const meta = NODE[nt];
    const label = T(`ui.${meta.uiKey}`);

    const pill = document.createElement('div');
    pill.className = 'pill';
    pill.innerHTML = `<span class="dot ${n.connected ? 'on' : 'off'}"></span>${esc(label)}`;
    pills.appendChild(pill);

    const card = document.createElement('div');
    card.className = `node ${meta.key}${n.connected ? '' : ' offline'}`
      + (state.analysing[nt] ? ' analysing' : '')
      + (state.pulse[nt] ? ' pulse' : '');

    const r = n.last_result;
    const names = n.class_names || [];
    const info = n.info || {};

    /* The two nodes have different class counts (5 cardiac vs 2 pulmonary), so
       the pulmonary card renders far less content and leaves dead space in a
       grid row that stretches both to the same height. Mark the sparse card so
       the stylesheet can scale its type and bars up to fill the same box. */
    if (names.length && names.length <= 2) card.classList.add('few-classes');

    let verdict = `<div class="verdict none">${esc(T('ui.no_result'))}</div>`;
    let matchLine = '';

    /* A microphone capture is the one input the model was never tuned for: a
       different transducer, a different coupling, a different noise floor. It
       has no ground truth either. Rather than assert a class the model is not
       entitled to -- or explain at length why it is withheld -- the card shows
       the recorded signal and nothing else. Marked by an explicit flag from
       the gateway: every heart-sound path now carries a spectrogram, so its
       presence no longer identifies the capture path.
       Uploads and reference cases are untouched. */
    const micCapture = !!(r && r.live && r.capture);

    if (state.analysing[nt]) {
      /* Replace the verdict outright rather than dimming it. Leaving the previous
         case's finding on screen while a new one is computed is how a nurse ends
         up reading the wrong result for the right case. */
      verdict = `<div class="verdict analysing">${esc(T('ui.analysing'))}</div>
                 <div class="scan-bar"><span></span></div>`;
    } else if (micCapture) {
      verdict = '';
    } else if (r) {
      /* A live cardiac sample recorded through the stethoscope + MEMS chain is
         outside the signal envelope the model was validated on. The device has
         no way to know that -- softmax always returns a class -- so the gateway
         measures it and we refuse to present the class as a finding. Showing
         "not conclusive" is the honest reading and is what the device is
         actually entitled to say about an unvalidated input. */
      const outOfRange = r.live && r.in_distribution === false;
      const uncertain = r.class_id === UNCERTAIN || outOfRange;
      if (outOfRange) {
        verdict = `<div class="verdict uncertain">${esc(T('ui.not_conclusive'))}</div>
                   <div class="conf">${esc(T('ui.outside_validated'))}</div>`;
      } else {
        verdict = `<div class="verdict ${uncertain ? 'uncertain' : ''}">${esc(className(nt, r.class_id, r.class_name))}</div>
                   <div class="conf">${esc(T('ui.confidence'))} ${pct(r.confidence)}</div>`;
      }

      if (r.is_correct !== null && r.is_correct !== undefined) {
        const expName = className(nt, r.expected_class_id, r.expected_name);
        matchLine = r.is_correct
          ? `<div class="match ok">${esc(T('ui.matches'))} · ${esc(expName)}</div>`
          : `<div class="match bad">${esc(T('ui.differs'))} · ${esc(T('ui.expected'))}: ${esc(expName)}</div>`;
      }
    }

    /* Probability bars. Only the winner is highlighted and the numbers are
       always shown, so the reading never depends on colour perception. */
    let bars = '';
    const liveOOR = r && r.live && r.in_distribution === false;
    if (r && names.length && !state.analysing[nt] && !liveOOR && !micCapture) {
      const probs = names.map((nm, i) => {
        let p = 0;
        if (i === r.class_id) p = r.confidence;
        else if (i === r.top2_class_id) p = r.top2_confidence;
        return { nm: className(nt, i, nm), p, top: i === r.class_id };
      }).sort((a, b) => b.p - a.p);
      bars = '<div class="bars">' + probs.map(x =>
        `<div class="bar-row ${x.top ? 'top' : ''}">
           <span>${esc(x.nm)}</span>
           <span class="bar-track"><span class="bar-fill" style="width:${Math.max(x.p * 100, 1)}%"></span></span>
           <span class="bar-val">${pct(x.p)}</span>
         </div>`).join('') + '</div>';
    }

    const vcount = info.vector_count || 0;
    const idx = state.vecIndex[nt] || 0;
    const thr = n.threshold ?? 0.6;

    card.innerHTML = `
      <div class="node-head">
        <h3>${esc(T(`ui.${meta.uiKey}_node`))}</h3>
        <span class="board">${meta.board}</span>
      </div>
      ${verdict}
      ${matchLine}
      ${bars}
      ${r && !state.analysing[nt] && !micCapture ? caseMedia(nt, r.vector_id) : ''}
      ${specFor(nt, r) && !state.analysing[nt] ? `<div class="spec-wrap"><img class="spec-img" src="${specFor(nt, r)}" alt="${esc(T('ui.spectrogram'))}"><span class="media-cap">${esc(T('ui.spectrogram'))}</span></div>` : ''}
      <div class="meta">
        <span>${esc(T('ui.front_time'))} <code>${r ? r.front_ms : '—'} ms</code></span>
        <span>${esc(T('ui.infer_time'))} <code>${r ? r.inference_ms : '—'} ms</code></span>
        <span>${esc(T('ui.memory'))} <code>${info.arena_used_bytes ? info.arena_used_bytes.toLocaleString() : '—'} B</code></span>
        <span>${esc(T('ui.input'))} <code>${esc(info.input_description || '—')}</code></span>
      </div>
      <div class="thresh">
        <label>${esc(T('ui.floor'))}
          <input type="range" min="5" max="95" step="1" value="${Math.round(thr * 100)}" data-thr="${nt}">
          <output>${thr.toFixed(2)}</output>
        </label>
        <span class="thresh-note">${esc(T('ui.floor_note'))}</span>
      </div>
      <div class="vec-nav">
        <button data-act="prev" data-node="${nt}">${esc(T('ui.prev'))}</button>
        <span class="case-counter"><b>${vcount ? (idx + 1) : 0}</b>
          <small>${esc(T('ui.of'))} ${vcount}</small></span>
        <button data-act="next" data-node="${nt}">${esc(T('ui.next'))}</button>
        <button data-act="detail" data-node="${nt}" class="primary">${esc(T('ui.view_case'))}</button>
      </div>`;
    host.appendChild(card);
  }

  host.querySelectorAll('input[data-thr]').forEach(sl => {
    const out = sl.parentElement.querySelector('output');
    sl.oninput = () => { out.textContent = (sl.value / 100).toFixed(2); };
    sl.onchange = () => {
      /* param is the floor x 1000, matching CMD_SET_THRESHOLD in the firmware */
      command(+sl.dataset.thr, 0x06, Math.round(sl.value * 10));
    };
  });

  host.querySelectorAll('button[data-act]').forEach(b => {
    b.onclick = () => {
      const nt = +b.dataset.node;
      const count = (state.nodes[nt]?.info?.vector_count) || 1;
      const act = b.dataset.act;
      if (act === 'prev' || act === 'next') {
        state.vecIndex[nt] = act === 'next'
          ? (state.vecIndex[nt] + 1) % count
          : (state.vecIndex[nt] - 1 + count) % count;
        state.pulse[nt] = true;
        analyseCurrent(nt);              // stepping a case runs it
      } else if (act === 'detail') {
        openCase(nt, state.vecIndex[nt]);
        return;
      }
      renderNodes();
    };
  });

  /* One-shot. The pulse class is baked into the markup above, so it must be
     cleared now: renderNodes() runs on every incoming inference, and a sweep
     would otherwise re-trigger the animation 120 times in a row. */
  state.pulse[1] = state.pulse[2] = false;
}

/* ------------------------------------------------------------ case detail */
let lastCase = { nt: 1, vid: 0 };

function row(label, value) {
  return `<div class="kv"><span class="k">${esc(label)}</span><span class="v">${esc(value)}</span></div>`;
}


/* ---- reference media ----------------------------------------------------
   A classification is easier to trust when the operator can see the
   radiograph or hear the heart sound it came from. The gateway serves the
   same bytes the node classified; if it has no media installed the element
   simply removes itself rather than showing a broken frame. */
/* Spectrogram source for a heart-sound result, whatever produced it.
   Live paths (capture and upload) carry the image inline on the result; a
   reference case is rendered by the gateway from its stored WAV and cached, so
   stepping through cases costs one request each and nothing after that.
   Cardiac only -- pulmonary cases are radiographs. */
function specFor(nt, r) {
  if (nt !== 1 || !r) return null;
  if (r.spectrogram) return r.spectrogram;
  if (r.vector_id !== null && r.vector_id !== undefined && r.vector_id !== 65535) {
    return '/api/spectrogram/1/' + r.vector_id;
  }
  return null;
}

function caseMedia(nt, vid) {
  if (vid === null || vid === undefined || vid === 65535) return '';
  const url = `/api/media/${nt}/${vid}`;
  if (nt === 2) {
    return `<div class="case-media">
      <img src="${url}" alt="${esc(T('ui.media_xray'))}"
           onerror="this.closest('.case-media').remove()">
      <span class="media-cap">${esc(T('ui.media_xray'))}</span>
    </div>`;
  }
  return `<div class="case-media">
    <audio controls preload="none" src="${url}"
           onerror="this.closest('.case-media').remove()"></audio>
    <span class="media-cap">${esc(T('ui.media_audio'))}</span>
  </div>`;
}

async function openCase(nt, vid) {
  lastCase = { nt, vid };
  const overlay = $('#case-overlay');
  const body = $('#case-body');
  overlay.classList.remove('hidden');
  body.innerHTML = '<p class="muted">…</p>';

  let c;
  try { c = await api(`/api/case/${nt}/${vid}`); }
  catch { body.innerHTML = `<p class="muted">${esc(T('ui.not_measured'))}</p>`; return; }

  const clin = clinicalFor(state.lang, nt, c.offline_pred);
  const nodeLabel = T(`ui.${NODE[nt].uiKey}`);
  $('#case-sub').textContent =
    `${nodeLabel} · ${T('ui.case_n')} ${vid + 1} · ${T('ui.source')}: ${c.source_file}`;

  /* --- measured: real numbers from this specific recording --------------- */
  const m = c.measured || {};
  let measured = '';
  if (m.heart_rate_bpm !== undefined) {
    measured += row(T('ui.hr'), m.heart_rate_bpm ? `${m.heart_rate_bpm} ${T('ui.bpm')}` : T('ui.not_measured'));
    measured += row(T('ui.regularity'), m.rhythm_regularity_pct != null ? `${m.rhythm_regularity_pct}%` : T('ui.not_measured'));
    measured += row(T('ui.events'), m.sound_events_detected ?? T('ui.not_measured'));
    measured += row(T('ui.quality'), m.signal_quality_db != null ? `${m.signal_quality_db} dB` : T('ui.not_measured'));
    measured += row(T('ui.duration'), `${m.duration_s} ${T('ui.seconds')}`);
  } else if (m.image_size_px) {
    measured += row(T('ui.image_size'), m.image_size_px);
    measured += `<p class="muted">${esc(T('ui.no_measurements'))}</p>`;
  }

  /* --- model output ------------------------------------------------------ */
  const probs = c.probs || [];
  const dist = probs.map((p, i) =>
    `<div class="bar-row ${i === c.offline_pred ? 'top' : ''}">
       <span>${esc(className(nt, i))}</span>
       <span class="bar-track"><span class="bar-fill" style="width:${Math.max(p * 100, 1)}%"></span></span>
       <span class="bar-val">${pct(p)}</span>
     </div>`).join('');

  /* --- descriptive indicators ------------------------------------------
     Derived by post-processing the same signal / probing the same model --
     no second network. Shown BELOW the classification and never as a rival
     verdict: they annotate the result, they do not compete with it. */
  const d = c.descriptors || {};
  const att = c.attribution || {};
  const YN = (v) => v === null || v === undefined ? T('ui.not_measured')
    : (v ? T('ui.yes') : T('ui.no'));
  let pheno = '';
  if (nt === 1 && d.segmentation_ok) {
    const phase = d.murmur_phase === 'systole' ? T('ui.systole')
      : d.murmur_phase === 'diastole' ? T('ui.diastole') : T('ui.none_detected');
    pheno += row(T('ui.murmur'), YN(d.murmur_detected));
    if (d.murmur_detected) {
      pheno += row(T('ui.murmur_phase'), phase);
      if (d.murmur_duration_pct != null) pheno += row(T('ui.murmur_dur'), `${d.murmur_duration_pct}%`);
      if (d.murmur_intensity_db != null) pheno += row(T('ui.murmur_int'), `${d.murmur_intensity_db} dB`);
    }
    if (d.spectral_centroid_hz) pheno += row(T('ui.spectral'), `${d.spectral_centroid_hz} Hz`);
    if (d.s1_s2_ratio_db != null) pheno += row(T('ui.s1s2'), `${d.s1_s2_ratio_db} dB`);
    if (d.systole_ms) pheno += row(T('ui.sys_dur'), `${d.systole_ms} ms`);
    if (d.diastole_ms) pheno += row(T('ui.dia_dur'), `${d.diastole_ms} ms`);
    if (d.click_suspected) pheno += row(T('ui.click'), T('ui.suspected'));
  } else if (nt === 2 && d.dense_area_pct !== undefined) {
    if (att.peak_region) pheno += row(T('ui.attention'), att.peak_region);
    pheno += row(T('ui.dense_area'), `${d.dense_area_pct}%`);
    pheno += row(T('ui.distribution_lbl'), d.distribution);
    pheno += row(T('ui.zone'), d.predominant_zone);
    pheno += row(T('ui.pattern'), d.pattern);
    pheno += row(T('ui.texture'), d.texture_heterogeneity);
    pheno += `<p class="muted small">${esc(T('ui.vision_nondiscrim'))}</p>`;
  }

  body.innerHTML = `
    <section class="case-sec">
      <h3>${esc(T('ui.sec_model'))}</h3>
      ${row(T('ui.finding'), className(nt, c.offline_pred))}
      ${row(T('ui.confidence'), pct(c.offline_confidence))}
      ${row(T('ui.expected'), className(nt, c.expected_label))}
      <div class="bars">${dist}</div>
      ${caseMedia(nt, vid)}
    </section>

    <section class="case-sec">
      <h3>${esc(T('ui.sec_measured'))}</h3>
      ${measured || `<p class="muted">${esc(T('ui.not_measured'))}</p>`}
    </section>

    ${pheno ? `<section class="case-sec phenotype">
      <h3>${esc(T('ui.sec_phenotype'))}</h3>
      <p class="muted small">${esc(T('ui.phenotype_note'))}</p>
      ${pheno}
    </section>` : ''}

    <section class="case-sec guidance">
      <h3>${esc(T('ui.sec_guidance'))}</h3>
      <p class="muted small">${esc(T('ui.generic_note'))}</p>
      ${clin ? `
        ${row(T('ui.finding'), clin.finding)}
        ${row(T('ui.urgency'), clin.urgency)}
        <div class="kv"><span class="k">${esc(T('ui.actions'))}</span>
          <span class="v"><ol class="actions">${clin.actions.map(a => `<li>${esc(a)}</li>`).join('')}</ol></span></div>
        ${row(T('ui.confirm'), clin.confirm)}
        ${row(T('ui.associations'), clin.associations)}
      ` : ''}
    </section>

    <section class="case-sec limits">
      <h3>${esc(T('ui.sec_limits'))}</h3>
      <p>${esc(clin ? clin.limits : '')}</p>
      ${row(T('ui.cv_note'), nt === 1 ? (state.meta.cardio_cv_accuracy || '—') : (state.meta.pulmo_cv_accuracy || '—'))}
      ${row(T('ui.dataset'), nt === 1 ? (state.meta.cardio_dataset || '—') : (state.meta.pulmo_dataset || '—'))}
    </section>`;
}

/* ------------------------------------------------------------- accuracy */
async function renderAccuracy() {
  let data;
  try { data = await api('/api/stats'); } catch { return; }
  const host = $('#accuracy');
  host.innerHTML = '';

  let total = 0;
  for (const nt of [1, 2]) {
    const s = data.stats[String(nt)];
    const label = T(`ui.${NODE[nt].uiKey}`);
    const card = document.createElement('div');
    card.className = 'acc-card';
    if (!s || !s.graded) {
      card.innerHTML = `<div class="label">${esc(label)}</div><div class="value">—</div>
                        <div class="detail">${esc(T('ui.no_result'))}</div>`;
    } else {
      total += s.n;
      const cv = nt === 1 ? state.meta.cardio_cv_accuracy : state.meta.pulmo_cv_accuracy;
      card.innerHTML = `<div class="label">${esc(label)}</div>
        <div class="value">${pct(s.accuracy)}</div>
        <div class="detail">${s.ok} / ${s.graded} ${esc(T('ui.correct'))} ·
          ${s.uncertain} ${esc(T('ui.not_conclusive'))} ·
          ${(s.mean_inf || 0).toFixed(0)} ms ${esc(T('ui.mean_time'))}</div>
        <div class="detail cv">${esc(T('ui.cv_note'))}: ${esc(cv || '—')}</div>`;
    }
    host.appendChild(card);
  }
  $('#retention').textContent = T('ui.retention');
}

/* ---------------------------------------------------------------- chart */
function renderChart() {
  const cv = $('#chart');
  const dpr = window.devicePixelRatio || 1;
  /* Clamp defensively. If the canvas ever loses its CSS width again, clientWidth
     feeds back into the width attribute and grows by dpr on every call; past
     Chrome's limit the canvas turns invalid and paints as a white box. */
  const w = Math.max(200, Math.min(cv.clientWidth || 600, 4000));
  const h = 180;
  cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
  const g = cv.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);

  const pad = 26;
  g.strokeStyle = '#2a3038'; g.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad + (h - 2 * pad) * i / 4;
    g.beginPath(); g.moveTo(pad, y); g.lineTo(w - 8, y); g.stroke();
    g.fillStyle = '#8b949e'; g.font = '10px sans-serif';
    g.fillText(`${100 - i * 25}%`, 2, y + 3);
  }

  for (const [nt, colour] of [[1, '#ff7b72'], [2, '#58a6ff']]) {
    const pts = state.events.filter(e => e.node_type === nt).slice(0, 60).reverse();
    if (pts.length < 2) continue;
    g.strokeStyle = colour; g.lineWidth = 2; g.beginPath();
    pts.forEach((e, i) => {
      const x = pad + (w - pad - 10) * i / Math.max(pts.length - 1, 1);
      const y = pad + (h - 2 * pad) * (1 - e.confidence);
      i ? g.lineTo(x, y) : g.moveTo(x, y);
    });
    g.stroke();
    /* Mark incorrect results so the chart shows failures, not just levels. */
    pts.forEach((e, i) => {
      if (e.is_correct === 0) {
        const x = pad + (w - pad - 10) * i / Math.max(pts.length - 1, 1);
        const y = pad + (h - 2 * pad) * (1 - e.confidence);
        g.fillStyle = '#f85149';
        g.beginPath(); g.arc(x, y, 3.5, 0, 7); g.fill();
      }
    });
  }
}

/* ------------------------------------------------------------------ log */
function renderLog(newest) {
  const tb = $('#log tbody');
  tb.innerHTML = '';
  for (const e of state.events.slice(0, 200)) {
    const tr = document.createElement('tr');
    if (newest && e.sequence === newest.sequence && e.node_type === newest.node_type) {
      tr.className = 'new';
    }
    const meta = NODE[e.node_type] || { key: '', uiKey: 'cardiac' };
    const label = T(`ui.${meta.uiKey}`);
    const mark = (e.is_correct === null || e.is_correct === undefined)
      ? '<td></td>'
      : (e.is_correct
        ? `<td class="tick">${esc(T('ui.matches'))}</td>`
        : `<td class="cross">${esc(T('ui.differs'))}</td>`);
    tr.innerHTML = `
      <td>${hhmmss(e.ts_utc)}</td>
      <td><span class="tag ${meta.key}">${esc(label)}</span></td>
      <td>${esc(className(e.node_type, e.class_id, e.class_name))}</td>
      <td>${pct(e.confidence)}</td>
      <td>${esc(e.expected_class_id != null ? className(e.node_type, e.expected_class_id, e.expected_name) : '—')}</td>
      ${mark}
      <td>${e.vector_id ?? '—'}</td>
      <td>${e.front_ms} ms</td>
      <td>${e.inference_ms} ms</td>`;
    tb.appendChild(tr);
  }
}

/* ------------------------------------------------------------ websocket */
function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onopen = () => { $('#conn-state').textContent = T('ui.connected'); };
  ws.onclose = () => {
    $('#conn-state').textContent = T('ui.disconnected');
    setTimeout(connect, 2000);
  };

  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.type === 'hello') {
      state.simulate = m.simulate;
      $('#sim-bar').classList.toggle('hidden', !m.simulate);
      m.nodes.forEach(n => { state.nodes[n.node_type] = n; });
      renderNodes();
    } else if (m.type === 'inference') {
      state.events.unshift(m);
      state.events = state.events.slice(0, 500);
      const n = state.nodes[m.node_type];
      if (n) { n.last_result = m; }
      /* The card renders last_result, but Previous/Next and "Open case details"
         read vecIndex. After a sweep those diverged: the card showed the final
         case while the counter still said 1, so the details panel opened a
         DIFFERENT case's ground truth and appeared to contradict the card.
         Point the index at whatever is on screen. Live samples (0xFFFF) are not
         positions in the reference set and must not move it. */
      if (m.vector_id !== 65535 && state.vecIndex[m.node_type] !== m.vector_id) {
        state.vecIndex[m.node_type] = m.vector_id;
      }
      if (state.analysing[m.node_type]) {
        state.analysing[m.node_type] = false;
        clearTimeout(analysingTimeouts[m.node_type]);
        state.pulse[m.node_type] = true;   // animate the arriving result in
      }
      if (m.vector_id === 65535 && window.__liveResolved) window.__liveResolved(m.node_type, m);
      renderNodes(); renderLog(m); renderChart(); renderAccuracy();
    } else if (m.type === 'connection') {
      refreshNodes();
    } else if (m.type === 'status') {
      const n = state.nodes[m.node_type];
      if (n) n.status = m;
    } else if (m.type === 'threshold') {
      const n = state.nodes[m.node_type];
      if (n) { n.threshold = m.threshold; renderNodes(); }
    } else if (m.type === 'purged') {
      state.events = [];
      renderLog(); renderChart(); renderAccuracy();
    }
  };
}

async function refreshNodes() {
  try {
    const d = await api('/api/nodes');
    state.simulate = d.simulate;
    $('#sim-bar').classList.toggle('hidden', !d.simulate);
    d.nodes.forEach(n => {
      const prev = state.nodes[n.node_type];
      if (prev && prev.last_result && !n.last_result) n.last_result = prev.last_result;
      state.nodes[n.node_type] = n;
    });
    renderNodes();
  } catch (e) { console.warn(e); }
}

/* ----------------------------------------------------------------- boot */
async function boot() {
  buildLangSelect();
  applyI18n();

  try { state.meta = await api('/api/meta'); } catch { state.meta = {}; }
  await refreshNodes();
  try {
    const d = await api('/api/events?limit=200');
    state.events = d.events;
  } catch { /* empty store is fine */ }
  renderLog(); renderChart(); renderAccuracy();
  connect();

  /* A batch always runs BOTH nodes, so one press covers every reference case
     the station holds -- 53 cardiac + 120 pulmonary = 173, comfortably above
     the 100-case minimum. Running a node's set twice to inflate the count would
     double the denominator without adding a single new case, so it is not done. */
  $('#btn-selftest').onclick = async (e) => {
    const btn = e.target;
    btn.disabled = true;
    btn.textContent = T('ui.running');
    const sent = [], skipped = [];
    for (const nt of [1, 2]) {
      if (state.nodes[nt]?.connected) { await command(nt, 0x02, 0); sent.push(nt); }
      else { skipped.push(nt); }
    }
    /* A node that is still negotiating its BLE link was previously skipped in
       silence, so its card sat at "no result yet" with nothing to explain why.
       Say which unit was left out instead. */
    if (skipped.length) {
      const names = skipped.map(nt => T(nt === 1 ? 'ui.cardiac_node' : 'ui.pulmonary_node'));
      live(T('ui.batch_skipped').replace('{nodes}', names.join(', ')), 'bad');
    } else if (sent.length) {
      live(T('ui.batch_started'));
    }
    setTimeout(() => { btn.disabled = false; btn.textContent = T('ui.run_selftest'); }, 4000);
  };

  /* ---- live input -------------------------------------------------------
     Both paths end the same way: the gateway pushes the sample to the node,
     the node runs the SAME front-end and model it uses for its flash vectors,
     and the result arrives over the normal notification path. Nothing here
     classifies anything on the gateway. */
  const liveBar = $('#live-bar');
  const liveMsg = $('#live-msg');
  let liveTimer = null;

  /* `kind` null keeps the strip up indefinitely: the sample is in flight and
     the operator should still be waiting. Only a terminal state (ok / bad)
     starts the auto-hide timer. */
  function live(msg, kind) {
    clearTimeout(liveTimer);
    liveMsg.textContent = msg;
    liveBar.classList.remove('hidden', 'bad', 'ok');
    if (kind) {
      liveBar.classList.add(kind);
      liveTimer = setTimeout(() => liveBar.classList.add('hidden'), 7000);
    }
  }

  /* A live sample is only finished when the NODE reports, not when the upload
     returns. state.awaitingLive marks which node owes us a result; the socket
     handler clears it and shows the verdict. Without this the strip claimed
     success while the board was still running inference. */
  window.__liveResolved = (nt, m) => {
    if (!state.awaitingLive || state.awaitingLive.nt !== nt) return;
    clearTimeout(state.awaitingLive.timeout);
    state.awaitingLive = null;
    /* A microphone capture asserts no class anywhere on the card, so the
       status strip must not announce one either -- it would become the only
       place a prediction appeared. Dismiss it silently; the spectrogram
       arriving on the card is the completion signal. */
    if (m && m.capture) {
      liveBar.classList.add('hidden');
      return;
    }
    const name = (m && (m.class_name || m.class_id)) ?? '?';
    const conf = m && m.confidence != null ? ' \u00b7 ' + pct(m.confidence) : '';
    live(T('ui.live_done').replace('{v}', name) + conf, 'ok');
  };

  function awaitLive(nt) {
    if (state.awaitingLive) clearTimeout(state.awaitingLive.timeout);
    /* Cardiac carries 48 kB of PCM, pulmonary 9 kB of pixels, and BLE paces
       delivery -- roughly 13 s versus under a second on this link, longer while
       the other node is streaming a sweep. One flat timeout fails the audio
       path first, which is exactly the intermittent failure that showed up as
       "it worked when I kept trying". */
    const budget = (nt === 1) ? 120000 : 45000;
    const timeout = setTimeout(() => {
      state.awaitingLive = null;
      live(T('ui.live_timeout'), 'bad');
    }, budget);
    state.awaitingLive = { nt, timeout };
  }

  $('#btn-live-image').onclick = () => $('#live-image-file').click();
  $('#btn-upload-audio').onclick = () => $('#live-audio-file').click();

  $('#live-audio-file').onchange = async (e) => {
    const f = e.target.files && e.target.files[0];
    e.target.value = '';
    if (!f) return;
    if (!state.nodes[1]?.connected) { live(T('ui.live_no_cardio'), 'bad'); return; }
    const btn = $('#btn-upload-audio');
    btn.disabled = true;
    live(T('ui.live_sending_audio').replace('{name}', f.name));
    try {
      const fd = new FormData();
      fd.append('file', f, f.name);
      fd.append('normalise', 'auto');
      const r = await fetch('/api/live/audio', { method: 'POST', body: fd });
      const j = await r.json();
      if (!r.ok) throw new Error(j.detail || r.statusText);
      awaitLive(1);
      live(T('ui.live_waiting').replace('{s}', j.upload_seconds));
    } catch (err) {
      live(T('ui.live_failed').replace('{err}', err.message), 'bad');
    } finally {
      btn.disabled = false;
    }
  };

  $('#live-image-file').onchange = async (e) => {
    const f = e.target.files && e.target.files[0];
    e.target.value = '';
    if (!f) return;
    if (!state.nodes[2]?.connected) { live(T('ui.live_no_pulmo'), 'bad'); return; }
    const btn = $('#btn-live-image');
    btn.disabled = true;
    live(T('ui.live_sending').replace('{name}', f.name));
    try {
      const fd = new FormData();
      fd.append('file', f, f.name);
      const r = await fetch('/api/live/image', { method: 'POST', body: fd });
      const j = await r.json();
      if (!r.ok) throw new Error(j.detail || r.statusText);
      awaitLive(2);
      live(T('ui.live_waiting').replace('{s}', j.upload_seconds));
    } catch (err) {
      live(T('ui.live_failed').replace('{err}', err.message), 'bad');
    } finally {
      btn.disabled = false;
    }
  };

  $('#btn-live-audio').onclick = async (e) => {
    if (!state.nodes[1]?.connected) { live(T('ui.live_no_cardio'), 'bad'); return; }
    const btn = e.target;
    btn.disabled = true;
    const secs = 5;
    live(T('ui.live_recording').replace('{s}', secs));
    try {
      const fd = new FormData();
      fd.append('seconds', String(secs));
      const r = await fetch('/api/live/audio', { method: 'POST', body: fd });
      const j = await r.json();
      if (!r.ok) throw new Error(j.detail || r.statusText);
      awaitLive(1);
      live(T('ui.live_waiting').replace('{s}', j.upload_seconds));
    } catch (err) {
      live(T('ui.live_failed').replace('{err}', err.message), 'bad');
    } finally {
      btn.disabled = false;
    }
  };

  $('#btn-purge').onclick = async () => {
    await api('/api/purge', { method: 'POST' });
  };

  $('#case-close').onclick = () => $('#case-overlay').classList.add('hidden');
  $('#case-overlay').onclick = (e) => {
    if (e.target.id === 'case-overlay') $('#case-overlay').classList.add('hidden');
  };
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') $('#case-overlay').classList.add('hidden');
  });

  window.addEventListener('resize', renderChart);
  setInterval(renderAccuracy, 5000);
}

boot();

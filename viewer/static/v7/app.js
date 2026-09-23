(() => {
  const video = document.getElementById("video");
  const overlay = document.getElementById("overlay");
  const world = document.getElementById("world");
  const frame = document.getElementById("frame");
  const stage = document.getElementById("stage");
  const anno = document.getElementById("anno");
  const track = document.getElementById("track");
  const playhead = document.getElementById("playhead");
  const timeline = document.getElementById("timeline");
  const sampleSelect = document.getElementById("sample");
  const playButton = document.getElementById("play");
  const clock = document.getElementById("clock");
  const statusEl = document.getElementById("status");
  const worldCaption = document.getElementById("world-caption");
  const statsEl = document.getElementById("stats");
  const rulerEl = document.getElementById("ruler");
  const tipEl = document.getElementById("tip");
  const ratesEl = document.getElementById("rates");
  const veil = document.getElementById("stage-veil");
  const hudTrack = document.getElementById("hud-track");
  const hudFrame = document.getElementById("hud-frame");
  const egoNote = document.getElementById("ego-note");
  const playheadTime = playhead.querySelector(".playhead-time");
  const playGlyph = playButton.querySelector(".glyph");
  let veilTimer = 0;
  let paintedFrame = -1;
  let lastWorldDraw = 0;
  const octx = overlay.getContext("2d");
  const wctx = world.getContext("2d");

  const RATES = [1, 0.5, 0.25];

  const FINGERS = [
    [0, 1, 2, 3, 4],
    [0, 5, 6, 7, 8],
    [0, 9, 10, 11, 12],
    [0, 13, 14, 15, 16],
    [0, 17, 18, 19, 20],
  ];
  const FINGER_COLORS = ["#f97316", "#22c55e", "#3b82f6", "#e879f9", "#22d3ee"];
  const HAND_LABEL = [
    { name: "L", color: "#c4b5fd" },
    { name: "R", color: "#93c5fd" },
  ];
  const TRAIL_RGB = ["167,139,250", "245,158,11"];
  // Curated blue → teal → indigo → periwinkle ramp, per the official reference,
  // shifted a little brighter so it holds up on the dark timeline substrate.
  const BLOCKS = [
    "#4d84f5", "#1fb6a6", "#6366f1", "#8b9dfb",
    "#22a2dd", "#7c6cf0", "#2fc4a8", "#5b8cff",
  ];

  const state = {
    samples: [],
    sample: null,
    segments: [],
    errors: [],
    hands: null,
    duration: 0,
    shownKey: "",
    blockId: "",
  };
  let loadToken = 0;

  function decode(b64, Ctor) {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i += 1) bytes[i] = bin.charCodeAt(i);
    const buffer = new ArrayBuffer(bytes.byteLength);
    new Uint8Array(buffer).set(bytes);
    return new Ctor(buffer);
  }

  function num(value) {
    return Number(value);
  }

  function fmt(seconds) {
    if (!Number.isFinite(seconds)) return "00:00.0";
    const sign = seconds < 0 ? "-" : "";
    const abs = Math.abs(seconds);
    const m = Math.floor(abs / 60);
    const s = abs - m * 60;
    return `${sign}${String(m).padStart(2, "0")}:${s.toFixed(1).padStart(4, "0")}`;
  }

  function currentSegment(time) {
    return state.segments.find((segment, index) => {
      const start = num(segment.start_ts);
      const end = num(segment.end_ts);
      const last = index === state.segments.length - 1;
      return time >= start && (time < end || (last && time <= end + 0.05));
    });
  }

  function setStatus(text, state = "busy") {
    statusEl.textContent = text;
    statusEl.dataset.state = state;
  }

  function videoName(sample) {
    return sample.title || sample.id;
  }

  async function boot() {
    const listed = await fetch("/api/samples").then((res) => res.json());
    const samples = listed.filter((sample) => sample.origin === "examples");
    state.samples = samples;

    sampleSelect.innerHTML = samples.map((sample) => (
      `<option value="${sample.id}">${escapeHtml(videoName(sample))}</option>`
    )).join("");
    sampleSelect.addEventListener("change", () => loadSample(sampleSelect.value));

    ratesEl.innerHTML = RATES.map((rate) => (
      `<button type="button" data-rate="${rate}" aria-pressed="${rate === 1}">${rate}×</button>`
    )).join("");
    ratesEl.addEventListener("click", (event) => {
      const button = event.target.closest("button[data-rate]");
      if (!button) return;
      const rate = Number(button.dataset.rate);
      video.playbackRate = rate;
      ratesEl.querySelectorAll("button").forEach((other) => {
        other.setAttribute("aria-pressed", String(Number(other.dataset.rate) === rate));
      });
    });

    document.querySelectorAll(".btn-step").forEach((button) => {
      button.addEventListener("click", () => stepFrame(Number(button.dataset.step)));
    });

    playButton.addEventListener("click", toggle);
    video.addEventListener("loadedmetadata", () => {
      state.duration = video.duration || 0;
      fitFrame();
      renderTimeline();
      renderRuler();
      renderStats();
      paint(true);
    });
    const hideVeil = () => {
      clearTimeout(veilTimer);
      veil.classList.remove("on");
    };
    const bufferedAhead = () => {
      const time = video.currentTime;
      for (let i = 0; i < video.buffered.length; i += 1) {
        if (video.buffered.start(i) <= time && video.buffered.end(i) > time) {
          return video.buffered.end(i) - time;
        }
      }
      return 0;
    };
    video.addEventListener("play", () => setPlayGlyph(true));
    video.addEventListener("pause", () => setPlayGlyph(false));
    video.addEventListener("loadeddata", hideVeil);
    video.addEventListener("waiting", () => {
      clearTimeout(veilTimer);
      veilTimer = setTimeout(() => {
        if (video.readyState < 3 && bufferedAhead() < 0.4) veil.classList.add("on");
      }, 400);
    });
    video.addEventListener("playing", () => {
      setPlayGlyph(true);
      hideVeil();
    });
    video.addEventListener("canplay", hideVeil);

    timeline.addEventListener("pointerdown", scrub);
    track.addEventListener("pointermove", showTip);
    track.addEventListener("pointerleave", () => tipEl.classList.remove("on"));

    window.addEventListener("resize", () => {
      fitFrame();
      renderRuler();
    });
    window.addEventListener("keydown", (event) => {
      if (event.target.matches("select")) return;
      if (event.code === "Space") {
        event.preventDefault();
        toggle();
      } else if (event.code === "ArrowRight") {
        video.currentTime = Math.min(state.duration, video.currentTime + 1);
      } else if (event.code === "ArrowLeft") {
        video.currentTime = Math.max(0, video.currentTime - 1);
      }
    });
    if (!samples.length) {
      setStatus("No videos in examples", "error");
      return;
    }
    await loadSample(samples[0].id);
    requestAnimationFrame(loop);
  }

  // One frame at the sample's own rate when known, else a 30fps nudge.
  function stepFrame(direction) {
    const fps = state.hands && state.hands.fps ? state.hands.fps : 30;
    const next = video.currentTime + direction / fps;
    video.currentTime = Math.max(0, Math.min(state.duration || 0, next));
  }

  async function loadSample(id) {
    const token = ++loadToken;
    const sample = state.samples.find((item) => item.id === id);
    state.sample = sample;
    state.hands = null;
    state.worldStamp = "";
    state.shownKey = "";
    state.blockId = "";

    if (sampleSelect.value !== id) sampleSelect.value = id;
    veil.classList.add("on");

    setStatus("Reading annotation", "busy");
    video.pause();
    video.src = sample.video_url;
    const annotation = await fetch(sample.annotation_url).then((res) => res.json());
    if (token !== loadToken) return;
    state.segments = annotation.segments || [];
    state.errors = annotation.errors || [];
    if (sample.hands_ready) {
      setStatus("Reading hand keypoints", "busy");
      const hands = await fetch(sample.keypoints_url).then((res) => res.json());
      if (token !== loadToken) return;
      if (hands.ready) {
        state.hands = {
          n: hands.n,
          focal: hands.focal,
          width: hands.width,
          height: hands.height,
          fps: hands.fps,
          world: decode(hands.world, Float32Array),
          valid: decode(hands.valid, Uint8Array),
          R: decode(hands.R, Float32Array),
          t: decode(hands.t, Float32Array),
          camPos: decode(hands.camPos, Float32Array),
          camZ: decode(hands.camZ, Float32Array),
          camR: hands.camR ? decode(hands.camR, Float32Array) : null,
          mesh: null,
          center: [0, 0, 0],
        };
        if (sample.mesh_ready) {
          setStatus("Reading hand mesh", "busy");
          const buffer = await fetch(sample.mesh_url).then((res) => res.arrayBuffer());
          if (token !== loadToken) return;
          state.hands.mesh = parseMesh(buffer);
        }
        setStatus("Keypoints overlaid", "ready");
      } else {
        setStatus("Hands not exported", "pending");
      }
    } else if (!state.segments.length) {
      setStatus("Annotation pending", "pending");
    } else {
      setStatus("Annotation loaded", "ready");
    }
    renderAnnotation(video.currentTime || 0);
    renderTimeline();
    renderRuler();
    renderStats();
    fitFrame();
    paint(true);
    veil.classList.remove("on");
  }

  function renderStats() {
    const sample = state.sample;
    if (!sample || !statsEl) return;
    const rows = [];
    if (state.duration) {
      rows.push(["duration", `${state.duration.toFixed(2)}<small>s</small>`]);
    }
    rows.push(["segments", state.segments.length
      ? String(state.segments.length)
      : "<small>pending</small>"]);
    const actions = state.segments.reduce(
      (sum, segment) => sum + (actionLabel(segment).verb ? 1 : 0), 0);
    if (actions) rows.push(["actions", String(actions)]);
    const w = state.hands ? state.hands.width : video.videoWidth;
    const h = state.hands ? state.hands.height : video.videoHeight;
    if (w && h) rows.push(["resolution", `${w}<small>×</small>${h}`]);
    if (state.hands && state.hands.fps) {
      rows.push(["frame rate", `${state.hands.fps.toFixed(0)}<small>fps</small>`]);
    }
    rows.push(["3d hands", state.hands
      ? (state.hands.mesh ? "MANO<small>+mesh</small>" : "keypoints")
      : "<small>not exported</small>"]);
    statsEl.innerHTML = rows.map(([label, value]) => (
      `<div><dt>${label}</dt><dd>${value}</dd></div>`
    )).join("");
  }

  // Adaptive ruler: aims for ~8-12 major ticks whatever the clip length.
  function renderRuler() {
    const duration = state.duration;
    if (!duration) { rulerEl.innerHTML = ""; return; }
    const candidates = [1, 2, 5, 10, 15, 30, 60, 120, 300];
    const major = candidates.find((s) => duration / s <= 12) || 600;
    const minor = major / 5;
    let html = "";
    for (let t = 0; t <= duration + 1e-6; t += minor) {
      const pct = (t / duration) * 100;
      if (pct > 100) break;
      const isMajor = Math.abs(t / major - Math.round(t / major)) < 1e-6;
      html += `<i class="${isMajor ? "major" : ""}" style="left:${pct}%"></i>`;
      if (isMajor) {
        // Labels are centre-anchored, so pin the first and last to the ends
        // instead of letting them hang off the edge of the ruler.
        const edge = pct < 1
          ? "left:0;transform:none"
          : (pct > 99 ? "right:0;left:auto;transform:none" : `left:${pct}%`);
        html += `<b style="${edge}">${fmtShort(t)}</b>`;
      }
    }
    rulerEl.innerHTML = html;
  }

  function fmtShort(seconds) {
    const m = Math.floor(seconds / 60);
    const s = Math.round(seconds - m * 60);
    return `${m}:${String(s).padStart(2, "0")}`;
  }

  // Tooltip follows the pointer across the track and names the segment under it.
  function showTip(event) {
    const rect = track.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
    const time = ratio * (state.duration || 0);
    const segment = state.segments.length ? currentSegment(time) : null;
    if (!segment) { tipEl.classList.remove("on"); return; }
    const action = actionLabel(segment);
    const label = action.action || (action.verb ? `${action.verb} → ${action.object}` : (segment.scene || "segment"));
    tipEl.innerHTML = `<b>${escapeHtml(label)}</b><s>${fmt(time)} · segment ${escapeHtml(segment.id)}</s>`;
    const host = timeline.getBoundingClientRect();
    tipEl.style.left = `${event.clientX - host.left}px`;
    tipEl.classList.add("on");
  }

  function setPlayGlyph(playing) {
    playGlyph.textContent = playing ? "❚❚" : "▶";
    playButton.dataset.playing = String(playing);
    playButton.setAttribute("aria-label", playing ? "Pause" : "Play");
  }

  function toggle() {
    if (video.paused) video.play();
    else video.pause();
  }

  function fitFrame() {
    const vw = video.videoWidth || 16;
    const vh = video.videoHeight || 9;
    // Drives the stacked-layout stage aspect (see the <=1100px block in the CSS).
    if (video.videoWidth) stage.style.setProperty("--stage-ar", `${vw} / ${vh}`);
    // The stage padding keeps the corner brackets and HUD chips off the panel
    // edge; fit inside it so a 16:9 clip uses the full column width.
    const pad = 12;
    const scale = Math.min(
      (stage.clientWidth - pad * 2) / vw,
      (stage.clientHeight - pad * 2) / vh,
    );
    if (!Number.isFinite(scale) || scale <= 0) return;
    frame.style.width = `${Math.floor(vw * scale)}px`;
    frame.style.height = `${Math.floor(vh * scale)}px`;
    overlay.width = vw;
    overlay.height = vh;
    const host = world.parentElement;
    const side = Math.max(2, Math.floor(Math.min(host.clientWidth - 8, host.clientHeight - 8)));
    world.width = side;
    world.height = side;
    world.style.width = `${side}px`;
    world.style.height = `${side}px`;
    // Assigning canvas.width blanks the bitmap even when the value is unchanged,
    // so drop the cached stamp or drawWorld would skip the repaint.
    state.worldStamp = "";
  }

  function frameIndex(time) {
    const hands = state.hands;
    if (!hands) return 0;
    const fps = hands.fps || 30;
    return Math.max(0, Math.min(hands.n - 1, Math.round(time * fps)));
  }

  function jointWorld(hand, frame, joint) {
    const hands = state.hands;
    const base = ((hand * hands.n + frame) * 21 + joint) * 3;
    return [hands.world[base], hands.world[base + 1], hands.world[base + 2]];
  }

  function validAt(hand, frame) {
    return state.hands.valid[hand * state.hands.n + frame] > 0;
  }

  function toCamera(frame, point) {
    const hands = state.hands;
    const r = frame * 9;
    const t = frame * 3;
    const x = point[0];
    const y = point[1];
    const z = point[2];
    return [
      hands.R[r] * x + hands.R[r + 1] * y + hands.R[r + 2] * z + hands.t[t],
      hands.R[r + 3] * x + hands.R[r + 4] * y + hands.R[r + 5] * z + hands.t[t + 1],
      hands.R[r + 6] * x + hands.R[r + 7] * y + hands.R[r + 8] * z + hands.t[t + 2],
    ];
  }

  function project(point, width, height) {
    const z = Math.abs(point[2]) < 1e-6 ? 1e-6 : point[2];
    const focal = state.hands.focal;
    return [
      focal * point[0] / z + width / 2,
      focal * point[1] / z + height / 2,
    ];
  }

  function drawOverlay(time) {
    const width = overlay.width;
    const height = overlay.height;
    octx.clearRect(0, 0, width, height);
    if (!state.hands || !width) return;
    const frame = frameIndex(time);
    for (let hand = 0; hand < 2; hand += 1) {
      if (!validAt(hand, frame)) continue;
      for (let step = 1; step <= 30; step += 1) {
        const future = frame + step;
        if (future >= state.hands.n || !validAt(hand, future)) continue;
        const cam = toCamera(frame, jointWorld(hand, future, 0));
        if (cam[2] <= 0.05) continue;
        const [u, v] = project(cam, state.hands.width, state.hands.height);
        const alpha = (1 - step / 30) * 0.85;
        octx.fillStyle = `rgba(${TRAIL_RGB[hand]},${alpha})`;
        octx.beginPath();
        octx.arc(u, v, 3.2, 0, Math.PI * 2);
        octx.fill();
      }
      const points = [];
      for (let joint = 0; joint < 21; joint += 1) {
        points.push(project(toCamera(frame, jointWorld(hand, frame, joint)), state.hands.width, state.hands.height));
      }
      octx.lineWidth = 3;
      octx.lineCap = "round";
      FINGERS.forEach((chain, finger) => {
        octx.strokeStyle = FINGER_COLORS[finger];
        octx.beginPath();
        chain.forEach((joint, index) => {
          const [u, v] = points[joint];
          if (index === 0) octx.moveTo(u, v);
          else octx.lineTo(u, v);
        });
        octx.stroke();
      });
      points.forEach(([u, v], joint) => {
        octx.fillStyle = joint === 0 ? "#f8fafc" : FINGER_COLORS[Math.floor((joint - 1) / 4)];
        octx.beginPath();
        octx.arc(u, v, joint === 0 ? 5 : 3.5, 0, Math.PI * 2);
        octx.fill();
      });
      const label = HAND_LABEL[hand];
      octx.fillStyle = label.color;
      octx.font = "700 28px sans-serif";
      octx.fillText(label.name, points[0][0] - 10, points[0][1] - 16);
    }
  }

  function renderAnnotation(time) {
    const sample = state.sample;
    if (!sample) return;
    if (!state.segments.length) {
      if (state.shownKey !== "empty") {
        state.shownKey = "empty";
        anno.innerHTML = `<span class="empty">Annotation not written yet</span>`;
      }
      return;
    }
    const segment = currentSegment(time) || state.segments[0];
    const start = num(segment.start_ts);
    const end = num(segment.end_ts);
    const progress = Math.max(0, Math.min(1, (time - start) / Math.max(1e-3, end - start)));
    const key = `${sample.id}:${segment.id}`;
    if (state.shownKey !== key) {
      state.shownKey = key;
      const label = actionLabel(segment);
      const error = state.errors.length
        ? `<span class="warn">${escapeHtml(state.errors.slice(0, 3).join("；"))}</span>`
        : "";
      const order = state.segments.indexOf(segment);
      anno.innerHTML = `
        <span class="seg">${order + 1} / ${state.segments.length}</span>
        <span class="scene">${escapeHtml(segment.scene || "scene")}</span>
        <span class="act"><b>${escapeHtml(label.verb)}</b><i>→</i><b>${escapeHtml(label.object)}</b></span>
        <span class="when">${start.toFixed(2)}s – ${end.toFixed(2)}s</span>
        <span class="bar"><span id="segbar"></span></span>
        ${error}`;
    }
    const bar = document.getElementById("segbar");
    if (bar) bar.style.width = `${(progress * 100).toFixed(1)}%`;
  }

  function actionLabel(segment) {
    const legacy = (segment.atomic_action || [])[0] || {};
    return {
      verb: segment.verb || legacy.verb || "",
      object: segment.object || legacy.object || "",
      action: segment.action || legacy.description || "",
    };
  }

  function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;",
    }[ch]));
  }

  function renderTimeline() {
    const duration = state.duration || 1;
    track.innerHTML = state.segments.map((segment, index) => {
      const start = num(segment.start_ts);
      const end = num(segment.end_ts);
      const left = (start / duration) * 100;
      const width = Math.max(0.4, ((end - start) / duration) * 100);
      const action = actionLabel(segment);
      const label = action.action || "";
      const title = label ? ` title="${escapeHtml(label)}"` : "";
      const color = BLOCKS[index % BLOCKS.length];
      return `<div class="block" data-id="${escapeHtml(segment.id)}" data-t="${start}"${title} style="left:${left}%;width:${width}%;background:linear-gradient(180deg,${color},${color}d9)">${escapeHtml(label)}</div>`;
    }).join("");
    track.classList.toggle("empty", state.segments.length === 0);
    state.blockId = "";
  }

  // Outlines the block under the playhead so the timeline and the annotation
  // panel visibly agree on which segment is current.
  function markCurrentBlock(time) {
    const segment = state.segments.length ? currentSegment(time) : null;
    const id = segment ? String(segment.id) : "";
    if (id === state.blockId) return;
    state.blockId = id;
    track.querySelectorAll(".block").forEach((block) => {
      block.classList.toggle("is-current", block.dataset.id === id && id !== "");
    });
  }

  function scrub(event) {
    const rect = track.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
    video.currentTime = ratio * (state.duration || 0);
    timeline.classList.add("scrubbing");
    const move = (ev) => {
      const next = Math.max(0, Math.min(1, (ev.clientX - rect.left) / rect.width));
      video.currentTime = next * (state.duration || 0);
    };
    const up = () => {
      timeline.classList.remove("scrubbing");
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  }

  function parseMesh(buffer) {
    const header = new DataView(buffer);
    const n = header.getUint32(0, true);
    const nv = header.getUint32(4, true);
    const nf = header.getUint32(8, true);
    const faces = new Int32Array(buffer, 12, nf * 3);
    const vertOffset = 12 + nf * 3 * 4;
    const verts = new Float16Array(buffer, vertOffset, 2 * n * nv * 3);
    return { n, nv, nf, faces, verts };
  }

  const VIEW_R = (() => {
    const az = 40 * Math.PI / 180;
    const el = 20 * Math.PI / 180;
    const ca = Math.cos(az);
    const sa = Math.sin(az);
    const ce = Math.cos(el);
    const se = Math.sin(el);
    return new Float64Array([
      ca, 0, sa,
      se * sa, ce, -se * ca,
      -ce * sa, se, ce * ca,
    ]);
  })();
  const LIGHTS = [
    [0, 10, -15],
    [0, 10, 15],
  ].map((v) => {
    const n = Math.hypot(v[0], v[1], v[2]);
    return [v[0] / n, v[1] / n, v[2] / n];
  });
  const HAND_BASE = [
    [0.867, 0.698, 0.859],
    [0.196, 0.663, 0.898],
  ];
  const HAND_TRAIL = ["rgb(221,178,219)", "rgb(50,169,229)"];
  const SKEL_COLORS = ["#ff3c3c", "#ffa500", "#28dc28", "#1e82ff", "#e646e6"];
  const CAM_VERTS = [
    [-0.05, -0.05, 0], [0.05, -0.05, 0], [0.05, 0.05, 0], [-0.05, 0.05, 0], [0, 0, -0.1],
  ];
  const CAM_EDGES = [[0, 1], [1, 2], [2, 3], [3, 0], [0, 4], [1, 4], [2, 4], [3, 4]];
  const meshLayer = document.createElement("canvas");
  const meshCtx = meshLayer.getContext("2d");

  function viewPoint(point, center) {
    const x = point[0] - center[0];
    const y = point[1] - center[1];
    const z = point[2] - center[2];
    const r = VIEW_R;
    return [
      r[0] * x + r[1] * y + r[2] * z,
      r[3] * x + r[4] * y + r[5] * z,
      r[6] * x + r[7] * y + r[8] * z,
    ];
  }

  function screenOf(view, scale, ox, oy) {
    return [ox + scale * view[0], oy - scale * view[1]];
  }

  function shadeFaces(verts, faces, base) {
    const vcount = verts.length / 3;
    const nx = new Float32Array(vcount);
    const ny = new Float32Array(vcount);
    const nz = new Float32Array(vcount);
    const fcount = faces.length / 3;
    let mx = 0;
    let my = 0;
    let mz = 0;
    for (let i = 0; i < vcount; i += 1) {
      mx += verts[i * 3];
      my += verts[i * 3 + 1];
      mz += verts[i * 3 + 2];
    }
    mx /= vcount;
    my /= vcount;
    mz /= vcount;
    for (let i = 0; i < fcount; i += 1) {
      const a = faces[i * 3];
      const b = faces[i * 3 + 1];
      const c = faces[i * 3 + 2];
      const ax = verts[b * 3] - verts[a * 3];
      const ay = verts[b * 3 + 1] - verts[a * 3 + 1];
      const az = verts[b * 3 + 2] - verts[a * 3 + 2];
      const bx = verts[c * 3] - verts[a * 3];
      const by = verts[c * 3 + 1] - verts[a * 3 + 1];
      const bz = verts[c * 3 + 2] - verts[a * 3 + 2];
      const cx = ay * bz - az * by;
      const cy = az * bx - ax * bz;
      const cz = ax * by - ay * bx;
      nx[a] += cx; ny[a] += cy; nz[a] += cz;
      nx[b] += cx; ny[b] += cy; nz[b] += cz;
      nx[c] += cx; ny[c] += cy; nz[c] += cz;
    }
    let outward = 0;
    for (let i = 0; i < vcount; i += 1) {
      outward += nx[i] * (verts[i * 3] - mx) + ny[i] * (verts[i * 3 + 1] - my) + nz[i] * (verts[i * 3 + 2] - mz);
    }
    const flip = outward < 0 ? -1 : 1;
    const shade = new Float32Array(vcount);
    for (let i = 0; i < vcount; i += 1) {
      let x = nx[i] * flip;
      let y = ny[i] * flip;
      let z = nz[i] * flip;
      const len = Math.hypot(x, y, z) || 1;
      x /= len; y /= len; z /= len;
      let diff = 0;
      for (const light of LIGHTS) {
        diff += Math.max(0, x * light[0] + y * light[1] + z * light[2]);
      }
      shade[i] = 0.4 + 0.5 * diff;
    }
    const colors = new Array(fcount);
    for (let i = 0; i < fcount; i += 1) {
      const a = faces[i * 3];
      const b = faces[i * 3 + 1];
      const c = faces[i * 3 + 2];
      const f = (shade[a] + shade[b] + shade[c]) / 3;
      const r = Math.max(0, Math.min(255, Math.round(f * base[0] * 255)));
      const g = Math.max(0, Math.min(255, Math.round(f * base[1] * 255)));
      const bl = Math.max(0, Math.min(255, Math.round(f * base[2] * 255)));
      colors[i] = `rgb(${r},${g},${bl})`;
    }
    return colors;
  }

  function copyMeshVerts(mesh, hand, frame) {
    const out = new Float32Array(mesh.nv * 3);
    const base = (hand * mesh.n + frame) * mesh.nv * 3;
    for (let i = 0; i < out.length; i += 1) out[i] = mesh.verts[base + i];
    return out;
  }

  function drawShadedMesh(verts, faces, center, scale, ox, oy, base) {
    const vcount = verts.length / 3;
    const fcount = faces.length / 3;
    const sx = new Float32Array(vcount);
    const sy = new Float32Array(vcount);
    const sz = new Float32Array(vcount);
    for (let i = 0; i < vcount; i += 1) {
      const view = viewPoint([verts[i * 3], verts[i * 3 + 1], verts[i * 3 + 2]], center);
      const [u, v] = screenOf(view, scale, ox, oy);
      sx[i] = u;
      sy[i] = v;
      sz[i] = view[2];
    }
    const depth = new Float32Array(fcount);
    const order = new Uint32Array(fcount);
    for (let i = 0; i < fcount; i += 1) {
      const a = faces[i * 3];
      const b = faces[i * 3 + 1];
      const c = faces[i * 3 + 2];
      depth[i] = (sz[a] + sz[b] + sz[c]) / 3;
      order[i] = i;
    }
    order.sort((i, j) => depth[j] - depth[i]);
    const colors = shadeFaces(verts, faces, base);
    if (meshLayer.width !== world.width || meshLayer.height !== world.height) {
      meshLayer.width = world.width;
      meshLayer.height = world.height;
    } else {
      meshCtx.clearRect(0, 0, meshLayer.width, meshLayer.height);
    }
    const w = meshLayer.width;
    const h = meshLayer.height;
    for (let k = 0; k < fcount; k += 1) {
      const fi = order[k];
      const a = faces[fi * 3];
      const b = faces[fi * 3 + 1];
      const c = faces[fi * 3 + 2];
      const maxX = Math.max(sx[a], sx[b], sx[c]);
      const minX = Math.min(sx[a], sx[b], sx[c]);
      const maxY = Math.max(sy[a], sy[b], sy[c]);
      const minY = Math.min(sy[a], sy[b], sy[c]);
      if (maxX < 0 || minX >= w || maxY < 0 || minY >= h) continue;
      meshCtx.fillStyle = colors[fi];
      meshCtx.beginPath();
      meshCtx.moveTo(sx[a], sy[a]);
      meshCtx.lineTo(sx[b], sy[b]);
      meshCtx.lineTo(sx[c], sy[c]);
      meshCtx.closePath();
      meshCtx.fill();
    }
    wctx.save();
    wctx.globalAlpha = 0.85;
    wctx.drawImage(meshLayer, 0, 0);
    wctx.restore();
  }

  // Faint 10cm graticule behind the hands, for depth and a sense of scale.
  // The projection is orthographic with one uniform scale, so screen distance
  // equals projected metric distance: a screen-space grid at `scale * 0.1` is a
  // truthful 10cm reference. Drawn first, so it never tints the hand colors.
  function drawGroundGrid(width, height, scale, ox, oy) {
    const step = scale * 0.1;
    if (!Number.isFinite(step) || step < 6) return;
    wctx.save();
    wctx.strokeStyle = "rgba(129, 165, 214, 0.10)";
    wctx.lineWidth = 1;
    wctx.beginPath();
    for (let x = ox % step; x < width; x += step) {
      wctx.moveTo(Math.round(x) + 0.5, 0);
      wctx.lineTo(Math.round(x) + 0.5, height);
    }
    for (let y = oy % step; y < height; y += step) {
      wctx.moveTo(0, Math.round(y) + 0.5);
      wctx.lineTo(width, Math.round(y) + 0.5);
    }
    wctx.stroke();
    // Crosshair on the wrist-centred origin.
    wctx.strokeStyle = "rgba(129, 165, 214, 0.26)";
    wctx.beginPath();
    wctx.moveTo(ox - 7, oy + 0.5); wctx.lineTo(ox + 7, oy + 0.5);
    wctx.moveTo(ox + 0.5, oy - 7); wctx.lineTo(ox + 0.5, oy + 7);
    wctx.stroke();
    wctx.restore();
  }

  function drawWorld(time) {
    const width = world.width;
    const height = world.height;
    const frame = state.hands ? frameIndex(time) : -1;
    const stamp = `${state.sample ? state.sample.id : ""}:${frame}:${width}:${height}:${state.hands && state.hands.mesh ? 1 : 0}`;
    if (stamp === state.worldStamp) return;
    state.worldStamp = stamp;

    // Left transparent so the panel's own gradient reads through the canvas.
    wctx.clearRect(0, 0, width, height);
    if (!state.hands) {
      worldCaption.textContent = "World frame awaiting model output";
      wctx.textAlign = "center";
      wctx.fillStyle = "#8398b4";
      wctx.font = "13px system-ui, sans-serif";
      wctx.fillText("World frame is still waiting for model output", width / 2, height / 2 - 9);
      wctx.fillStyle = "#5b6e88";
      wctx.font = "11px ui-monospace, monospace";
      wctx.fillText("hands.npz → keypoints.npz + mesh.bin", width / 2, height / 2 + 13);
      wctx.textAlign = "start";
      return;
    }

    const hands = state.hands;
    const wrists = [];
    for (let hand = 0; hand < 2; hand += 1) {
      if (validAt(hand, frame)) wrists.push(jointWorld(hand, frame, 0));
    }
    if (wrists.length) {
      hands.center = wrists.reduce((acc, p) => [acc[0] + p[0], acc[1] + p[1], acc[2] + p[2]], [0, 0, 0])
        .map((v) => v / wrists.length);
    }
    const center = hands.center;
    const windowM = 0.5;
    const scale = (1 - 2 * 0.08) * Math.min(width, height) / windowM;
    const ox = width / 2;
    const oy = height / 2;
    const projectWorld = (point) => {
      const view = viewPoint(point, center);
      return screenOf(view, scale, ox, oy);
    };

    drawGroundGrid(width, height, scale, ox, oy);

    const trailStart = Math.max(0, frame - 60);
    wctx.strokeStyle = "rgba(168, 182, 198, 0.5)";
    wctx.lineWidth = 1;
    wctx.beginPath();
    for (let i = trailStart; i <= frame; i += 1) {
      const [u, v] = projectWorld([
        hands.camPos[i * 3], hands.camPos[i * 3 + 1], hands.camPos[i * 3 + 2],
      ]);
      if (i === trailStart) wctx.moveTo(u, v);
      else wctx.lineTo(u, v);
    }
    wctx.stroke();

    if (hands.camR) {
      const rb = frame * 9;
      const camT = [hands.camPos[frame * 3], hands.camPos[frame * 3 + 1], hands.camPos[frame * 3 + 2]];
      const pts = CAM_VERTS.map((p) => {
        const x = hands.camR[rb] * p[0] + hands.camR[rb + 1] * p[1] + hands.camR[rb + 2] * p[2] + camT[0];
        const y = hands.camR[rb + 3] * p[0] + hands.camR[rb + 4] * p[1] + hands.camR[rb + 5] * p[2] + camT[1];
        const z = hands.camR[rb + 6] * p[0] + hands.camR[rb + 7] * p[1] + hands.camR[rb + 8] * p[2] + camT[2];
        return projectWorld([x, y, z]);
      });
      wctx.strokeStyle = "rgba(180, 194, 210, 0.85)";
      wctx.lineWidth = 1.6;
      CAM_EDGES.forEach(([a, b]) => {
        wctx.beginPath();
        wctx.moveTo(pts[a][0], pts[a][1]);
        wctx.lineTo(pts[b][0], pts[b][1]);
        wctx.stroke();
      });
      wctx.fillStyle = "rgba(160, 176, 194, 0.9)";
      wctx.font = "10px ui-monospace, monospace";
      wctx.fillText("cam", pts[4][0] + 7, pts[4][1] - 4);
    }

    if (hands.mesh && hands.mesh.n === hands.n) {
      for (let hand = 0; hand < 2; hand += 1) {
        if (!validAt(hand, frame)) continue;
        const verts = copyMeshVerts(hands.mesh, hand, frame);
        drawShadedMesh(verts, hands.mesh.faces, center, scale, ox, oy, HAND_BASE[hand]);
      }
    }

    for (let hand = 0; hand < 2; hand += 1) {
      wctx.strokeStyle = HAND_TRAIL[hand];
      wctx.lineWidth = 1.5;
      wctx.beginPath();
      let started = false;
      for (let i = trailStart; i <= frame; i += 1) {
        if (!validAt(hand, i)) continue;
        const [u, v] = projectWorld(jointWorld(hand, i, 0));
        if (!started) {
          wctx.moveTo(u, v);
          started = true;
        } else wctx.lineTo(u, v);
      }
      wctx.stroke();
      if (!validAt(hand, frame)) continue;
      const points = [];
      for (let joint = 0; joint < 21; joint += 1) points.push(projectWorld(jointWorld(hand, frame, joint)));
      wctx.lineWidth = 3;
      wctx.lineCap = "round";
      // Dark halo under the per-finger strokes. On the dark plate this reads as
      // a contact shadow rather than the outline it was on white, which keeps
      // the official finger colors legible over the shaded mesh.
      wctx.strokeStyle = "rgba(2, 6, 12, 0.5)";
      FINGERS.forEach((chain) => {
        wctx.beginPath();
        chain.forEach((joint, index) => {
          const [u, v] = points[joint];
          if (index === 0) wctx.moveTo(u, v);
          else wctx.lineTo(u, v);
        });
        wctx.stroke();
      });
      wctx.lineWidth = 1.5;
      FINGERS.forEach((chain, finger) => {
        wctx.strokeStyle = SKEL_COLORS[finger];
        wctx.beginPath();
        chain.forEach((joint, index) => {
          const [u, v] = points[joint];
          if (index === 0) wctx.moveTo(u, v);
          else wctx.lineTo(u, v);
        });
        wctx.stroke();
      });
      points.forEach(([u, v], joint) => {
        wctx.fillStyle = joint === 0 ? "#f5f5f5" : SKEL_COLORS[Math.floor((joint - 1) / 4)];
        wctx.beginPath();
        wctx.arc(u, v, joint === 0 ? 3 : 2, 0, Math.PI * 2);
        wctx.fill();
      });
    }

    worldCaption.textContent = `~${Math.round(width / scale * 100)}cm view`;
  }

  function paint(drawWorldNow) {
    const time = video.currentTime || 0;
    state.duration = video.duration || state.duration;
    clock.textContent = `${fmt(time)} / ${fmt(state.duration)}`;
    const ratio = state.duration ? time / state.duration : 0;
    const trackRect = track.getBoundingClientRect();
    const hostRect = timeline.getBoundingClientRect();
    playhead.style.left = `${trackRect.left - hostRect.left + ratio * trackRect.width}px`;
    if (playheadTime) playheadTime.textContent = fmt(time);
    updateFrameBadge(time);
    markCurrentBlock(time);
    drawOverlay(time);
    if (drawWorldNow) drawWorld(time);
    renderAnnotation(time);
  }

  // The shaded mesh is the expensive part. Redraw it at about 10fps so the
  // video decoder keeps the main thread; the skeleton still follows every frame.
  function loop(now) {
    const frame = state.hands
      ? frameIndex(video.currentTime || 0)
      : Math.floor((video.currentTime || 0) * 30);
    const frameChanged = frame !== paintedFrame;
    if (frame !== paintedFrame) paintedFrame = frame;
    const drawWorldNow = frameChanged && (video.paused || now - lastWorldDraw > 100);
    if (drawWorldNow) lastWorldDraw = now;
    if (frameChanged) paint(drawWorldNow);
    requestAnimationFrame(loop);
  }
  function updateFrameBadge(time) {
    const w = state.hands ? state.hands.width : video.videoWidth;
    const h = state.hands ? state.hands.height : video.videoHeight;
    hudTrack.textContent = w && h ? `${w} × ${h}` : "";
    if (state.hands && state.hands.fps) {
      hudFrame.textContent =
        `T ${fmt(time)}   F ${String(frameIndex(time)).padStart(5, "0")} / ${state.hands.n}`;
      egoNote.textContent = `undistorted · ${state.hands.fps.toFixed(0)} fps · 21-pt skeleton`;
    } else {
      hudFrame.textContent = `T ${fmt(time)}`;
      egoNote.textContent = "undistorted · no hand estimates";
    }
  }

  boot().catch((error) => {
    setStatus("Failed to load", "error");
    anno.textContent = String(error);
  });
})();

from __future__ import annotations

import gc
import os
import tempfile
import threading
from pathlib import Path

import psutil
import whisper
from flask import Flask, jsonify, render_template_string, request
from werkzeug.utils import secure_filename
from whisper import _MODELS
from whisper.tokenizer import LANGUAGES


SUPPORTED_MODELS = ("tiny", "base", "large")
DEFAULT_MODEL_NAME = os.environ.get("WHISPER_WEB_MODEL", "large")
if DEFAULT_MODEL_NAME not in SUPPORTED_MODELS:
    DEFAULT_MODEL_NAME = "large"

HOST = os.environ.get("WHISPER_WEB_HOST", "127.0.0.1")
PORT = int(os.environ.get("WHISPER_WEB_PORT", "5000"))
CACHE_DIR = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "whisper"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024
app.config["JSON_AS_ASCII"] = False

_state_lock = threading.Lock()
_current_model = None
_current_model_name = None


HTML = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Whisper UI</title>
    <style>
      :root {
        --bg: #f4f4f1;
        --bg-strong: #ffffff;
        --fg: #111111;
        --muted: #6f6f6b;
        --line: #d8d8d2;
        --line-strong: #bcbcb4;
        --soft: #efefea;
        --focus: #111111;
        --radius: 24px;
        --panel-shadow: 0 24px 80px rgba(17, 17, 17, 0.06);
      }

      * {
        box-sizing: border-box;
      }

      body {
        margin: 0;
        font-family: "Avenir Next", "Helvetica Neue", Helvetica, Arial, sans-serif;
        color: var(--fg);
        background:
          radial-gradient(circle at top left, rgba(17, 17, 17, 0.04), transparent 28%),
          linear-gradient(180deg, #fafaf8 0%, #f4f4f1 100%);
      }

      .shell {
        min-height: 100vh;
        padding: 28px;
      }

      .panel {
        width: min(1180px, 100%);
        margin: 0 auto;
        padding: 30px;
        border: 1px solid rgba(17, 17, 17, 0.08);
        border-radius: var(--radius);
        background: rgba(255, 255, 255, 0.86);
        box-shadow: var(--panel-shadow);
        backdrop-filter: blur(16px);
      }

      .hero {
        display: grid;
        grid-template-columns: minmax(0, 1fr) 220px;
        gap: 20px;
        align-items: end;
        padding-bottom: 24px;
        border-bottom: 1px solid var(--line);
      }

      .eyebrow,
      .section-title,
      label,
      th,
      .footer,
      .meta,
      .meta-label {
        font-family: "SFMono-Regular", "Menlo", "Monaco", "Courier New", monospace;
        letter-spacing: 0.08em;
      }

      .eyebrow {
        display: inline-block;
        margin-bottom: 18px;
        font-size: 11px;
        text-transform: uppercase;
        color: var(--muted);
      }

      h1 {
        margin: 0;
        max-width: 9ch;
        font-size: clamp(34px, 4vw, 52px);
        font-weight: 600;
        letter-spacing: -0.05em;
        line-height: 0.94;
      }

      .sub {
        display: none;
      }

      .hero-meta {
        display: grid;
        gap: 12px;
        align-content: start;
      }

      .meta-card {
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 14px 16px;
        background: rgba(255, 255, 255, 0.72);
      }

      .meta-label {
        margin: 0 0 6px;
        font-size: 11px;
        text-transform: uppercase;
        color: var(--muted);
      }

      .meta-value {
        margin: 0;
        font-size: 18px;
        letter-spacing: -0.03em;
      }

      .workspace {
        display: grid;
        grid-template-columns: minmax(320px, 420px) minmax(0, 1fr);
        gap: 18px;
        margin-top: 24px;
      }

      .stack {
        display: grid;
        gap: 16px;
      }

      .card {
        border: 1px solid var(--line);
        border-radius: 22px;
        padding: 18px;
        background: rgba(255, 255, 255, 0.86);
      }

      .grid {
        display: grid;
        gap: 14px;
      }

      .inline-grid {
        display: grid;
        grid-template-columns: minmax(0, 1fr) 180px 140px;
        gap: 10px;
      }

      .section {
        margin-top: 0;
      }

      label,
      .section-title {
        display: block;
        margin-bottom: 8px;
        font-size: 12px;
        text-transform: uppercase;
        color: var(--muted);
      }

      input[type="file"],
      select,
      textarea,
      button {
        width: 100%;
        border: 1px solid var(--line);
        border-radius: 16px;
        background: var(--bg-strong);
        color: var(--fg);
        font: inherit;
      }

      input[type="file"],
      select,
      .model-button,
      .record-button,
      .clear-button,
      .file-trigger,
      #submit-btn {
        min-height: 48px;
        padding: 0 14px;
      }

      .visually-hidden {
        position: absolute;
        width: 1px;
        height: 1px;
        padding: 0;
        margin: -1px;
        overflow: hidden;
        clip: rect(0, 0, 0, 0);
        white-space: nowrap;
        border: 0;
      }

      textarea {
        min-height: 620px;
        padding: 18px;
        resize: vertical;
        line-height: 1.6;
        font-size: 15px;
        background: rgba(255, 255, 255, 0.92);
      }

      #submit-btn,
      .model-button.active,
      .record-button.recording {
        background: var(--fg);
        color: var(--bg-strong);
        border-color: var(--fg);
      }

      .model-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 10px;
      }

      .model-button,
      .file-trigger,
      .record-button,
      .clear-button,
      #submit-btn {
        cursor: pointer;
        transition:
          opacity 160ms ease,
          background 160ms ease,
          color 160ms ease,
          border-color 160ms ease,
          transform 160ms ease;
      }

      button:hover:enabled {
        opacity: 0.92;
        transform: translateY(-1px);
      }

      .file-trigger:hover {
        opacity: 0.92;
        transform: translateY(-1px);
      }

      button:disabled {
        opacity: 0.45;
        cursor: not-allowed;
      }

      .status {
        min-height: 24px;
        margin: 0;
        color: var(--muted);
        font-size: 13px;
      }

      .meta {
        margin: 10px 0 0;
        color: var(--muted);
        font-size: 12px;
      }

      .audio-preview {
        width: 100%;
        margin-top: 10px;
        filter: grayscale(1);
      }

      .file-trigger {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border: 1px solid var(--line);
        border-radius: 16px;
        background: var(--bg-strong);
        color: var(--fg);
        font-family: "Avenir Next", "Helvetica Neue", Helvetica, Arial, sans-serif;
        font-size: 16px;
        letter-spacing: -0.01em;
        text-transform: none;
        margin-bottom: 0;
      }

      .result-header {
        display: flex;
        align-items: end;
        justify-content: space-between;
        gap: 16px;
        margin-bottom: 12px;
      }

      .result-title {
        margin: 0;
        font-size: 24px;
        letter-spacing: -0.04em;
      }

      .result-sub {
        margin: 4px 0 0;
        color: var(--muted);
        font-size: 13px;
      }

      table {
        width: 100%;
        border-collapse: collapse;
        overflow: hidden;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.7);
        font-size: 14px;
      }

      th,
      td {
        text-align: left;
        border-top: 1px solid var(--line);
        padding: 12px 10px;
      }

      th {
        font-size: 12px;
        text-transform: uppercase;
        color: var(--muted);
        background: rgba(17, 17, 17, 0.02);
      }

      td strong {
        font-weight: 600;
      }

      .footer {
        margin-top: 10px;
        font-size: 12px;
        color: var(--muted);
      }

      input[type="file"]:focus-visible,
      select:focus-visible,
      textarea:focus-visible,
      button:focus-visible {
        outline: 2px solid var(--focus);
        outline-offset: 2px;
        border-color: var(--focus);
      }

      @media (prefers-reduced-motion: reduce) {
        *,
        *::before,
        *::after {
          animation: none !important;
          transition: none !important;
          scroll-behavior: auto !important;
        }
      }

      @media (max-width: 980px) {
        .hero,
        .workspace {
          grid-template-columns: 1fr;
        }
      }

      @media (max-width: 640px) {
        .shell {
          padding: 14px;
        }

        .panel {
          padding: 18px;
        }

        .model-grid,
        .inline-grid {
          grid-template-columns: 1fr;
        }

        table,
        thead,
        tbody,
        th,
        td,
        tr {
          display: block;
        }

        thead {
          display: none;
        }

        tr {
          border-top: 1px solid var(--line);
          padding: 10px 0;
        }

        td {
          border-top: 0;
          padding: 6px 0;
        }
      }
    </style>
  </head>
  <body>
    <main class="shell">
      <section class="panel">
        <header class="hero">
          <div>
            <span class="eyebrow">Whisper</span>
            <h1>Whisper Transcript Console</h1>
          </div>

          <div class="hero-meta">
            <div class="meta-card">
              <p class="meta-label">Mode</p>
              <p class="meta-value" id="hero-current-model">{{ default_model_name }}</p>
            </div>
          </div>
        </header>

        <div class="workspace">
          <div class="stack">
            <div class="card section">
              <div class="section-title">Model Switch</div>
              <div class="model-grid">
                {% for model_name in models %}
                <button class="model-button{% if model_name == default_model_name %} active{% endif %}" data-model="{{ model_name }}" type="button">
                  {{ model_name }}
                </button>
                {% endfor %}
              </div>
              <p id="model-status" class="meta">Current target model: {{ default_model_name }}</p>
            </div>

            <form id="transcribe-form" class="card grid section">
              <div>
                <label for="audio">Audio Source</label>
                <div class="inline-grid">
                  <input id="audio" class="visually-hidden" name="audio" type="file" accept="audio/*,video/*" />
                  <label for="audio" class="file-trigger">Choose File</label>
                  <button id="record-btn" class="record-button" type="button">Start Recording</button>
                  <button id="clear-recording-btn" class="clear-button" type="button" disabled>Clear</button>
                </div>
                <p id="audio-source" class="meta">No file selected. No recording captured.</p>
                <audio id="audio-preview" class="audio-preview" controls hidden></audio>
              </div>

              <div>
                <label for="language">Language</label>
                <select id="language" name="language">
                  <option value="">Auto detect</option>
                  {% for code, name in languages %}
                  <option value="{{ code }}">{{ name }}</option>
                  {% endfor %}
                </select>
              </div>

              <button id="submit-btn" type="submit">Generate Text</button>
              <p id="status" class="status"></p>
            </form>

            <div class="card section">
              <div class="section-title">Live Memory</div>
              <table>
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>Cached</th>
                    <th>Loaded</th>
                    <th>File Size</th>
                    <th>Process RSS</th>
                  </tr>
                </thead>
                <tbody id="model-table-body"></tbody>
              </table>
              <p id="memory-summary" class="footer">Refreshing...</p>
            </div>
          </div>

          <div class="card section">
            <div class="result-header">
              <div>
                <p class="section-title">Output</p>
                <h2 class="result-title">Transcript</h2>
              </div>
            </div>
            <textarea id="result" readonly placeholder="Your transcript will appear here."></textarea>
          </div>
        </div>
      </section>
    </main>

    <script>
      const form = document.getElementById("transcribe-form");
      const statusNode = document.getElementById("status");
      const submitBtn = document.getElementById("submit-btn");
      const resultNode = document.getElementById("result");
      const modelStatusNode = document.getElementById("model-status");
      const memorySummaryNode = document.getElementById("memory-summary");
      const tableBody = document.getElementById("model-table-body");
      const modelButtons = [...document.querySelectorAll(".model-button")];
      const heroCurrentModelNode = document.getElementById("hero-current-model");
      const fileInput = document.getElementById("audio");
      const recordBtn = document.getElementById("record-btn");
      const clearRecordingBtn = document.getElementById("clear-recording-btn");
      const audioSourceNode = document.getElementById("audio-source");
      const audioPreviewNode = document.getElementById("audio-preview");

      let selectedModel = "{{ default_model_name }}";
      let recordedBlob = null;
      let mediaStream = null;
      let audioContext = null;
      let sourceNode = null;
      let processorNode = null;
      let silentGainNode = null;
      let isRecording = false;
      let recordingChunks = [];
      let recordingSampleRate = 44100;

      function getRecordingFilename() {
        return "recording.wav";
      }

      async function stopRecordingGraph() {
        isRecording = false;

        if (processorNode) {
          processorNode.disconnect();
          processorNode.onaudioprocess = null;
          processorNode = null;
        }

        if (sourceNode) {
          sourceNode.disconnect();
          sourceNode = null;
        }

        if (silentGainNode) {
          silentGainNode.disconnect();
          silentGainNode = null;
        }

        if (mediaStream) {
          mediaStream.getTracks().forEach((track) => track.stop());
          mediaStream = null;
        }

        if (audioContext) {
          await audioContext.close();
          audioContext = null;
        }
      }

      function updateAudioSourceLabel() {
        if (recordedBlob) {
          audioSourceNode.textContent = `Using recorded audio: ${getRecordingFilename()}`;
          clearRecordingBtn.disabled = false;
          return;
        }

        if (fileInput.files.length) {
          audioSourceNode.textContent = `Using uploaded file: ${fileInput.files[0].name}`;
          clearRecordingBtn.disabled = true;
          return;
        }

        audioSourceNode.textContent = "No file selected. No recording captured.";
        clearRecordingBtn.disabled = true;
      }

      function setAudioPreview(blob, filename) {
        if (!blob) {
          audioPreviewNode.hidden = true;
          audioPreviewNode.removeAttribute("src");
          return;
        }

        audioPreviewNode.hidden = false;
        audioPreviewNode.src = URL.createObjectURL(blob);
        audioPreviewNode.title = filename;
      }

      function mergeAudioChunks(chunks) {
        const totalLength = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
        const merged = new Float32Array(totalLength);
        let offset = 0;

        for (const chunk of chunks) {
          merged.set(chunk, offset);
          offset += chunk.length;
        }

        return merged;
      }

      function writeWavString(view, offset, value) {
        for (let index = 0; index < value.length; index += 1) {
          view.setUint8(offset + index, value.charCodeAt(index));
        }
      }

      function encodeWav(samples, sampleRate) {
        const buffer = new ArrayBuffer(44 + samples.length * 2);
        const view = new DataView(buffer);

        writeWavString(view, 0, "RIFF");
        view.setUint32(4, 36 + samples.length * 2, true);
        writeWavString(view, 8, "WAVE");
        writeWavString(view, 12, "fmt ");
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, 1, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * 2, true);
        view.setUint16(32, 2, true);
        view.setUint16(34, 16, true);
        writeWavString(view, 36, "data");
        view.setUint32(40, samples.length * 2, true);

        let offset = 44;
        for (let index = 0; index < samples.length; index += 1) {
          const sample = Math.max(-1, Math.min(1, samples[index]));
          view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
          offset += 2;
        }

        return new Blob([view], { type: "audio/wav" });
      }

      function setActiveModelButton(modelName) {
        modelButtons.forEach((button) => {
          button.classList.toggle("active", button.dataset.model === modelName);
        });
      }

      function formatMb(value) {
        if (value === null || value === undefined) {
          return "--";
        }
        return `${value.toFixed(1)} MB`;
      }

      async function refreshStatus() {
        const response = await fetch("/api/status");
        const payload = await response.json();

        tableBody.innerHTML = payload.models.map((model) => `
          <tr>
            <td><strong>${model.name}</strong></td>
            <td>${model.cached ? "Yes" : "No"}</td>
            <td>${model.loaded ? "Yes" : "No"}</td>
            <td>${formatMb(model.cache_size_mb)}</td>
            <td>${formatMb(model.process_memory_mb)}</td>
          </tr>
        `).join("");

        const currentModel = payload.current_model || "none";
        modelStatusNode.textContent = `Current target model: ${selectedModel} | Loaded now: ${currentModel}`;
        memorySummaryNode.textContent = `Process RSS: ${formatMb(payload.process_memory_mb)} | Device: ${payload.device}`;
        heroCurrentModelNode.textContent = currentModel === "none" ? selectedModel : currentModel;
      }

      async function switchModel(modelName) {
        selectedModel = modelName;
        setActiveModelButton(modelName);
        statusNode.textContent = `Switching to ${modelName}.`;

        try {
          const response = await fetch("/api/models/select", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ model: modelName }),
          });
          const payload = await response.json();
          if (!response.ok) {
            throw new Error(payload.error || "Model switch failed.");
          }

          statusNode.textContent = `Model ready: ${payload.current_model}.`;
          await refreshStatus();
        } catch (error) {
          statusNode.textContent = error.message;
        }
      }

      modelButtons.forEach((button) => {
        button.addEventListener("click", () => {
          switchModel(button.dataset.model);
        });
      });

      fileInput.addEventListener("change", () => {
        if (fileInput.files.length) {
          recordedBlob = null;
          setAudioPreview(fileInput.files[0], fileInput.files[0].name);
        } else {
          setAudioPreview(null);
        }

        updateAudioSourceLabel();
      });

      clearRecordingBtn.addEventListener("click", () => {
        if (!recordedBlob || isRecording) {
          return;
        }

        recordedBlob = null;
        setAudioPreview(null);
        updateAudioSourceLabel();
        statusNode.textContent = "Recording cleared.";
      });

      recordBtn.addEventListener("click", async () => {
        if (!window.AudioContext && !window.webkitAudioContext) {
          statusNode.textContent = "This browser does not support recording.";
          return;
        }

        if (isRecording) {
          const mergedSamples = mergeAudioChunks(recordingChunks);
          recordedBlob = encodeWav(mergedSamples, recordingSampleRate);
          setAudioPreview(recordedBlob, getRecordingFilename());
          updateAudioSourceLabel();
          recordBtn.textContent = "Start Recording";
          recordBtn.classList.remove("recording");
          await stopRecordingGraph();
          statusNode.textContent = "Recording captured. Generate text to transcribe it.";
          return;
        }

        try {
          mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: {
              channelCount: 1,
              echoCancellation: true,
              noiseSuppression: true,
              autoGainControl: true,
            },
          });
          recordingChunks = [];
          fileInput.value = "";

          const AudioContextClass = window.AudioContext || window.webkitAudioContext;
          audioContext = new AudioContextClass();
          recordingSampleRate = audioContext.sampleRate;
          sourceNode = audioContext.createMediaStreamSource(mediaStream);
          processorNode = audioContext.createScriptProcessor(4096, 1, 1);
          silentGainNode = audioContext.createGain();
          silentGainNode.gain.value = 0;

          // 直接采样 PCM 再导出 WAV，避免某些浏览器的 MediaRecorder 录出空声道。
          processorNode.onaudioprocess = (event) => {
            if (!isRecording) {
              return;
            }

            const input = event.inputBuffer.getChannelData(0);
            recordingChunks.push(new Float32Array(input));
          };

          sourceNode.connect(processorNode);
          processorNode.connect(silentGainNode);
          silentGainNode.connect(audioContext.destination);

          isRecording = true;
          recordBtn.textContent = "Stop Recording";
          recordBtn.classList.add("recording");
          clearRecordingBtn.disabled = true;
          statusNode.textContent = "Recording...";
        } catch (error) {
          await stopRecordingGraph();
          statusNode.textContent = "Microphone access failed.";
        }
      });

      form.addEventListener("submit", async (event) => {
        event.preventDefault();

        if (!fileInput.files.length && !recordedBlob) {
          statusNode.textContent = "Please choose an audio file or record one.";
          return;
        }

        const formData = new FormData(form);
        formData.set("model", selectedModel);
        if (recordedBlob) {
          formData.set("audio", recordedBlob, getRecordingFilename());
        }
        submitBtn.disabled = true;
        recordBtn.disabled = true;
        clearRecordingBtn.disabled = true;
        resultNode.value = "";
        statusNode.textContent = `Transcribing with ${selectedModel}.`;

        try {
          const response = await fetch("/api/transcribe", {
            method: "POST",
            body: formData,
          });

          const payload = await response.json();
          if (!response.ok) {
            throw new Error(payload.error || "Transcription failed.");
          }

          resultNode.value = payload.text || "";
          statusNode.textContent = `Done with ${payload.model}.`;
          await refreshStatus();
        } catch (error) {
          statusNode.textContent = error.message;
        } finally {
          submitBtn.disabled = false;
          recordBtn.disabled = false;
          updateAudioSourceLabel();
        }
      });

      updateAudioSourceLabel();
      refreshStatus();
      setInterval(refreshStatus, 2000);
    </script>
  </body>
</html>
"""


def get_device_label() -> str:
    return "cuda" if whisper.torch.cuda.is_available() else "cpu"


def get_process_memory_mb() -> float:
    process = psutil.Process(os.getpid())
    return round(process.memory_info().rss / 1024 / 1024, 1)


def get_model_cache_path(model_name: str) -> Path:
    return CACHE_DIR / Path(_MODELS[model_name]).name


def build_model_rows() -> list[dict[str, object]]:
    rows = []
    current_memory = get_process_memory_mb()

    for model_name in SUPPORTED_MODELS:
        cache_path = get_model_cache_path(model_name)
        cached = cache_path.exists()
        rows.append(
            {
                "name": model_name,
                "cached": cached,
                "loaded": model_name == _current_model_name and _current_model is not None,
                "cache_size_mb": round(cache_path.stat().st_size / 1024 / 1024, 1)
                if cached
                else None,
                "process_memory_mb": current_memory
                if model_name == _current_model_name and _current_model is not None
                else None,
            }
        )

    return rows


def load_selected_model(model_name: str):
    global _current_model, _current_model_name

    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {model_name}")

    with _state_lock:
        if _current_model_name == model_name and _current_model is not None:
            return _current_model

        if _current_model is not None:
            # 切模型前先释放旧模型，避免多个权重同时常驻导致内存被放大。
            del _current_model
            _current_model = None
            _current_model_name = None
            gc.collect()

        _current_model = whisper.load_model(model_name)
        _current_model_name = model_name
        return _current_model


@app.get("/")
def index():
    languages = sorted((code, name.title()) for code, name in LANGUAGES.items())
    return render_template_string(
        HTML,
        default_model_name=DEFAULT_MODEL_NAME,
        languages=languages,
        models=SUPPORTED_MODELS,
    )


@app.get("/api/status")
def get_status():
    return jsonify(
        {
            "current_model": _current_model_name,
            "device": get_device_label(),
            "models": build_model_rows(),
            "process_memory_mb": get_process_memory_mb(),
        }
    )


@app.post("/api/models/select")
def select_model():
    payload = request.get_json(silent=True) or request.form
    model_name = payload.get("model", DEFAULT_MODEL_NAME)

    try:
        load_selected_model(model_name)
        return jsonify(
            {
                "current_model": _current_model_name,
                "models": build_model_rows(),
                "process_memory_mb": get_process_memory_mb(),
            }
        )
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400


@app.post("/api/transcribe")
def transcribe_audio():
    file = request.files.get("audio")
    if file is None or file.filename == "":
        return jsonify({"error": "Please upload an audio file."}), 400

    model_name = request.form.get("model") or DEFAULT_MODEL_NAME
    language = request.form.get("language") or None
    suffix = Path(secure_filename(file.filename)).suffix or ".tmp"

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            file.save(temp_file)
            temp_path = temp_file.name

        model = load_selected_model(model_name)
        result = model.transcribe(temp_path, language=language)
        return jsonify({"model": model_name, "text": result.get("text", "").strip()})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500
    finally:
        # 上传文件先落到临时目录，让 ffmpeg 和 whisper 走普通文件路径最稳定。
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=False)

// ── Element refs ────────────────────────────────────────────────────────────
const submitBtn   = document.getElementById("submit");
const statusBox   = document.getElementById("status");
const fileInput   = document.getElementById("file");
const urlInput    = document.getElementById("url");
const engineInput = document.getElementById("engine");
const downloadLink = document.getElementById("download");

const qaPanel    = document.getElementById("qa-panel");
const indexBar   = document.getElementById("index-bar");
const indexIcon  = document.getElementById("index-icon");
const indexLabel = document.getElementById("index-label");
const questionEl = document.getElementById("question");
const askBtn     = document.getElementById("ask-btn");
const answerBox  = document.getElementById("answer-box");
const sourcesEl  = document.getElementById("sources");

// ── State ────────────────────────────────────────────────────────────────────
let pollTimer      = null;
let indexPollTimer = null;
let activeJobId    = null;

// ── Helpers ──────────────────────────────────────────────────────────────────
function setStatus(text) { statusBox.textContent = text; }

function stopPolling() {
  if (pollTimer)      { clearInterval(pollTimer);      pollTimer      = null; }
  if (indexPollTimer) { clearInterval(indexPollTimer); indexPollTimer = null; }
}

async function fetchJson(url, opts = {}) {
  const res = await fetch(url, opts);
  const body = await res.json();
  if (!res.ok) throw new Error(body.detail || JSON.stringify(body));
  return body;
}

// ── Transcription polling ────────────────────────────────────────────────────
function beginTranscriptionPolling(jobId) {
  stopPolling();
  pollTimer = setInterval(async () => {
    try {
      const job = await fetchJson(`/transcriptions/${jobId}`);
      setStatus(
        `Job: ${job.job_id}\n` +
        `Status: ${job.status}\n` +
        `Engine: ${job.engine}\n` +
        `Chunks: ${job.metadata.chunk_count ?? "-"}\n` +
        `Elapsed: ${job.metadata.elapsed_seconds ?? "-"}s\n` +
        `Failure: ${job.metadata.failure_reason ?? "-"}`
      );

      if (job.status === "completed") {
        clearInterval(pollTimer);
        pollTimer = null;

        if (job.text_url) {
          downloadLink.href = job.text_url;
          downloadLink.classList.remove("hidden");
        }

        // Show Q&A panel and start polling index status
        qaPanel.classList.remove("hidden");
        beginIndexPolling(jobId);
      }

      if (job.status === "failed") {
        clearInterval(pollTimer);
        pollTimer = null;
      }
    } catch (err) {
      clearInterval(pollTimer);
      pollTimer = null;
      setStatus(`Polling failed: ${err.message}`);
    }
  }, 2000);
}

// ── Index status polling ─────────────────────────────────────────────────────
function setIndexBar(status) {
  indexBar.className = "index-bar";
  if (status === "ready") {
    indexBar.classList.add("ready");
    indexIcon.textContent  = "✅";
    indexLabel.textContent = "Index ready — ask your questions below";
    askBtn.disabled = false;
  } else if (status === "failed") {
    indexBar.classList.add("failed");
    indexIcon.textContent  = "❌";
    indexLabel.textContent = "Index build failed — Q&A unavailable";
    askBtn.disabled = true;
  } else {
    indexIcon.textContent  = "⏳";
    indexLabel.textContent = "Building index…";
    askBtn.disabled = true;
  }
}

function beginIndexPolling(jobId) {
  setIndexBar("building");
  indexPollTimer = setInterval(async () => {
    try {
      const data = await fetchJson(`/transcriptions/${jobId}/index-status`);
      setIndexBar(data.status);
      if (data.status === "ready" || data.status === "failed") {
        clearInterval(indexPollTimer);
        indexPollTimer = null;
      }
    } catch (err) {
      clearInterval(indexPollTimer);
      indexPollTimer = null;
      setIndexBar("failed");
    }
  }, 2000);
}

// ── Submit transcription ─────────────────────────────────────────────────────
submitBtn.addEventListener("click", async () => {
  downloadLink.classList.add("hidden");
  qaPanel.classList.add("hidden");
  answerBox.classList.add("hidden");
  sourcesEl.classList.add("hidden");
  askBtn.disabled = true;
  stopPolling();

  const engine = engineInput.value;
  const file   = fileInput.files[0];
  const url    = urlInput.value.trim();

  if (!file && !url) { setStatus("Please upload a file or enter a URL."); return; }
  if (file && url)   { setStatus("Please provide only one input: file or URL."); return; }

  setStatus("Submitting…");

  try {
    let job;
    if (file) {
      const form = new FormData();
      form.append("file", file);
      form.append("engine", engine);
      job = await fetchJson("/transcriptions", { method: "POST", body: form });
    } else {
      job = await fetchJson("/transcriptions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ podcast_url: url, engine }),
      });
    }

    activeJobId = job.job_id;
    setStatus(`Job queued.\nJob ID: ${job.job_id}\nStatus: ${job.status}`);
    beginTranscriptionPolling(job.job_id);
  } catch (err) {
    setStatus(`Submit failed: ${err.message}`);
  }
});

// ── Ask question (streaming) ─────────────────────────────────────────────────
askBtn.addEventListener("click", async () => {
  const question = questionEl.value.trim();
  if (!question) return;
  if (!activeJobId) return;

  // Reset UI
  answerBox.textContent = "";
  answerBox.classList.remove("hidden");
  sourcesEl.innerHTML = "";
  sourcesEl.classList.add("hidden");
  askBtn.disabled = true;

  try {
    const res = await fetch(`/transcriptions/${activeJobId}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, top_k: 5 }),
    });

    if (!res.ok) {
      const body = await res.json();
      throw new Error(body.detail || "Query failed");
    }

    // Read SSE stream token by token
    const reader  = res.body.getReader();
    const decoder = new TextDecoder();
    let   buffer  = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop(); // keep incomplete last line

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const payload = line.slice(6).trim();
        if (payload === "[DONE]") break;

        try {
          const event = JSON.parse(payload);
          if (event.token) {
            answerBox.textContent += event.token;
          }
          if (event.error) {
            answerBox.textContent += `\n[Error: ${event.error}]`;
          }
        } catch { /* ignore malformed event */ }
      }
    }
  } catch (err) {
    answerBox.textContent = `Error: ${err.message}`;
  } finally {
    askBtn.disabled = false;
  }
});

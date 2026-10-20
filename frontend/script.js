const form = document.getElementById("searchForm");
const queryInput = document.getElementById("queryInput");
const methodSelect = document.getElementById("methodSelect");
const topKInput = document.getElementById("topKInput");
const searchButton = document.getElementById("searchButton");
const connectionStatus = document.getElementById("connectionStatus");
const statusDot = document.querySelector(".status-dot");
const methodNote = document.getElementById("methodNote");
const resultsTitle = document.getElementById("resultsTitle");
const resultsSubtitle = document.getElementById("resultsSubtitle");
const resultCount = document.getElementById("resultCount");
const resultsList = document.getElementById("resultsList");
const messageBox = document.getElementById("messageBox");
const loadingState = document.getElementById("loadingState");

const methodDescriptions = {
  hybrid: "Combines BM25, TF-IDF and semantic rankings using Reciprocal Rank Fusion.",
  bm25: "Ranks documents using term frequency, inverse document frequency and document-length normalization.",
  tfidf: "Ranks documents using TF-IDF term weights and cosine similarity.",
  semantic: "Ranks documents by similarity between sentence embeddings."
};

function escapeHTML(value) {
  return String(value ?? "").replace(/[&<>"']/g, char => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  })[char]);
}

function setConnected(isConnected, message) {
  connectionStatus.textContent = message;
  statusDot.classList.toggle("online", isConnected);
  statusDot.classList.toggle("offline", !isConnected);
}

async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    if (!response.ok) throw new Error("Search service did not respond.");
    const data = await response.json();
    setConnected(true, `Search engine online · ${data.document_count.toLocaleString()} documents`);
  } catch (error) {
    setConnected(false, "Search service unavailable");
  }
}

function showMessage(message) {
  messageBox.textContent = message;
  messageBox.classList.remove("hidden");
}

function clearMessage() {
  messageBox.textContent = "";
  messageBox.classList.add("hidden");
}

function setLoading(isLoading) {
  loadingState.classList.toggle("hidden", !isLoading);
  searchButton.disabled = isLoading;
  searchButton.innerHTML = isLoading
    ? "<span>Searching…</span><span>↻</span>"
    : "<span>Search collection</span><span aria-hidden=\"true\">↗</span>";
}

function renderResults(results, query, method) {
  resultsList.innerHTML = "";
  resultCount.innerHTML = `${results.length} <span>documents</span>`;
  resultsTitle.textContent = results.length ? "Ranked documents" : "No matching documents";
  resultsSubtitle.textContent = `Results for “${query}” · ${methodSelect.options[methodSelect.selectedIndex].text}`;

  if (!results.length) {
    resultsList.innerHTML = '<div class="empty-state">No documents were returned for this query.<br />Try different keywords or select another retrieval method.</div>';
    return;
  }

  resultsList.innerHTML = results.map((item, index) => {
    const title = item.title && item.title.trim() ? item.title : `Document ${item.doc_id}`;
    const snippet = item.snippet || "No document text available.";
    const meta = [
      `<span>▤ Document ID: <strong>${escapeHTML(item.doc_id)}</strong></span>`,
      item.author ? `<span>✎ ${escapeHTML(item.author)}</span>` : "",
    ].filter(Boolean).join("");
    const bib = item.bib ? `<div class="result-bib">${escapeHTML(item.bib)}</div>` : "";
    return `
      <article class="result-card">
        <div class="result-topline">
          <span class="rank-label">RANK ${String(index + 1).padStart(2, "0")}</span>
          <span class="score-pill">Score&nbsp; ${Number(item.score).toFixed(5)}</span>
        </div>
        <h3>${escapeHTML(title)}</h3>
        <div class="doc-meta">${meta}</div>
        <p class="result-snippet">${escapeHTML(snippet)}</p>
        ${bib}
      </article>`;
  }).join("");
}

async function runSearch(query) {
  clearMessage();
  resultsList.innerHTML = "";
  setLoading(true);
  resultsTitle.textContent = "Searching…";
  resultsSubtitle.textContent = "Running your query against the Cranfield collection.";
  resultCount.innerHTML = "— <span>documents</span>";

  try {
    const response = await fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        method: methodSelect.value,
        top_k: Number(topKInput.value)
      })
    });

    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Search failed.");
    renderResults(data.results || [], query, data.method || methodSelect.value);
  } catch (error) {
    resultsTitle.textContent = "Search could not be completed";
    resultsSubtitle.textContent = "Check that the Python search server is running.";
    resultCount.innerHTML = "— <span>documents</span>";
    showMessage(error.message || "Unexpected error while searching.");
  } finally {
    setLoading(false);
  }
}

form.addEventListener("submit", event => {
  event.preventDefault();
  const query = queryInput.value.trim();
  if (!query) {
    showMessage("Please enter a search query first.");
    queryInput.focus();
    return;
  }
  runSearch(query);
});

methodSelect.addEventListener("change", () => {
  methodNote.textContent = methodDescriptions[methodSelect.value] || "";
});

document.querySelectorAll(".suggestion").forEach(button => {
  button.addEventListener("click", () => {
    queryInput.value = button.dataset.query || "";
    queryInput.focus();
    runSearch(queryInput.value.trim());
  });
});

checkHealth();

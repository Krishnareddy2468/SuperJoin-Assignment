"use strict";

const API = "";
const state = {
  view: "documents",
  documents: [], facts: [], relations: [], failures: [],
  selectedFiles: [], query: "", filter: "all", loading: false,
};
let refreshTimer = null;
let lastFocusedElement = null;

const $ = (selector) => document.querySelector(selector);
const elements = {
  system: $("#system-state"), form: $("#upload-form"), input: $("#file-input"),
  drop: $("#drop-zone"), queue: $("#upload-queue"), upload: $("#upload-button"),
  panel: $("#result-panel"), notice: $("#notice"), search: $("#search-input"),
  filter: $("#filter-select"), refresh: $("#refresh-button"), drawer: $("#detail-drawer"),
  drawerPanel: $(".drawer-panel"), drawerTitle: $("#drawer-title"),
  drawerKicker: $("#drawer-kicker"), drawerContent: $("#drawer-content"),
};

const escapeHtml = (value = "") => String(value).replace(/[&<>'"]/g, (character) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
}[character]));
const humanize = (value = "") => String(value).replaceAll("_", " ").replace(/\b\w/g, (c) => c.toUpperCase());
const list = (payload) => Array.isArray(payload) ? payload : (payload?.items || payload?.results || []);
const idOf = (item) => item?.id ?? item?.document_id ?? item?.fact_id ?? item?.relation_id;
const statusOf = (item) => String(item?.status || item?.relation_type || item?.type || "unknown").toLowerCase();
// Domain objects name themselves differently: an entity has canonical_name, a predicate
// has display_name and key, a value has raw. Missing the first two dropped straight
// through to JSON.stringify, so every row in the UI was titled with a raw object.
const textOf = (value) => typeof value === "object" && value !== null
  ? value.display_name || value.canonical_name || value.name || value.label || value.raw
    || value.key || value.display || value.text || value.state || JSON.stringify(value)
  : (value ?? "Not provided");
const number = (value) => Number.isFinite(Number(value)) ? Number(value).toLocaleString() : "—";
const fileSize = (bytes) => bytes < 1e6 ? `${(bytes / 1e3).toFixed(0)} KB` : `${(bytes / 1e6).toFixed(1)} MB`;

async function request(path, options = {}) {
  const { headers = {}, ...requestOptions } = options;
  const response = await fetch(`${API}${path}`, { ...requestOptions, headers: { Accept: "application/json", ...headers } });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try { const body = await response.json(); message = body.detail || body.message || message; } catch (_) { /* use status */ }
    throw new Error(message);
  }
  if (response.status === 204) return null;
  return response.json();
}

function showNotice(message, tone = "warning") {
  elements.notice.textContent = message;
  elements.notice.classList.remove("hidden");
  elements.notice.dataset.tone = tone;
}
function hideNotice() { elements.notice.classList.add("hidden"); }
function setSystem(online, detail) {
  elements.system.className = `system-state ${online ? "online" : "offline"}`;
  elements.system.lastElementChild.textContent = detail;
}
function loading() {
  elements.panel.replaceChildren($("#loading-template").content.cloneNode(true));
}
function empty(title, message) {
  elements.panel.innerHTML = `<div class="empty-state"><strong>${escapeHtml(title)}</strong><p>${escapeHtml(message)}</p></div>`;
}

async function boot() {
  bindEvents();
  loading();
  try {
    const health = await request("/health");
    // The connection dot already says whether the browser reached the API; putting the word
    // "Online" next to "offline mode" in the same label read as a contradiction. State the two
    // things separately, and say the mode either way instead of only when the LLM is off.
    const mode = health?.extraction_mode === "hybrid"
      ? "System ready · hybrid mode (Gemini enabled)"
      : "System ready · offline mode (no LLM)";
    setSystem(true, mode);
    if (health?.llm_enabled === false) showNotice("No LLM is configured. Deterministic extraction remains available.");
    await loadAll();
  } catch (error) {
    setSystem(false, "API unavailable");
    showNotice(`The interface is ready, but the FactLayer API is unavailable: ${error.message}`);
    empty("Waiting for the API", "Start the FastAPI service, then use Refresh to load documents and results.");
    updateSummary();
  }
}

// A relation type the dropdown offers, mapped to the concrete types the API stores.
// "reconciled" is a family, not a single type, so it has to fan out into its members.
const RELATION_FILTERS = {
  corroborates: ["corroborates"],
  contradicts: ["contradicts", "likely_contradiction"],
  reconciled: ["reconciled_by_period", "reconciled_by_scope", "reconciled_by_basis", "reconciled_by_as_of", "reconciled_by_unit"],
  incomparable: ["incomparable"],
  needs_review: ["needs_review"],
};

function relationsUrl() {
  // Ask the server for the type being viewed. Filtering client-side over one page meant the
  // interesting verdicts were unreachable: relations come back oldest-first, the first page
  // was entirely needs_review, and so corroborates, contradicts and reconciled all rendered
  // as "no relations found" while thousands of them sat in the database.
  const types = state.view === "relations" ? RELATION_FILTERS[state.filter] : null;
  if (!types) return "/relations?limit=500";
  return types.map((type) => `/relations?limit=500&relation_type=${encodeURIComponent(type)}`);
}

async function loadRelations() {
  const target = relationsUrl();
  if (!Array.isArray(target)) return list(await request(target));
  const settled = await Promise.allSettled(target.map((url) => request(url)));
  return settled.flatMap((result) => result.status === "fulfilled" ? list(result.value) : []);
}

function interleave(groups) {
  const merged = [];
  const longest = Math.max(0, ...groups.map((group) => group.length));
  for (let index = 0; index < longest; index += 1) {
    for (const group of groups) if (index < group.length) merged.push(group[index]);
  }
  return merged;
}

async function loadAll() {
  state.loading = true;
  loading();
  const results = await Promise.allSettled([
    request("/documents"), loadRelations(), request("/failures?limit=200"),
  ]);
  state.documents = results[0].status === "fulfilled" ? list(results[0].value) : [];
  state.relations = results[1].status === "fulfilled" ? list(results[1].value) : [];
  state.failures = results[2].status === "fulfilled" ? list(results[2].value) : [];
  const factResults = await Promise.allSettled(state.documents.map((doc) => request(`/documents/${idOf(doc)}/facts?limit=500`)));
  // Round-robin the documents rather than concatenating them. Concatenating put every fact
  // from the first-ingested file ahead of the others, so the top of a long list looked like
  // only one document had been processed and you had to scroll hundreds of rows to see the rest.
  state.facts = interleave(factResults.map((result) => result.status === "fulfilled" ? list(result.value) : []));
  state.loading = false;
  updateSummary();
  configureFilter();
  render();
  const failures = results.filter((result) => result.status === "rejected").length + factResults.filter((result) => result.status === "rejected").length;
  if (failures) showNotice(`${failures} result request${failures === 1 ? "" : "s"} could not be loaded. Available data is shown.`);
  else hideNotice();
  scheduleProcessingRefresh();
}

function scheduleProcessingRefresh() {
  window.clearTimeout(refreshTimer);
  const active = state.documents.some((doc) => ["queued", "processing"].includes(statusOf(doc)));
  if (active) refreshTimer = window.setTimeout(() => loadAll().catch(() => setSystem(false, "API unavailable")), 3000);
}

function updateSummary() {
  $("#document-count").textContent = number(state.documents.length);
  const factTotal = state.documents.reduce((sum, doc) => sum + Number(doc.fact_count || 0), 0) || state.facts.length;
  const relationTotal = state.documents.reduce((sum, doc) => sum + Number(doc.relation_count || 0), 0) || state.relations.length;
  const reviewTotal = state.failures.length + state.facts.filter((fact) => statusOf(fact).includes("review")).length
    + state.relations.filter((relation) => ["needs_review", "incomparable", "likely_contradiction"].includes(statusOf(relation))).length;
  $("#fact-count").textContent = number(factTotal);
  $("#relation-count").textContent = number(relationTotal);
  $("#review-count").textContent = number(reviewTotal);
}

const viewConfig = {
  documents: { placeholder: "Search documents…", filter: ["all", "complete", "processing", "partial", "failed"] },
  facts: { placeholder: "Search subjects, predicates, or values…", filter: ["all", "high", "medium", "low", "needs_review"] },
  relations: { placeholder: "Search relationships…", filter: ["all", "corroborates", "contradicts", "reconciled", "incomparable", "needs_review"] },
  failures: { placeholder: "Search failure reasons or stages…", filter: ["all", "document", "page", "extraction", "grounding", "normalization", "linking"] },
};

function configureFilter() {
  const config = viewConfig[state.view];
  elements.search.placeholder = config.placeholder;
  elements.filter.innerHTML = config.filter.map((value) => `<option value="${value}">${value === "all" ? "All" : humanize(value)}</option>`).join("");
  state.filter = "all";
}

function searchable(item) { return JSON.stringify(item).toLowerCase(); }
function filtered(items) {
  return items.filter((item) => {
    const queryMatch = !state.query || searchable(item).includes(state.query);
    if (state.filter === "all") return queryMatch;
    // The server already narrowed relations to the chosen type. Re-checking here would
    // drop valid rows, because a stored type like "reconciled_by_scope" does not contain
    // the dropdown's family name "reconciled" in the way this substring test expects.
    if (state.view === "relations") return queryMatch;
    if (state.view === "facts" && ["high", "medium", "low"].includes(state.filter)) {
      const confidence = Number(item.confidence ?? 0);
      const band = confidence >= .8 ? "high" : confidence >= .5 ? "medium" : "low";
      return queryMatch && band === state.filter;
    }
    return queryMatch && searchable(item).includes(state.filter);
  });
}

function render() {
  if (state.loading) return loading();
  const items = filtered(state[state.view]);
  if (!items.length) return empty(`No ${state.view} found`, state.query || state.filter !== "all"
    ? "Try clearing the search or changing the filter." : `Upload and process PDFs to populate ${state.view}.`);
  elements.panel.innerHTML = items.map((item) => row(item, state.view)).join("");
  elements.panel.querySelectorAll("[data-item-id]").forEach((button) => button.addEventListener("click", () => openDetails(button.dataset.type, button.dataset.itemId)));
}

function badge(status) {
  const normalized = String(status || "unknown").toLowerCase();
  let style = normalized;
  if (normalized.startsWith("reconciled")) style = "reconciled";
  if (normalized.includes("review") || normalized.includes("likely")) style = "review";
  return `<span class="badge ${escapeHtml(style)}">${escapeHtml(humanize(normalized))}</span>`;
}

function row(item, type) {
  const id = idOf(item);
  if (type === "documents") {
    const title = item.original_filename || item.filename || item.name || `Document ${id}`;
    return resultRow(type, id, title, `${number(item.page_count)} pages · ${escapeHtml(item.extraction_mode || "mode unknown")}`, `${number(item.fact_count)} facts`, badge(statusOf(item)));
  }
  if (type === "facts") {
    const subject = textOf(item.subject || item.entity);
    const predicate = textOf(item.predicate);
    const value = textOf(item.normalized_value ?? item.value?.normalized ?? item.value ?? item.raw_value);
    return resultRow(type, id, `${subject} · ${predicate}`, value, confidenceLabel(item.confidence), badge(item.review_state || "grounded"));
  }
  if (type === "relations") {
    const relation = item.relation_type || item.type || "needs_review";
    const title = item.title || `${textOf(item.fact_a?.subject || "Fact A")} ↔ ${textOf(item.fact_b?.subject || "Fact B")}`;
    return resultRow(type, id, title, item.explanation || humanize(relation), confidenceLabel(item.confidence), badge(relation));
  }
  const title = `${humanize(item.stage || item.type || "Processing")} failure`;
  return resultRow(type, id, title, item.reason || item.message || "No reason supplied", item.page_index != null ? `PDF page ${Number(item.page_index) + 1}` : "Document level", badge(item.recoverable ? "review" : "failed"));
}

function resultRow(type, id, title, subtitle, meta, status) {
  return `<button class="result-row" type="button" data-type="${type}" data-item-id="${escapeHtml(id)}">
    <span class="row-title"><strong>${escapeHtml(title)}</strong><span>${escapeHtml(subtitle)}</span></span>
    <span class="row-meta">${escapeHtml(meta)}</span>${status}<span class="row-arrow" aria-hidden="true">›</span>
  </button>`;
}
function confidenceLabel(value) { return value == null ? "Confidence unknown" : `${Math.round(Number(value) * 100)}% confidence`; }

async function openDetails(type, id) {
  const collection = state[type] || [];
  let item = collection.find((candidate) => String(idOf(candidate)) === String(id));
  openDrawer("Loading…", humanize(type), `<div class="loading-state"><span class="spinner"></span></div>`);
  try {
    if (type === "documents") item = await request(`/documents/${id}`);
    if (type === "facts") item = await request(`/facts/${id}`);
    if (type === "relations") item = await request(`/relations/${id}`);
    renderDetails(type, item || {});
  } catch (error) {
    renderDetails(type, item || {}, error.message);
  }
}

function openDrawer(title, kicker, content) {
  if (!elements.drawer.classList.contains("open")) lastFocusedElement = document.activeElement;
  elements.drawerTitle.textContent = title;
  elements.drawerKicker.textContent = kicker;
  elements.drawerContent.innerHTML = content;
  elements.drawer.classList.add("open");
  elements.drawer.setAttribute("aria-hidden", "false");
  document.body.style.overflow = "hidden";
  elements.drawerPanel.focus();
}
function closeDrawer() {
  elements.drawer.classList.remove("open");
  elements.drawer.setAttribute("aria-hidden", "true");
  document.body.style.overflow = "";
  if (lastFocusedElement instanceof HTMLElement) lastFocusedElement.focus();
}

function renderDetails(type, item, warning = "") {
  if (type === "documents") return documentDetails(item, warning);
  if (type === "facts") return factDetails(item, warning);
  if (type === "relations") return relationDetails(item, warning);
  return failureDetails(item, warning);
}
function warningHtml(message) { return message ? `<div class="notice">Some details could not be loaded: ${escapeHtml(message)}</div>` : ""; }
function fields(entries) {
  return `<div class="detail-grid">${entries.map(([label, value]) => `<div class="detail-field"><small>${escapeHtml(label)}</small><strong>${escapeHtml(textOf(value))}</strong></div>`).join("")}</div>`;
}
function documentDetails(doc, warning) {
  const title = doc.original_filename || doc.filename || doc.name || "Document";
  openDrawer(title, "Document", `${warningHtml(warning)}<section class="detail-section"><h3>Processing summary</h3>${fields([
    ["Status", humanize(statusOf(doc))], ["PDF pages", doc.page_count], ["Grounded facts", doc.fact_count],
    ["Relationships", doc.relation_count], ["Extraction mode", doc.extraction_mode || "Unknown"], ["Document ID", idOf(doc)],
  ])}</section>${warnings(doc.warnings)}`);
}
function factDetails(fact, warning) {
  const title = `${textOf(fact.subject || fact.entity)} · ${textOf(fact.predicate)}`;
  const evidence = list(fact.evidence || fact.evidences);
  openDrawer(title, "Grounded fact", `${warningHtml(warning)}<section class="detail-section"><h3>Fact</h3>${fields([
    ["Raw value", fact.raw_value ?? fact.value?.raw ?? fact.value], ["Normalized value", fact.normalized_value ?? fact.value?.normalized],
    ["Unit", fact.unit ?? fact.value?.unit], ["Confidence", confidenceLabel(fact.confidence)],
  ])}</section><section class="detail-section"><h3>Context envelope</h3>${fields(contextEntries(fact.context))}</section>
    <section class="detail-section" data-evidence><h3>Source evidence</h3>${evidenceCards(evidence)}</section>${warnings(fact.warnings)}`);
  if (!evidence.length) loadEvidence(idOf(fact));
}
async function loadEvidence(factId) {
  try {
    const payload = await request(`/facts/${factId}/evidence`);
    const container = elements.drawerContent.querySelector("[data-evidence]");
    if (container) container.innerHTML = `<h3>Source evidence</h3>${evidenceCards(list(payload))}`;
  } catch (_) { /* the visible empty evidence state remains */ }
}
function relationDetails(relation, warning) {
  const type = relation.relation_type || relation.type || "needs_review";
  const evidence = [...list(relation.fact_a?.evidence), ...list(relation.fact_b?.evidence), ...list(relation.evidence)];
  openDrawer(humanize(type), "Fact relationship", `${warningHtml(warning)}<section class="detail-section">${badge(type)}</section>
    <section class="detail-section"><h3>Explanation</h3><div class="reason-box">${escapeHtml(relation.explanation || "No explanation was supplied.")}</div></section>
    <section class="detail-section"><h3>Compared values</h3>${fields([
      ["Fact A", textOf(relation.fact_a?.normalized_value ?? relation.fact_a?.value ?? relation.fact_a_id)],
      ["Fact B", textOf(relation.fact_b?.normalized_value ?? relation.fact_b?.value ?? relation.fact_b_id)],
      ["Confidence", confidenceLabel(relation.confidence)], ["Rule version", relation.rule_version],
    ])}</section><section class="detail-section"><h3>Context differences</h3>${fields(contextDiffEntries(relation.context_diff))}</section>
    <section class="detail-section"><h3>Evidence side by side</h3>${evidenceCards(evidence)}</section>`);
}
function failureDetails(failure, warning) {
  openDrawer(humanize(failure.stage || "Failure"), "Visible pipeline failure", `${warningHtml(warning)}
    <section class="detail-section"><h3>What happened</h3><div class="reason-box">${escapeHtml(failure.reason || failure.message || "No reason supplied")}</div></section>
    <section class="detail-section"><h3>Location and handling</h3>${fields([
      ["Document", failure.document_name || failure.document_id], ["PDF page", failure.page_index != null ? Number(failure.page_index) + 1 : "Document level"],
      ["Stage", humanize(failure.stage)], ["Recoverable", failure.recoverable == null ? "Unknown" : failure.recoverable ? "Yes" : "No"],
    ])}</section>${failure.rejected_output ? `<section class="detail-section"><h3>Rejected output</h3><pre>${escapeHtml(failure.rejected_output)}</pre></section>` : ""}`);
}
function contextEntries(context = {}) {
  const entries = Object.entries(context || {}).filter(([, value]) => value != null && value !== "").map(([key, value]) => [humanize(key), textOf(value)]);
  return entries.length ? entries : [["Context", "Not provided"]];
}
// context_diff is a list of differences, not a context object. Running it through
// contextEntries() labelled each row with its array index and printed the raw object,
// which is the least useful rendering of the most interesting part of a relationship.
function contextDiffEntries(differences) {
  if (!Array.isArray(differences) || !differences.length) return [["Context", "No differences"]];
  return differences.map((item) => {
    const left = contextSide(item.left);
    const right = contextSide(item.right);
    const note = item.explains_difference ? " — explains the difference" : "";
    return [humanize(item.field || "field"), `${left} vs ${right}${note}`];
  });
}
function contextSide(value) {
  if (value == null || value === "") return "not stated";
  if (typeof value !== "object") return String(value);
  if (value.start || value.end) return `${value.start || "?"} to ${value.end || "?"}`;
  const parts = Object.values(value).filter((part) => part != null && part !== "");
  return parts.length ? parts.join(" ") : "not stated";
}
function evidenceCards(evidence) {
  if (!evidence.length) return `<div class="empty-state"><strong>No evidence loaded</strong><p>This fact should not persist without verified evidence.</p></div>`;
  return `<div class="evidence-grid">${evidence.map((item) => {
    const page = item.page_index != null ? Number(item.page_index) + 1 : item.pdf_page || "?";
    const docId = item.document_id || item.document?.id;
    const link = docId ? `/documents/${encodeURIComponent(docId)}/file#page=${page}` : "";
    return `<article class="evidence-card"><span class="badge complete">Verified quote</span><blockquote>“${escapeHtml(item.quote || item.text || "Quote unavailable")}”</blockquote>
      <footer>${escapeHtml(item.document_name || item.filename || "Source document")} · PDF page ${escapeHtml(page)}${item.bbox ? " · location captured" : ""}
      ${link ? ` · <a href="${link}" target="_blank" rel="noopener">Open PDF</a>` : ""}</footer></article>`;
  }).join("")}</div>`;
}
function warnings(items) {
  const values = list(items);
  return values.length ? `<section class="detail-section"><h3>Warnings</h3><ul class="warning-list">${values.map((item) => `<li>${escapeHtml(textOf(item))}</li>`).join("")}</ul></section>` : "";
}

function selectFiles(files) {
  const accepted = [...files].filter((file) => file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf"));
  const rejected = files.length - accepted.length;
  state.selectedFiles = accepted;
  elements.queue.innerHTML = accepted.map((file) => `<div class="queue-item"><span>${escapeHtml(file.name)}</span><small>${fileSize(file.size)}</small></div>`).join("");
  elements.upload.disabled = !accepted.length;
  if (rejected) showNotice(`${rejected} non-PDF file${rejected === 1 ? " was" : "s were"} ignored.`);
}
async function uploadFiles(event) {
  event.preventDefault();
  if (!state.selectedFiles.length) return;
  elements.upload.disabled = true;
  elements.upload.textContent = "Uploading…";
  try {
    for (const file of state.selectedFiles) {
      const body = new FormData(); body.append("file", file);
      await request("/documents", { method: "POST", body, headers: {} });
    }
    state.selectedFiles = []; elements.input.value = ""; elements.queue.innerHTML = "";
    showNotice("Upload accepted. Processing status will update as facts become available.", "success");
    await loadAll();
  } catch (error) { showNotice(`Upload failed: ${error.message}`); }
  finally { elements.upload.textContent = "Process selected PDFs"; elements.upload.disabled = !state.selectedFiles.length; }
}

function bindEvents() {
  elements.input.addEventListener("change", () => selectFiles(elements.input.files));
  elements.form.addEventListener("submit", uploadFiles);
  ["dragenter", "dragover"].forEach((name) => elements.drop.addEventListener(name, (event) => { event.preventDefault(); elements.drop.classList.add("dragging"); }));
  ["dragleave", "drop"].forEach((name) => elements.drop.addEventListener(name, (event) => { event.preventDefault(); elements.drop.classList.remove("dragging"); }));
  elements.drop.addEventListener("drop", (event) => selectFiles(event.dataTransfer.files));
  document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((item) => { item.classList.toggle("active", item === tab); item.setAttribute("aria-selected", item === tab); });
    state.view = tab.dataset.view; state.query = ""; elements.search.value = ""; configureFilter(); render(); if (state.view === "relations") loadRelations().then((items) => { state.relations = items; render(); }).catch(() => {});
  }));
  elements.search.addEventListener("input", () => { state.query = elements.search.value.trim().toLowerCase(); render(); });
  elements.filter.addEventListener("change", async () => {
    state.filter = elements.filter.value;
    // Relations are filtered by the server now, so changing the type needs a refetch.
    // Every other view already holds everything it needs and only has to re-render.
    if (state.view !== "relations") return render();
    state.loading = true;
    loading();
    try {
      state.relations = await loadRelations();
    } catch (error) {
      state.relations = [];
      showNotice(error.message);
    }
    state.loading = false;
    render();
  });
  elements.refresh.addEventListener("click", async () => { try { await loadAll(); setSystem(true, "System ready"); } catch (error) { setSystem(false, "API unavailable"); showNotice(error.message); render(); } });
  $("#drawer-close").addEventListener("click", closeDrawer);
  $("#drawer-backdrop").addEventListener("click", closeDrawer);
  document.addEventListener("keydown", (event) => { if (event.key === "Escape") closeDrawer(); });
}

document.addEventListener("DOMContentLoaded", boot);

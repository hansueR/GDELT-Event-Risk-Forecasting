const FILES = {
  series: "data/dashboard_series.json",
  summary: "data/summary_metrics.json",
  alert: "data/alert_metrics.json",
  metadata: "data/metadata.json",
  latest: "data/latest_predictions.json",
  history: "data/prediction_history.json",
  modelMetadata: "data/model_metadata.json",
};

const state = {
  series: [],
  summary: [],
  alert: [],
  latest: [],
  history: [],
  metadata: {},
  modelMetadata: {},
};

const els = {
  asset: document.getElementById("assetSelect"),
  horizon: document.getElementById("horizonSelect"),
  method: document.getElementById("methodSelect"),
  start: document.getElementById("startDate"),
  end: document.getElementById("endDate"),
  status: document.getElementById("status"),
  message: document.getElementById("message"),
};

async function fetchJson(path, fallback) {
  try {
    const response = await fetch(path, { cache: "no-store" });
    if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
    return await response.json();
  } catch (error) {
    return { ...fallback, warnings: [`Could not load ${path}. ${error.message}`] };
  }
}

function asArray(payload, key) {
  if (Array.isArray(payload)) return payload;
  return Array.isArray(payload?.[key]) ? payload[key] : [];
}

function unique(values) {
  return [...new Set(values.filter((value) => value !== null && value !== undefined && value !== ""))];
}

function fmt(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "";
  return Number(value).toFixed(digits);
}

function methodRows(source) {
  const start = els.start.value ? new Date(els.start.value) : null;
  const end = els.end.value ? new Date(els.end.value) : null;
  return source
    .filter((row) => row.asset === els.asset.value && row.horizon === els.horizon.value && row.method === els.method.value)
    .filter((row) => {
      const date = new Date(row.prediction_date || row.date);
      return (!start || date >= start) && (!end || date <= end);
    })
    .sort((a, b) => new Date(a.prediction_date || a.date) - new Date(b.prediction_date || b.date));
}

function selectedSeriesRows() {
  const start = els.start.value ? new Date(els.start.value) : null;
  const end = els.end.value ? new Date(els.end.value) : null;
  return state.series
    .filter((row) => row.asset === els.asset.value && row.horizon === els.horizon.value)
    .filter((row) => {
      const date = new Date(row.date);
      return (!start || date >= start) && (!end || date <= end);
    })
    .sort((a, b) => new Date(a.date) - new Date(b.date));
}

function getMethods() {
  return unique(
    state.latest.map((row) => row.method)
      .concat(state.history.map((row) => row.method))
      .concat(state.summary.map((row) => row.method)),
  ).sort();
}

function setOptions(select, values, emptyLabel) {
  select.disabled = values.length === 0;
  select.innerHTML = values.length
    ? values.map((value) => `<option value="${value}">${value}</option>`).join("")
    : `<option value="">${emptyLabel}</option>`;
}

function populateControls() {
  const assets = unique(state.latest.map((row) => row.asset).concat(state.history.map((row) => row.asset), state.series.map((row) => row.asset)));
  setOptions(els.asset, assets, "No asset data");

  const horizons = unique(state.latest.map((row) => row.horizon).concat(state.history.map((row) => row.horizon), state.series.map((row) => row.horizon)));
  setOptions(els.horizon, horizons, "No horizon data");

  const methods = getMethods();
  setOptions(els.method, methods, "No model data");

  const dates = unique(
    state.history.map((row) => row.prediction_date)
      .concat(state.latest.map((row) => row.prediction_date), state.series.map((row) => row.date)),
  ).sort();
  if (dates.length) {
    els.start.value = dates[0];
    els.end.value = dates[dates.length - 1];
  }
}

function latestSelection() {
  const rows = state.latest
    .filter((row) => row.asset === els.asset.value && row.horizon === els.horizon.value && row.method === els.method.value)
    .sort((a, b) => new Date(b.prediction_date) - new Date(a.prediction_date));
  return rows[0] || null;
}

function renderLatestCard() {
  const row = latestSelection();
  const fields = row
    ? [
        ["Asset", row.asset],
        ["Horizon", row.horizon],
        ["Method", row.method],
        ["Predicted log RV", fmt(row.predicted_log_rv, 4)],
        ["Alert score", fmt(row.alert_score, 3)],
        ["Risk level", row.risk_level || "unknown"],
        ["Generated at", row.generated_at || ""],
        ["Training end", row.training_end_date || state.modelMetadata.training_end_date || ""],
      ]
    : [["Status", "No latest prediction"]];
  document.getElementById("latestCard").innerHTML = fields
    .map(([label, value]) => `<div class="metricBox"><span>${label}</span><strong>${value ?? ""}</strong></div>`)
    .join("");
}

function renderPriceEvent(rows) {
  Plotly.react("priceEventChart", [
    {
      x: rows.map((row) => row.date),
      y: rows.map((row) => row.close),
      name: "Close",
      type: "scatter",
      mode: "lines",
      line: { color: "#0f766e", width: 2 },
    },
    {
      x: rows.map((row) => row.date),
      y: rows.map((row) => row.event_intensity),
      name: "Event intensity",
      type: "bar",
      yaxis: "y2",
      marker: { color: "rgba(180, 83, 9, 0.35)" },
    },
  ], {
    margin: { l: 52, r: 54, t: 12, b: 42 },
    legend: { orientation: "h" },
    xaxis: { title: "Date" },
    yaxis: { title: "Close" },
    yaxis2: { title: "Event", overlaying: "y", side: "right", rangemode: "tozero" },
  }, { responsive: true, displayModeBar: false });
}

function renderForecast(rows) {
  Plotly.react("forecastChart", [
    {
      x: rows.map((row) => row.prediction_date),
      y: rows.map((row) => row.predicted_log_rv),
      name: "Predicted log RV",
      type: "scatter",
      mode: "lines+markers",
      line: { color: "#b45309", width: 2 },
    },
    {
      x: rows.map((row) => row.prediction_date),
      y: rows.map((row) => row.actual_log_rv),
      name: "Actual log RV",
      type: "scatter",
      mode: "lines+markers",
      line: { color: "#243b53", width: 2 },
    },
  ], {
    margin: { l: 52, r: 20, t: 12, b: 42 },
    legend: { orientation: "h" },
    xaxis: { title: "Prediction date" },
    yaxis: { title: "Log realized volatility" },
  }, { responsive: true, displayModeBar: false });
}

function renderAlertScore(rows) {
  const scoredRows = rows.filter((row) => row.alert_score !== null && row.alert_score !== undefined && !Number.isNaN(Number(row.alert_score)));
  Plotly.react("alertScoreChart", [{
    x: scoredRows.map((row) => row.prediction_date),
    y: scoredRows.map((row) => row.alert_score),
    name: "Alert score",
    type: "scatter",
    mode: "lines+markers",
    line: { color: "#b42318", width: 2 },
  }], {
    margin: { l: 52, r: 20, t: 12, b: 42 },
    yaxis: { title: "Score", range: [0, 1] },
    xaxis: { title: "Prediction date" },
  }, { responsive: true, displayModeBar: false });
}

function renderAlertMetrics() {
  const rows = state.alert.filter((row) => row.asset === els.asset.value && row.horizon === els.horizon.value);
  Plotly.react("alertMetricsChart", [
    { x: rows.map((row) => row.method), y: rows.map((row) => row.auc), name: "AUC", type: "bar" },
    { x: rows.map((row) => row.method), y: rows.map((row) => row.ap), name: "AP", type: "bar" },
    { x: rows.map((row) => row.method), y: rows.map((row) => row.p_at_10), name: "P@10", type: "bar" },
    { x: rows.map((row) => row.method), y: rows.map((row) => row.r_at_10), name: "R@10", type: "bar" },
  ], {
    barmode: "group",
    margin: { l: 44, r: 12, t: 12, b: 80 },
    legend: { orientation: "h" },
    yaxis: { rangemode: "tozero" },
  }, { responsive: true, displayModeBar: false });
}

function renderLatestTable() {
  const columns = ["prediction_date", "asset", "horizon", "method", "predicted_log_rv", "alert_score", "risk_level", "status"];
  const rows = state.latest;
  document.querySelector("#latestTable thead").innerHTML = `<tr>${columns.map((col) => `<th>${col}</th>`).join("")}</tr>`;
  document.querySelector("#latestTable tbody").innerHTML = rows
    .map((row) => `<tr>${columns.map((col) => `<td>${["predicted_log_rv", "alert_score"].includes(col) ? fmt(row[col], 4) : row[col] ?? ""}</td>`).join("")}</tr>`)
    .join("");
}

function renderSummaryTable() {
  const columns = ["method", "mae", "rmse", "r2", "qlike", "auc", "ap", "p_at_10", "r_at_10"];
  const rows = state.summary.filter((row) => row.asset === els.asset.value && row.horizon === els.horizon.value);
  document.querySelector("#summaryTable thead").innerHTML = `<tr>${columns.map((col) => `<th>${col}</th>`).join("")}</tr>`;
  document.querySelector("#summaryTable tbody").innerHTML = rows
    .map((row) => `<tr>${columns.map((col) => `<td>${col === "method" ? row[col] || "" : fmt(row[col])}</td>`).join("")}</tr>`)
    .join("");
}

function renderMessage(warnings) {
  const messages = [...warnings];
  if (!state.series.length && !state.latest.length && !state.history.length) messages.push("No real dashboard data is available.");
  if (!methodRows(state.history).length && !latestSelection()) messages.push("No historical prediction rows match the selected filters.");
  els.message.hidden = messages.length === 0;
  els.message.textContent = messages.join(" ");
}

function render(warnings = []) {
  const historyRows = methodRows(state.history);
  renderLatestCard();
  renderPriceEvent(selectedSeriesRows());
  renderForecast(historyRows);
  renderAlertScore(historyRows);
  renderAlertMetrics();
  renderLatestTable();
  renderSummaryTable();
  renderMessage(warnings);
}

async function init() {
  const [seriesPayload, summaryPayload, alertPayload, metadataPayload, latestPayload, historyPayload, modelPayload] = await Promise.all([
    fetchJson(FILES.series, { series: [] }),
    fetchJson(FILES.summary, { metrics: [] }),
    fetchJson(FILES.alert, { metrics: [] }),
    fetchJson(FILES.metadata, {}),
    fetchJson(FILES.latest, { predictions: [] }),
    fetchJson(FILES.history, { predictions: [] }),
    fetchJson(FILES.modelMetadata, {}),
  ]);

  state.series = asArray(seriesPayload, "series");
  state.summary = asArray(summaryPayload, "metrics");
  state.alert = asArray(alertPayload, "metrics");
  state.latest = asArray(latestPayload, "predictions");
  state.history = asArray(historyPayload, "predictions");
  state.metadata = metadataPayload;
  state.modelMetadata = modelPayload;

  const warnings = [
    ...(seriesPayload.warnings || []),
    ...(summaryPayload.warnings || []),
    ...(alertPayload.warnings || []),
    ...(metadataPayload.warnings || []),
    ...(latestPayload.warnings || []),
    ...(historyPayload.warnings || []),
    ...(modelPayload.warnings || []),
  ];

  populateControls();
  [els.asset, els.horizon, els.method, els.start, els.end].forEach((el) => el.addEventListener("change", () => render(warnings)));
  els.status.textContent = state.modelMetadata.training_end_date
    ? `Online model trained through ${state.modelMetadata.training_end_date}`
    : "Static historical dashboard";
  render(warnings);
}

init();

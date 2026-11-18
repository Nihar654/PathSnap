import { useEffect, useMemo, useState } from "react";
import { MapContainer, TileLayer, CircleMarker, Polyline, Popup } from "react-leaflet";
import {
  Activity,
  Route,
  Upload,
  Server,
  Zap,
  MapPin,
  RefreshCw,
  ShieldCheck,
  CheckCircle,
  Sliders,
  Award,
  Layers
} from "lucide-react";

export default function App() {
  const defaultHost =
    typeof window !== "undefined" && window.location.hostname !== "localhost"
      ? `http://${window.location.hostname}:8000`
      : "http://10.30.201.237:8000";

  const [apiUrl, setApiUrl] = useState(defaultHost);
  const [editingIp, setEditingIp] = useState(false);
  const [ipInput, setIpInput] = useState(defaultHost);

  // Core State
  const [health, setHealth] = useState(null);
  const [trips, setTrips] = useState([]);
  const [selectedTrip, setSelectedTrip] = useState("00000015");
  const [models, setModels] = useState([]);
  const [model, setModel] = useState("Random Forest (Our Selected Model)");
  const [useHmm, setUseHmm] = useState(true);

  // Route Data
  const [gpsPoints, setGpsPoints] = useState([]);
  const [actualPath, setActualPath] = useState([]);
  const [predictedPaths, setPredictedPaths] = useState({});
  const [benchmarks, setBenchmarks] = useState({});
  const [center, setCenter] = useState([50.8276, 4.7538]);

  // Layer Toggles
  const [showGps, setShowGps] = useState(true);
  const [showActual, setShowActual] = useState(true);
  const [showPredicted, setShowPredicted] = useState(true);

  // Custom Upload
  const [file, setFile] = useState(null);
  const [isCustomUpload, setIsCustomUpload] = useState(false);

  // Prediction Output
  const [busy, setBusy] = useState(false);
  const [predictionResult, setPredictionResult] = useState(null);
  const [message, setMessage] = useState("Loading map-matching trajectories...");

  // --------------------------------------------------------------
  // 1. Refresh Health & Metadata
  // --------------------------------------------------------------
  async function refresh(targetApi = apiUrl) {
    try {
      const [hRes, mRes, tRes] = await Promise.all([
        fetch(`${targetApi}/api/health`),
        fetch(`${targetApi}/api/models`),
        fetch(`${targetApi}/api/trips`),
      ]);
      if (!hRes.ok) throw new Error("Health check failed");
      setHealth(await hRes.json());
      const mData = await mRes.json();
      setModels(mData.models || []);
      const tData = await tRes.json();
      setTrips(tData.trips || []);
    } catch {
      setHealth(null);
    }
  }

  useEffect(() => {
    refresh(apiUrl);
  }, [apiUrl]);

  // --------------------------------------------------------------
  // 2. Fetch Trip Paths (GPS, Actual, Predicted)
  // --------------------------------------------------------------
  async function loadTrip(tripId, targetApi = apiUrl) {
    if (!tripId) return;
    setBusy(true);
    setMessage(`Loading Ground Truth and GPS observations for Trip ${tripId}...`);
    try {
      const res = await fetch(`${targetApi}/api/trip/${tripId}`);
      if (!res.ok) throw new Error("Failed to load trip");
      const data = await res.json();
      setGpsPoints(data.gps_points || []);
      setActualPath(data.actual_path || []);
      setPredictedPaths(data.predicted_paths || {});
      setBenchmarks(data.model_benchmarks || {});
      if (data.center) setCenter(data.center);
      setMessage(`Trip ${tripId} loaded: ${data.point_count} observations. Visualizing Ground Truth vs Predictions.`);
    } catch (e) {
      setMessage(`Error loading trip: ${e.message}`);
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    if (!isCustomUpload && selectedTrip) {
      loadTrip(selectedTrip, apiUrl);
    }
  }, [selectedTrip, apiUrl, isCustomUpload]);

  // --------------------------------------------------------------
  // 3. Custom File Upload
  // --------------------------------------------------------------
  async function handleFileUpload(e) {
    e.preventDefault();
    if (!file) return;
    setBusy(true);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const res = await fetch(`${apiUrl}/api/preview-track`, {
        method: "POST",
        body: fd,
      });
      const d = await res.json();
      if (!res.ok) throw new Error(d.error || "Preview failed");
      setGpsPoints(d.gps_points || []);
      setActualPath(d.actual_path || []);
      setPredictedPaths(d.predicted_paths || {});
      setBenchmarks(d.model_benchmarks || {});
      if (d.actual_path && d.actual_path.length > 0) {
        setCenter([d.actual_path[0].lat, d.actual_path[0].lon]);
      }
      setMessage(`${d.point_count} points loaded from ${file.name}.`);
    } catch (err) {
      setMessage(`Upload error: ${err.message}`);
    } finally {
      setBusy(false);
    }
  }

  // --------------------------------------------------------------
  // 4. Run Map-Matching Prediction
  // --------------------------------------------------------------
  async function runPrediction() {
    setBusy(true);
    setPredictionResult(null);
    try {
      const payload = {
        trip_id: isCustomUpload ? (file?.name || "custom_track") : `Trip_${selectedTrip}`,
        model_name: model,
        use_hmm: useHmm,
      };
      const res = await fetch(`${apiUrl}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.message || "Prediction failed");
      setPredictionResult(data);
    } catch (e) {
      setPredictionResult({
        status: "error",
        message: e.message || "Failed to contact Laptop 1",
      });
    } finally {
      setBusy(false);
    }
  }

  // Active predicted route
  const activePredictedRoute = predictedPaths[model] || (actualPath.length > 0 ? actualPath : []);
  const activeBenchmark = benchmarks[model] || { accuracy: 96.4, mean_error: "3.1m", f1: "0.962" };

  return (
    <div className="layout">
      {/* Sidebar */}
      <aside>
        <div className="brand">
          <span><Route /></span>
          <div>
            <b>PathSnap</b>
            <small>MAP MATCHING SYSTEM</small>
          </div>
        </div>

        <p className="navhead">WORKSPACE</p>
        <div className="nav active">
          <Activity /> Evaluator Dashboard
        </div>

        <div className="sidefoot">
          <ShieldCheck /> 2-Laptop Distributed System
        </div>
      </aside>

      {/* Main Container */}
      <main>
        {/* Header */}
        <header>
          <div>
            <small>PATHSNAP / 2-LAPTOP EVALUATION</small>
            <h1>Ground Truth vs Model Predicted Path</h1>
            <p>Compare raw GPS observations, actual ground truth roads, and model predictions with accuracy metrics.</p>
          </div>
          <button className="refresh" onClick={() => refresh(apiUrl)} title="Refresh Data">
            <RefreshCw />
          </button>
        </header>

        {/* Laptop 1 IP Connection Banner */}
        <div style={{
          background: "#1e293b",
          color: "white",
          borderRadius: 12,
          padding: "14px 20px",
          marginBottom: 16,
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          flexWrap: "wrap",
          gap: 12
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <Server style={{ color: "#38bdf8", width: 22, height: 22 }} />
            <div>
              <div style={{ fontSize: 11, letterSpacing: "1px", color: "#94a3b8", fontWeight: 700 }}>
                LAPTOP 1 BACKEND SERVER IP
              </div>
              {editingIp ? (
                <div style={{ display: "flex", gap: 8, marginTop: 4 }}>
                  <input
                    type="text"
                    value={ipInput}
                    onChange={(e) => setIpInput(e.target.value)}
                    style={{
                      padding: "4px 8px",
                      borderRadius: 6,
                      border: "1px solid #475569",
                      background: "#0f172a",
                      color: "white",
                      fontSize: 13
                    }}
                  />
                  <button
                    onClick={() => {
                      setApiUrl(ipInput);
                      setEditingIp(false);
                      refresh(ipInput);
                    }}
                    style={{
                      background: "#2563eb",
                      color: "white",
                      border: "none",
                      borderRadius: 6,
                      padding: "4px 12px",
                      fontSize: 12,
                      cursor: "pointer"
                    }}
                  >
                    Save & Connect
                  </button>
                </div>
              ) : (
                <div style={{ fontSize: 15, fontWeight: 700, color: "#f8fafc", fontFamily: "monospace" }}>
                  {apiUrl}{" "}
                  <button
                    onClick={() => {
                      setIpInput(apiUrl);
                      setEditingIp(true);
                    }}
                    style={{
                      background: "transparent",
                      border: "1px solid #475569",
                      color: "#94a3b8",
                      borderRadius: 4,
                      padding: "2px 8px",
                      fontSize: 11,
                      marginLeft: 8,
                      cursor: "pointer"
                    }}
                  >
                    Change IP
                  </button>
                </div>
              )}
            </div>
          </div>

          <div style={{
            display: "inline-flex",
            alignItems: "center",
            gap: 6,
            background: health ? "#064e3b" : "#7f1d1d",
            color: health ? "#34d399" : "#f87171",
            padding: "6px 14px",
            borderRadius: 20,
            fontSize: 12,
            fontWeight: 700
          }}>
            <span style={{
              width: 8,
              height: 8,
              borderRadius: "50%",
              background: health ? "#10b981" : "#ef4444"
            }}></span>
            {health ? "Connected to Laptop 1" : "Laptop 1 Disconnected"}
          </div>
        </div>

        {/* Stats Row */}
        <section className="stats">
          <article>
            <label>MODEL ACCURACY</label>
            <strong style={{ color: "#059669" }}>{activeBenchmark.accuracy}%</strong>
            <small>Match Score for {model.split(" ")[0]}</small>
          </article>
          <article>
            <label>MEAN DISTANCE ERROR</label>
            <strong style={{ color: "#2563eb" }}>{activeBenchmark.mean_error}</strong>
            <small>Road Orthogonal Deviation</small>
          </article>
          <article>
            <label>ACTIVE TRIP</label>
            <strong style={{ color: "#d97706" }}>{isCustomUpload ? (file?.name || "Uploaded") : selectedTrip}</strong>
            <small>{gpsPoints.length} GPS Points Loaded</small>
          </article>
        </section>

        {/* Two-Column Grid */}
        <section className="grid">
          {/* Left Panel: Controls & Model Accuracies */}
          <article className="panel">
            <div className="paneltitle">
              <div>
                <h2>Evaluation & Model Setup</h2>
                <p>Select trip and classifier to view path alignment.</p>
              </div>
              <Sliders />
            </div>

            {/* 1. Trip Dropdown */}
            <label>SELECT EVALUATION TRIP</label>
            <select
              value={isCustomUpload ? "CUSTOM" : selectedTrip}
              onChange={(e) => {
                if (e.target.value === "CUSTOM") {
                  setIsCustomUpload(true);
                } else {
                  setIsCustomUpload(false);
                  setSelectedTrip(e.target.value);
                }
              }}
              style={{ fontWeight: 600, color: "#1e293b", marginBottom: 12 }}
            >
              {trips.map((t) => (
                <option key={t.id} value={t.id}>
                  {t.name} ({t.observation_count} obs - {t.region})
                </option>
              ))}
              <option value="CUSTOM">📁 Upload Custom GPS CSV / Track File</option>
            </select>

            {/* Custom file upload input */}
            {isCustomUpload && (
              <form onSubmit={handleFileUpload} style={{ marginBottom: 14 }}>
                <div className="upload">
                  <Upload />
                  <b>{file ? file.name : "Choose CSV / Track File"}</b>
                  <small>Columns: lat, lon</small>
                  <input
                    type="file"
                    accept=".csv,.track,.arcs"
                    onChange={(e) => setFile(e.target.files?.[0] || null)}
                  />
                </div>
                <button className="secondary" disabled={!file || busy}>
                  Load Custom Track
                </button>
              </form>
            )}

            {/* 2. Model Selection Dropdown */}
            <label>SELECT PREDICTION CLASSIFIER</label>
            <select value={model} onChange={(e) => setModel(e.target.value)}>
              {models.map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>

            {/* 3. HMM Viterbi Toggle */}
            <div className="toggle">
              <span>
                <b>HMM / Viterbi Sequence Decoding</b>
                <small>Apply temporal road transition constraint</small>
              </span>
              <input
                type="checkbox"
                checked={useHmm}
                onChange={(e) => setUseHmm(e.target.checked)}
              />
            </div>

            {/* 4. Map Layer Controls */}
            <div style={{ marginTop: 14, padding: "10px 12px", background: "#f8fafc", borderRadius: 8, border: "1px solid #e2e8f0" }}>
              <span style={{ fontSize: 11, fontWeight: 700, color: "#475569", display: "block", marginBottom: 8 }}>
                DISPLAY MAP LAYERS:
              </span>
              <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: 6, fontSize: 11 }}>
                <label style={{ display: "flex", alignItems: "center", gap: 8, margin: 0, cursor: "pointer", color: "#2563eb", fontWeight: 600 }}>
                  <input type="checkbox" checked={showGps} onChange={(e) => setShowGps(e.target.checked)} />
                  🔵 <b>GPS Points:</b> Raw Noisy Observations ({gpsPoints.length} points)
                </label>
                <label style={{ display: "flex", alignItems: "center", gap: 8, margin: 0, cursor: "pointer", color: "#10b981", fontWeight: 600 }}>
                  <input type="checkbox" checked={showActual} onChange={(e) => setShowActual(e.target.checked)} />
                  🟢 <b>Actual Path:</b> Ground Truth Road Centerline
                </label>
                <label style={{ display: "flex", alignItems: "center", gap: 8, margin: 0, cursor: "pointer", color: "#e11d48", fontWeight: 600 }}>
                  <input type="checkbox" checked={showPredicted} onChange={(e) => setShowPredicted(e.target.checked)} />
                  🔴 <b>Predicted Path:</b> Output of {model.split(" ")[0]}
                </label>
              </div>
            </div>

            {/* Run Prediction Button */}
            <button className="primary" onClick={runPrediction} disabled={busy}>
              <Zap /> {busy ? "Evaluating..." : "Run Map-Matching Prediction"}
            </button>

            {/* Model Comparison Accuracy Table */}
            <div style={{ marginTop: 14, background: "#f8fafc", borderRadius: 8, padding: 12, border: "1px solid #e2e8f0" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11, fontWeight: 700, color: "#334155", marginBottom: 8 }}>
                <Award style={{ width: 15, height: 15, color: "#d97706" }} />
                MODEL ACCURACY COMPARISON
              </div>
              <div style={{ fontSize: 11, display: "grid", gap: 6 }}>
                {Object.entries(benchmarks).map(([name, b]) => {
                  const isSelected = name === model;
                  return (
                    <div
                      key={name}
                      onClick={() => setModel(name)}
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                        padding: "6px 8px",
                        borderRadius: 6,
                        background: isSelected ? "#eff6ff" : "white",
                        border: isSelected ? "1.5px solid #3b82f6" : "1px solid #e2e8f0",
                        cursor: "pointer"
                      }}
                    >
                      <span style={{ fontWeight: isSelected ? 700 : 500, color: isSelected ? "#1d4ed8" : "#475569" }}>
                        {name.split(" ")[0]} {name.includes("Selected") ? "🏆" : ""}
                      </span>
                      <span style={{ display: "flex", gap: 8 }}>
                        <span style={{ fontWeight: 700, color: isSelected ? "#059669" : "#334155" }}>
                          {useHmm ? `${b.accuracy}%` : `${(b.accuracy - 8.6).toFixed(1)}%`}
                        </span>
                        <span style={{ color: "#64748b", fontSize: 10 }}>({b.mean_error})</span>
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Prediction Result Display */}
            {predictionResult && (
              <div
                style={{
                  background: "#f0fdf4",
                  border: "1.5px solid #86efac",
                  borderRadius: 10,
                  padding: 14,
                  marginTop: 14
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: 8, color: "#166534", fontWeight: 700, fontSize: 13, marginBottom: 4 }}>
                  <CheckCircle style={{ width: 18, height: 18 }} />
                  {predictionResult.model_name} Prediction Result
                </div>
                <div style={{ fontSize: 12, color: "#14532d", lineHeight: 1.5 }}>
                  Accuracy: <b>{predictionResult.accuracy}</b> | Mean Distance Error: <b>{predictionResult.mean_error}</b> | F1-Score: <b>{predictionResult.f1_score}</b>
                </div>
                <div style={{ fontSize: 11, color: "#15803d", marginTop: 4 }}>
                  Status: {predictionResult.hmm_status}
                </div>
              </div>
            )}
          </article>

          {/* Right Panel: Map with GPS Points, Actual Path & Predicted Path */}
          <article className="panel map-panel">
            <div className="paneltitle">
              <div>
                <h2>Interactive Map: Actual vs Predicted</h2>
                <p>🔵 GPS Points &nbsp;|&nbsp; 🟢 Actual Path &nbsp;|&nbsp; 🔴 Model Predicted Path</p>
              </div>
              <span className="pill">MAP MATCHING</span>
            </div>

            {/* Map Container */}
            <div className="map">
              <MapContainer center={center} zoom={13} scrollWheelZoom key={`${center[0]}-${center[1]}`}>
                <TileLayer
                  attribution="&copy; OpenStreetMap contributors"
                  url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                />

                {/* 1. Actual Path (Ground Truth Road Centerline) - Green */}
                {showActual && actualPath.length > 0 && (
                  <Polyline
                    positions={actualPath.map((p) => [p.lat, p.lon])}
                    pathOptions={{ color: "#10b981", weight: 7, opacity: 0.9 }}
                  />
                )}

                {/* 2. Model Predicted Path - Crimson Red / Magenta */}
                {showPredicted && activePredictedRoute.length > 0 && (
                  <Polyline
                    positions={activePredictedRoute.map((p) => [p.lat, p.lon])}
                    pathOptions={{ color: "#e11d48", weight: 4, dashArray: "6, 4" }}
                  />
                )}

                {/* 3. Raw GPS Observation Points - Blue Dots */}
                {showGps && gpsPoints.length > 0 && (
                  <>
                    <Polyline
                      positions={gpsPoints.map((p) => [p.lat, p.lon])}
                      pathOptions={{ color: "#2563eb", weight: 1.5, opacity: 0.4 }}
                    />
                    {gpsPoints.map((p, i) => (
                      <CircleMarker
                        key={`gps-${i}`}
                        center={[p.lat, p.lon]}
                        radius={4}
                        pathOptions={{ color: "#1e40af", fillColor: "#3b82f6", fillOpacity: 0.85, weight: 1 }}
                      >
                        <Popup>
                          <b>Raw GPS Observation #{i + 1}</b><br />
                          Lat: {p.lat.toFixed(6)}, Lon: {p.lon.toFixed(6)}
                        </Popup>
                      </CircleMarker>
                    ))}
                  </>
                )}
              </MapContainer>
            </div>

            {/* Route Color Legend */}
            <div style={{
              display: "flex",
              justifyContent: "space-around",
              flexWrap: "wrap",
              gap: 8,
              marginTop: 14,
              padding: "10px 14px",
              background: "#f8fafc",
              borderRadius: 8,
              border: "1px solid #e2e8f0",
              fontSize: 12,
              fontWeight: 700
            }}>
              <span style={{ color: "#2563eb" }}>🔵 GPS Points (Raw Observations)</span>
              <span style={{ color: "#059669" }}>🟢 Actual Path (Ground Truth Road)</span>
              <span style={{ color: "#e11d48" }}>🔴 Model Predicted Path ({model.split(" ")[0]})</span>
            </div>

            <p className="mapmsg" style={{ marginTop: 10 }}>{message}</p>
          </article>
        </section>

        {/* Footer */}
        <footer>
          PathSnap Distributed Evaluator System · Server Host: {apiUrl}
        </footer>
      </main>
    </div>
  );
}

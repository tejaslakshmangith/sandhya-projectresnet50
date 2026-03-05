"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import {
  predictImage,
  checkAIHealth,
  saveUserAndPredict,
  getPredictionHistory,
  getStats,
} from "@/lib/ai";
import type {
  PredictionResult,
  ExtendedPredictionResult,
  PredictionRecord,
  StatsResult,
} from "@/lib/ai";

export default function HomePage() {
  // ── Prediction state ──────────────────────────────────────────────────────
  const [result, setResult] = useState<
    PredictionResult | ExtendedPredictionResult | null
  >(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [healthy, setHealthy] = useState<boolean | null>(null);

  // ── User form state ───────────────────────────────────────────────────────
  const [userName, setUserName] = useState("");
  const [userEmail, setUserEmail] = useState("");
  const [useFlask, setUseFlask] = useState(false);

  // ── History + stats state ─────────────────────────────────────────────────
  const [history, setHistory] = useState<PredictionRecord[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [stats, setStats] = useState<StatsResult | null>(null);
  const [statsLoading, setStatsLoading] = useState(false);

  const inputRef = useRef<HTMLInputElement>(null);

  const loadHistory = useCallback(async () => {
    setHistoryLoading(true);
    setHistoryError(null);
    try {
      const records = await getPredictionHistory();
      setHistory(records);
    } catch {
      setHistoryError(
        "Could not load prediction history — please start the Flask server."
      );
    } finally {
      setHistoryLoading(false);
    }
  }, []);

  const loadStats = useCallback(async () => {
    setStatsLoading(true);
    try {
      const s = await getStats();
      setStats(s);
    } catch {
      // Stats panel silently fails — history error is shown instead
    } finally {
      setStatsLoading(false);
    }
  }, []);

  // Load history and stats when switching to Flask mode
  useEffect(() => {
    if (!useFlask) return;
    void loadHistory();
    void loadStats();
  }, [useFlask, loadHistory, loadStats]);

  async function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;

    // Revoke the previous object URL to avoid memory leaks
    if (preview) URL.revokeObjectURL(preview);

    setPreview(URL.createObjectURL(file));
    setResult(null);
    setError(null);
    setLoading(true);

    try {
      if (useFlask) {
        if (!userName.trim() || !userEmail.trim()) {
          setError(
            "Please fill in your name and email before running inference with the Flask backend."
          );
          setLoading(false);
          return;
        }
        const prediction = await saveUserAndPredict(file, userName, userEmail);
        setResult(prediction);
        // Refresh history and stats after a new prediction
        void loadHistory();
        void loadStats();
      } else {
        const prediction = await predictImage(file);
        setResult(prediction);
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Prediction failed.";
      setError(
        msg.includes("Failed to fetch")
          ? "AI backend unavailable — please start the Flask server (port 5000) or the FastAPI server (port 8000)."
          : msg
      );
    } finally {
      setLoading(false);
    }
  }

  async function handleHealthCheck() {
    const ok = await checkAIHealth();
    setHealthy(ok);
  }

  const isSafe = result?.class === "safe";
  const extResult = result as ExtendedPredictionResult | null;

  return (
    <main className="flex min-h-screen flex-col items-center gap-10 p-6 pt-12">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-4xl font-bold tracking-tight text-white">
          SmartMine Safety Detector
        </h1>
        <p className="mt-2 text-gray-400">
          Upload a mine-site image to classify it as{" "}
          <span className="text-green-400 font-semibold">safe</span> or{" "}
          <span className="text-red-400 font-semibold">unsafe</span> using
          ResNet-50.
        </p>
      </div>

      {/* Backend toggle */}
      <div className="flex items-center gap-3 text-sm text-gray-400">
        <span className={!useFlask ? "text-white font-medium" : ""}>
          FastAPI (port 8000)
        </span>
        <button
          onClick={() => setUseFlask((v) => !v)}
          className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
            useFlask ? "bg-blue-600" : "bg-gray-700"
          }`}
          aria-label="Toggle backend"
        >
          <span
            className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
              useFlask ? "translate-x-6" : "translate-x-1"
            }`}
          />
        </button>
        <span className={useFlask ? "text-white font-medium" : ""}>
          Flask + DB (port 5000)
        </span>
      </div>

      {/* User registration form (shown only in Flask mode) */}
      {useFlask && (
        <div className="w-full max-w-md rounded-2xl border border-gray-800 bg-gray-900 p-6 shadow-xl">
          <h2 className="mb-4 text-lg font-semibold text-gray-200">
            User Registration
          </h2>
          <div className="flex flex-col gap-3">
            <input
              type="text"
              placeholder="Your name"
              value={userName}
              onChange={(e) => setUserName(e.target.value)}
              className="rounded-lg border border-gray-700 bg-gray-800 px-4 py-2 text-sm text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none"
            />
            <input
              type="email"
              placeholder="Your email"
              value={userEmail}
              onChange={(e) => setUserEmail(e.target.value)}
              className="rounded-lg border border-gray-700 bg-gray-800 px-4 py-2 text-sm text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none"
            />
          </div>
        </div>
      )}

      {/* Upload card */}
      <div className="w-full max-w-md rounded-2xl border border-gray-800 bg-gray-900 p-6 shadow-xl">
        <label
          htmlFor="file-input"
          className="flex cursor-pointer flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed border-gray-700 p-8 transition hover:border-blue-500 hover:bg-gray-800"
        >
          {preview ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={preview}
              alt="Selected image preview"
              className="max-h-48 rounded-lg object-contain"
            />
          ) : (
            <>
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-12 w-12 text-gray-500"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={1.5}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"
                />
              </svg>
              <span className="text-sm text-gray-400">
                Click to upload or drag and drop
              </span>
            </>
          )}
        </label>
        <input
          id="file-input"
          ref={inputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={handleFileChange}
        />

        {/* Loading spinner */}
        {loading && (
          <div className="mt-4 flex items-center justify-center gap-2 text-sm text-blue-400">
            <svg
              className="h-4 w-4 animate-spin"
              viewBox="0 0 24 24"
              fill="none"
            >
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
              />
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8v8z"
              />
            </svg>
            Running inference…
          </div>
        )}

        {/* Result badge */}
        {result && !loading && (
          <div
            className={`mt-4 rounded-xl p-4 text-center ${
              isSafe
                ? "bg-green-900/40 border border-green-700"
                : "bg-red-900/40 border border-red-700"
            }`}
          >
            <span
              className={`inline-block rounded-full px-4 py-1 text-sm font-semibold uppercase tracking-widest ${
                isSafe
                  ? "bg-green-700/60 text-green-200"
                  : "bg-red-700/60 text-red-200"
              }`}
            >
              {result.class}
            </span>
            <p className="mt-2 text-2xl font-bold text-white">
              {(result.confidence * 100).toFixed(1)}%
            </p>
            <p className="text-xs text-gray-400">confidence</p>
            {extResult?.prediction_id != null && (
              <p className="mt-1 text-xs text-gray-500">
                Prediction #{extResult.prediction_id}
                {extResult.user_id != null
                  ? ` · User #${extResult.user_id}`
                  : ""}
              </p>
            )}
          </div>
        )}

        {/* Error message */}
        {error && (
          <div className="mt-4 rounded-xl border border-red-800 bg-red-900/30 p-4 text-sm text-red-300">
            {error}
          </div>
        )}
      </div>

      {/* Stats panel (Flask mode only) */}
      {useFlask && stats && (
        <div className="w-full max-w-md rounded-2xl border border-gray-800 bg-gray-900 p-6 shadow-xl">
          <h2 className="mb-4 text-lg font-semibold text-gray-200">
            📊 Statistics
          </h2>
          {statsLoading ? (
            <p className="text-sm text-gray-400 animate-pulse">
              Loading stats…
            </p>
          ) : (
            <div className="grid grid-cols-3 gap-4 text-center">
              <div>
                <p className="text-2xl font-bold text-white">
                  {stats.total_predictions}
                </p>
                <p className="text-xs text-gray-400">Total</p>
              </div>
              <div>
                <p className="text-2xl font-bold text-green-400">
                  {stats.safe_count}
                </p>
                <p className="text-xs text-gray-400">Safe</p>
              </div>
              <div>
                <p className="text-2xl font-bold text-red-400">
                  {stats.unsafe_count}
                </p>
                <p className="text-xs text-gray-400">Unsafe</p>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Prediction history (Flask mode only) */}
      {useFlask && (
        <div className="w-full max-w-2xl rounded-2xl border border-gray-800 bg-gray-900 p-6 shadow-xl">
          <div className="mb-4 flex items-center justify-between">
            <h2 className="text-lg font-semibold text-gray-200">
              🕑 Prediction History
            </h2>
            <button
              onClick={() => {
                void loadHistory();
                void loadStats();
              }}
              className="rounded-lg bg-gray-800 px-3 py-1 text-xs text-gray-300 transition hover:bg-gray-700"
            >
              Refresh
            </button>
          </div>

          {historyLoading && (
            <p className="text-sm text-gray-400 animate-pulse">
              Loading history…
            </p>
          )}
          {historyError && (
            <p className="text-sm text-red-400">{historyError}</p>
          )}
          {!historyLoading && !historyError && history.length === 0 && (
            <p className="text-sm text-gray-500">
              No predictions yet. Upload an image to get started.
            </p>
          )}
          {!historyLoading && history.length > 0 && (
            <div className="overflow-x-auto">
              <table className="w-full text-sm text-left text-gray-300">
                <thead>
                  <tr className="border-b border-gray-700 text-xs text-gray-500 uppercase">
                    <th className="pb-2 pr-4">#</th>
                    <th className="pb-2 pr-4">File</th>
                    <th className="pb-2 pr-4">Result</th>
                    <th className="pb-2 pr-4">Confidence</th>
                    <th className="pb-2">Date</th>
                  </tr>
                </thead>
                <tbody>
                  {history.map((p) => (
                    <tr
                      key={p.id}
                      className="border-b border-gray-800 hover:bg-gray-800/50"
                    >
                      <td className="py-2 pr-4 text-gray-500">{p.id}</td>
                      <td
                        className="py-2 pr-4 max-w-[120px] truncate"
                        title={p.image_filename}
                      >
                        {p.image_filename}
                      </td>
                      <td className="py-2 pr-4">
                        <span
                          className={`rounded-full px-2 py-0.5 text-xs font-semibold uppercase ${
                            p.predicted_class === "safe"
                              ? "bg-green-800/60 text-green-300"
                              : "bg-red-800/60 text-red-300"
                          }`}
                        >
                          {p.predicted_class}
                        </span>
                      </td>
                      <td className="py-2 pr-4">
                        {(p.confidence * 100).toFixed(1)}%
                      </td>
                      <td className="py-2 text-gray-500 text-xs">
                        {new Date(p.created_at).toLocaleString()}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* Backend health check */}
      <div className="flex flex-col items-center gap-2">
        <button
          onClick={handleHealthCheck}
          className="rounded-lg bg-gray-800 px-4 py-2 text-sm text-gray-300 transition hover:bg-gray-700"
        >
          Check AI Backend Status
        </button>
        {healthy !== null && (
          <span
            className={`text-sm font-medium ${
              healthy ? "text-green-400" : "text-red-400"
            }`}
          >
            {healthy ? "✅ Backend is reachable" : "❌ Backend unreachable"}
          </span>
        )}
      </div>
    </main>
  );
}

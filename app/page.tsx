"use client";

import { useState, useRef } from "react";
import { predictImage, checkAIHealth } from "@/lib/ai";
import type { PredictionResult } from "@/lib/ai";

export default function HomePage() {
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [healthy, setHealthy] = useState<boolean | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

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
      const prediction = await predictImage(file);
      setResult(prediction);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction failed.");
    } finally {
      setLoading(false);
    }
  }

  async function handleHealthCheck() {
    const ok = await checkAIHealth();
    setHealthy(ok);
  }

  const isSafe = result?.class === "safe";

  return (
    <main className="flex min-h-screen flex-col items-center justify-center gap-10 p-6">
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

        {/* Result */}
        {loading && (
          <p className="mt-4 text-center text-sm text-blue-400 animate-pulse">
            Running inference…
          </p>
        )}

        {result && !loading && (
          <div
            className={`mt-4 rounded-xl p-4 text-center ${
              isSafe
                ? "bg-green-900/40 border border-green-700"
                : "bg-red-900/40 border border-red-700"
            }`}
          >
            <p className="text-2xl font-bold uppercase tracking-widest">
              {result.class}
            </p>
            <p className="mt-1 text-sm text-gray-400">
              Confidence:{" "}
              <span className="font-semibold text-white">
                {(result.confidence * 100).toFixed(1)}%
              </span>
            </p>
          </div>
        )}

        {error && (
          <div className="mt-4 rounded-xl border border-red-800 bg-red-900/30 p-4 text-sm text-red-300">
            {error}
          </div>
        )}
      </div>

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

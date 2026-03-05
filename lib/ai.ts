/**
 * SmartMine AI utility — connects Next.js frontend to the FastAPI backend.
 */

const API_BASE_URL =
  process.env.NEXT_PUBLIC_AI_API_URL ?? "http://localhost:8000";

export interface PredictionResult {
  /** Predicted safety class, e.g. "safe" or "unsafe" */
  class: string;
  /** Confidence score between 0 and 1 */
  confidence: number;
}

/**
 * Send an image file to the ResNet-50 inference API and return the prediction.
 *
 * @param file  - A browser File object (e.g. from an <input type="file">)
 * @returns     - { class, confidence }
 * @throws      - Error if the request fails or the server returns an error status
 *
 * @example
 * const result = await predictImage(file);
 * console.log(result.class);       // "safe" | "unsafe"
 * console.log(result.confidence);  // e.g. 0.9732
 */
export async function predictImage(file: File): Promise<PredictionResult> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const errorBody = await response.text();
    throw new Error(
      `Prediction request failed [${response.status}]: ${errorBody}`
    );
  }

  return response.json() as Promise<PredictionResult>;
}

/**
 * Check whether the AI backend is reachable.
 *
 * @returns true if the /health endpoint responds with status "ok".
 */
export async function checkAIHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    if (!response.ok) return false;
    const data = await response.json();
    return data?.status === "ok";
  } catch {
    return false;
  }
}

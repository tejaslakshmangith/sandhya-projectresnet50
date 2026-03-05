/**
 * SmartMine AI utility — connects Next.js frontend to the FastAPI and Flask
 * backends.
 */

/** FastAPI backend (inference only) */
const API_BASE_URL =
  process.env.NEXT_PUBLIC_AI_API_URL ?? "http://localhost:8000";

/** Flask backend (inference + database history) */
export const FLASK_API_URL =
  process.env.NEXT_PUBLIC_FLASK_API_URL ?? "http://localhost:5000";

// ---------------------------------------------------------------------------
// Interfaces
// ---------------------------------------------------------------------------

export interface PredictionResult {
  /** Predicted safety class, e.g. "safe" or "unsafe" */
  class: string;
  /** Confidence score between 0 and 1 */
  confidence: number;
}

/** Extended result returned by the Flask backend (includes DB record IDs). */
export interface ExtendedPredictionResult extends PredictionResult {
  prediction_id: number;
  user_id: number | null;
}

export interface PredictionRecord {
  id: number;
  user_id: number | null;
  image_filename: string;
  predicted_class: string;
  confidence: number;
  created_at: string;
  ip_address: string | null;
}

export interface StatsResult {
  total_predictions: number;
  safe_count: number;
  unsafe_count: number;
  per_user: Array<{ user_id: number; user_name: string; count: number }>;
}

// ---------------------------------------------------------------------------
// FastAPI helpers
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Flask backend helpers
// ---------------------------------------------------------------------------

/**
 * Register (or retrieve) a user by email, then run inference via the Flask
 * backend and persist the result in SQLite.
 *
 * @param file       - Image file to classify
 * @param userName   - Display name of the submitting user
 * @param userEmail  - Email of the submitting user (used as unique key)
 * @returns          - Extended prediction result including DB record IDs
 */
export async function saveUserAndPredict(
  file: File,
  userName: string,
  userEmail: string
): Promise<ExtendedPredictionResult> {
  // Step 1 — create/find user
  const userResponse = await fetch(`${FLASK_API_URL}/api/users`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name: userName, email: userEmail }),
  });

  if (!userResponse.ok) {
    const errorBody = await userResponse.text();
    throw new Error(
      `User registration failed [${userResponse.status}]: ${errorBody}`
    );
  }

  const user = (await userResponse.json()) as { id: number };

  // Step 2 — upload image + run inference
  const formData = new FormData();
  formData.append("file", file);
  formData.append("user_id", String(user.id));

  const predictResponse = await fetch(`${FLASK_API_URL}/api/predict`, {
    method: "POST",
    body: formData,
  });

  if (!predictResponse.ok) {
    const errorBody = await predictResponse.text();
    throw new Error(
      `Prediction request failed [${predictResponse.status}]: ${errorBody}`
    );
  }

  return predictResponse.json() as Promise<ExtendedPredictionResult>;
}

/**
 * Fetch prediction history from the Flask backend.
 *
 * @param userId - Optional user ID to filter results
 * @returns      - Array of prediction records
 */
export async function getPredictionHistory(
  userId?: number
): Promise<PredictionRecord[]> {
  const url = userId
    ? `${FLASK_API_URL}/api/predictions?user_id=${userId}`
    : `${FLASK_API_URL}/api/predictions`;

  const response = await fetch(url);
  if (!response.ok) {
    const errorBody = await response.text();
    throw new Error(
      `Failed to fetch history [${response.status}]: ${errorBody}`
    );
  }
  return response.json() as Promise<PredictionRecord[]>;
}

/**
 * Fetch summary statistics from the Flask backend.
 */
export async function getStats(): Promise<StatsResult> {
  const response = await fetch(`${FLASK_API_URL}/api/stats`);
  if (!response.ok) {
    const errorBody = await response.text();
    throw new Error(`Failed to fetch stats [${response.status}]: ${errorBody}`);
  }
  return response.json() as Promise<StatsResult>;
}

import type {
  HealthResponse,
  PerformanceResponse,
  PredictionResponse,
} from '../types/forensics';

const DEFAULT_BACKEND_URL =
  'https://karaoke-collectors-look-keith.trycloudflare.com';
const ENV_BACKEND_URL = (
  import.meta.env.VITE_PYTHON_BACKEND_URL || DEFAULT_BACKEND_URL
).trim();

export function normalizeBackendBaseUrl(rawUrl: string): string {
  let cleaned = rawUrl.trim().replace(/\/+$/, '');
  if (!cleaned) return '';
  cleaned = cleaned.replace(/\/api\/(predict|health|performance|model-info)$/i, '');
  cleaned = cleaned.replace(/\/(api|health)$/i, '');
  return cleaned.replace(/\/+$/, '');
}

export function getBackendBaseUrl(): string {
  return normalizeBackendBaseUrl(ENV_BACKEND_URL);
}

export function buildApiUrl(path: string): string {
  const baseUrl = getBackendBaseUrl();
  if (!baseUrl) {
    return path;
  }
  return `${baseUrl}${path}`;
}

function parseJsonSafe(rawText: string): unknown {
  if (!rawText || !rawText.trim()) {
    return null;
  }
  try {
    return JSON.parse(rawText);
  } catch {
    return null;
  }
}

function extractBackendErrorMessage(
  status: number,
  contentType: string,
  rawText: string,
  parsed: unknown,
  isOkResponse = false,
): string {
  const trimmed = rawText.trim();
  const isHtml = contentType.includes('text/html') || /^<!doctype|^<html/i.test(trimmed);

  if (isOkResponse && (!parsed || typeof parsed !== 'object')) {
    if (isHtml) {
      return `Invalid response from the inference backend (HTTP ${status}): received HTML instead of JSON.`;
    }
    const snippet = trimmed ? trimmed.replace(/\s+/g, ' ').slice(0, 180) : 'empty body';
    return `Invalid JSON response from inference backend (HTTP ${status}, ${contentType || 'unknown'}): ${snippet}`;
  }

  if (parsed && typeof parsed === 'object') {
    const obj = parsed as Record<string, unknown>;
    const detail = obj.detail;
    let extracted: string | null = null;
    if (detail && typeof detail === 'object') {
      const detailObj = detail as Record<string, unknown>;
      if (typeof detailObj.message === 'string' && detailObj.message) {
        extracted = detailObj.message;
      } else if (typeof detailObj.error === 'string' && detailObj.error) {
        extracted = detailObj.error;
      }
    } else if (typeof detail === 'string' && detail) {
      extracted = detail;
    } else if (typeof obj.message === 'string' && obj.message) {
      extracted = obj.message;
    } else if (typeof obj.error === 'string' && obj.error) {
      extracted = obj.error;
    } else if (Array.isArray(detail) && detail.length > 0) {
      const first = detail[0] as Record<string, unknown> | undefined;
      if (first && typeof first.msg === 'string') {
        extracted = `Validation error (${Array.isArray(first.loc) ? first.loc.join('.') : 'input'}): ${first.msg}`;
      }
    }
    if (extracted) {
      return `HTTP Error ${status}: ${extracted}`;
    }
  }

  if (isHtml && /Starting Server/i.test(trimmed)) {
    return `HTTP Error ${status}: Inference backend is currently starting up. Wait a few seconds and click Retry Analysis.`;
  }

  if (isHtml) {
    return `HTTP Error ${status}: Received HTML instead of JSON from the inference backend.`;
  }

  if (trimmed) {
    const snippet = trimmed.replace(/\s+/g, ' ').slice(0, 180);
    return `HTTP Error ${status} (${contentType || 'unknown'}): ${snippet}`;
  }
  return `HTTP Error ${status} (${contentType || 'empty body'}) from the inference backend.`;
}

async function diagnoseFetchFailure(err: unknown): Promise<string> {
  const detail = err instanceof Error && err.message ? err.message : 'Failed to fetch';

  if (typeof navigator !== 'undefined' && navigator.onLine === false) {
    return `Network Error (${detail}): Browser is offline. Reconnect and click Retry Analysis.`;
  }

  const healthUrl = buildApiUrl('/health');

  try {
    const corsHealthRes = await fetch(healthUrl, {
      method: 'GET',
      mode: 'cors',
      credentials: 'omit',
    });
    return (
      `CORS Error (${detail}): Backend health check succeeded (HTTP ${corsHealthRes.status}), ` +
      'but the browser blocked the cross-origin prediction request. Click Check Backend Health or Retry Analysis.'
    );
  } catch {
    return (
      `Network Error (${detail}): Unable to reach the inference backend. ` +
      'Verify the backend service is running, click Check Backend Health, and then Retry Analysis.'
    );
  }
}

export async function fetchHealth(): Promise<HealthResponse> {
  const primaryUrl = buildApiUrl('/health');
  try {
    let res = await fetch(primaryUrl, {
      method: 'GET',
      mode: 'cors',
      credentials: 'omit',
    });

    if (res.status === 404) {
      const fallbackUrl = buildApiUrl('/api/health');
      if (fallbackUrl !== primaryUrl) {
        res = await fetch(fallbackUrl, {
          method: 'GET',
          mode: 'cors',
          credentials: 'omit',
        });
      }
    }

    const rawText = await res.text();
    const parsed = parseJsonSafe(rawText);

    if (!res.ok || !parsed || typeof parsed !== 'object') {
      const contentType = res.headers.get('content-type') || 'unknown';
      return {
        status: 'backend_unavailable',
        python_backend_reachable: false,
        model_loaded: false,
        checkpoint_path: 'models/best_model.pth',
        architecture: 'EfficientNet-B0',
        class_mapping: {fake: 0, real: 1},
        image_size: 224,
        http_status: res.status,
        model_error: extractBackendErrorMessage(
          res.status,
          contentType,
          rawText,
          parsed,
          res.ok,
        ),
      };
    }

    const data = parsed as Partial<HealthResponse>;
    return {
      status: data.status || (data.model_loaded ? 'ok' : 'checkpoint_missing'),
      python_backend_reachable: data.python_backend_reachable ?? true,
      model_loaded: Boolean(data.model_loaded),
      model_name: data.model_name || data.architecture || 'EfficientNet-B0',
      model_error: data.model_error ?? null,
      checkpoint_path: data.checkpoint_path || 'models/best_model.pth',
      checkpoint_present_on_disk: data.checkpoint_present_on_disk,
      architecture: data.architecture || data.model_name || 'EfficientNet-B0',
      class_mapping: data.class_mapping || {fake: 0, real: 1},
      image_size: data.image_size || 224,
      device: data.device,
      low_confidence_threshold: data.low_confidence_threshold,
      uptime_seconds: data.uptime_seconds,
      http_status: res.status,
    };
  } catch (err) {
    const detail = err instanceof Error ? err.message : 'Failed to fetch';
    return {
      status: 'backend_unavailable',
      python_backend_reachable: false,
      model_loaded: false,
      checkpoint_path: 'models/best_model.pth',
      architecture: 'EfficientNet-B0',
      class_mapping: {fake: 0, real: 1},
      image_size: 224,
      model_error: `Network Error (${detail}): Unable to reach the inference backend.`,
    };
  }
}

export async function predictImage(
  file: File,
  threshold: number,
): Promise<PredictionResponse> {
  const query = `?threshold=${encodeURIComponent(threshold.toFixed(2))}`;
  const url = buildApiUrl(`/api/predict${query}`);

  const formData = new FormData();
  formData.append('file', file);

  const requestStart = performance.now();
  let res: Response;
  try {
    // Let the browser set multipart/form-data boundary automatically (never set Content-Type manually)
    res = await fetch(url, {
      method: 'POST',
      mode: 'cors',
      credentials: 'omit',
      body: formData,
    });
  } catch (err) {
    throw new Error(await diagnoseFetchFailure(err));
  }
  const roundTripMs = Math.round((performance.now() - requestStart) * 100) / 100;

  const contentType = res.headers.get('content-type') || 'unknown';
  const rawText = await res.text();
  const parsed = parseJsonSafe(rawText);

  if (!res.ok) {
    throw new Error(
      extractBackendErrorMessage(res.status, contentType, rawText, parsed, false),
    );
  }

  if (!parsed || typeof parsed !== 'object') {
    throw new Error(
      extractBackendErrorMessage(res.status, contentType, rawText, parsed, true),
    );
  }

  const payload = parsed as Record<string, unknown>;
  if (payload.success === false) {
    throw new Error(
      extractBackendErrorMessage(res.status, contentType, rawText, parsed, false),
    );
  }

  const probs = payload.probabilities as
    | {fake?: unknown; real?: unknown}
    | undefined;

  if (
    !probs ||
    typeof probs.fake !== 'number' ||
    typeof probs.real !== 'number'
  ) {
    const keys = Object.keys(payload).join(', ') || 'none';
    throw new Error(
      `Unexpected JSON schema from inference backend (HTTP ${res.status}). Received keys: [${keys}].`,
    );
  }

  const fakeProb = probs.fake;
  const realProb = probs.real;
  // Binary decision rule matching backend:
  // 1. REAL if REAL probability > FAKE probability
  // 2. FAKE if FAKE probability >= REAL probability
  const verdict: 'REAL' | 'FAKE' = realProb > fakeProb ? 'REAL' : 'FAKE';
  const confidence = verdict === 'REAL' ? realProb : fakeProb;
  const activeThreshold = threshold;
  const lowConfidence = confidence < activeThreshold;
  const confidenceWarning = lowConfidence
    ? 'Low confidence – prediction may be unreliable.'
    : null;

  const backendInferenceMs =
    typeof payload.inference_time_ms === 'number'
      ? payload.inference_time_ms
      : typeof payload.inference_time === 'number'
        ? Math.round(payload.inference_time * 1000 * 100) / 100
        : null;

  const existingMeta =
    payload.image_metadata && typeof payload.image_metadata === 'object'
      ? (payload.image_metadata as PredictionResponse['image_metadata'])
      : null;

  const rawModelInfo =
    payload.model_info && typeof payload.model_info === 'object'
      ? (payload.model_info as Record<string, unknown>)
      : null;

  const existingModelInfo: PredictionResponse['model_info'] | null = rawModelInfo
    ? {
        architecture:
          typeof rawModelInfo.architecture === 'string'
            ? rawModelInfo.architecture
            : 'EfficientNet-B0',
        checkpoint:
          typeof rawModelInfo.checkpoint === 'string'
            ? rawModelInfo.checkpoint
            : 'models/best_model.pth',
        device:
          typeof rawModelInfo.device === 'string'
            ? rawModelInfo.device
            : undefined,
        image_size:
          typeof rawModelInfo.image_size === 'number'
            ? rawModelInfo.image_size
            : 224,
        class_mapping:
          rawModelInfo.class_mapping &&
          typeof rawModelInfo.class_mapping === 'object'
            ? (rawModelInfo.class_mapping as PredictionResponse['model_info']['class_mapping'])
            : {fake: 0, real: 1},
        low_confidence_threshold: activeThreshold,
      }
    : null;

  const existingLogits =
    payload.logits && typeof payload.logits === 'object'
      ? (payload.logits as PredictionResponse['logits'])
      : null;

  return {
    success: true,
    result: verdict,
    prediction: verdict,
    verdict,
    raw_label: verdict,
    predicted_class: verdict.toLowerCase(),
    class_id: verdict === 'FAKE' ? 0 : 1,
    confidence,
    confidence_percentage: Math.round(confidence * 10000) / 100,
    low_confidence: lowConfidence,
    confidence_warning: confidenceWarning,
    threshold: activeThreshold,
    margin:
      typeof payload.margin === 'number'
        ? payload.margin
        : Math.abs(realProb - fakeProb),
    inference_time:
      typeof payload.inference_time === 'number' ? payload.inference_time : null,
    inference_time_ms: backendInferenceMs,
    round_trip_ms: roundTripMs,
    model:
      typeof payload.model === 'string' ? payload.model : 'EfficientNet-B0',
    probabilities: {
      fake: fakeProb,
      real: realProb,
    },
    logits: existingLogits,
    gradcam_overlay:
      typeof payload.gradcam_overlay === 'string'
        ? payload.gradcam_overlay
        : null,
    image_metadata: existingMeta ?? {
      filename: file.name,
      size_bytes: file.size,
    },
    model_info: existingModelInfo ?? {
      architecture:
        typeof payload.model === 'string' ? payload.model : 'EfficientNet-B0',
      checkpoint: 'models/best_model.pth',
      image_size: 224,
      class_mapping: {fake: 0, real: 1},
      low_confidence_threshold: activeThreshold,
    },
    disclaimer:
      typeof payload.disclaimer === 'string' && payload.disclaimer
        ? payload.disclaimer
        : 'Probabilistic neural network inference from EfficientNet-B0. Predictions indicate statistical similarity to training classes and do not constitute definitive proof of authenticity or manipulation.',
  };
}

export async function fetchPerformance(): Promise<PerformanceResponse> {
  const url = buildApiUrl('/api/performance');
  const res = await fetch(url, {method: 'GET', mode: 'cors', credentials: 'omit'});
  const contentType = res.headers.get('content-type') || 'unknown';
  const rawText = await res.text();
  const parsed = parseJsonSafe(rawText);

  if (!res.ok || !parsed || typeof parsed !== 'object') {
    throw new Error(
      extractBackendErrorMessage(res.status, contentType, rawText, parsed),
    );
  }
  return parsed as PerformanceResponse;
}

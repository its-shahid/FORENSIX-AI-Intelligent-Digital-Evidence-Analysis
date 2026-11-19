import {
  AlertTriangle,
  CheckCircle2,
  Grid,
  Loader2,
  RefreshCw,
  ShieldAlert,
  Trash2,
  Upload,
  ZoomIn,
} from 'lucide-react';
import {type ChangeEvent, type DragEvent, useEffect, useRef, useState} from 'react';
import {predictImage} from '../services/api';
import type {
  ForensicActivityEntry,
  HealthResponse,
  PredictionResponse,
} from '../types/forensics';

interface DetectViewProps {
  health: HealthResponse | null;
  healthLoading: boolean;
  onRefreshHealth: () => Promise<HealthResponse>;
  onRecordActivity?: (entry: Omit<ForensicActivityEntry, 'id' | 'timestamp'>) => void;
}

interface ScanHistoryEntry {
  id: string;
  previewUrl: string;
  result: PredictionResponse;
  timestamp: string;
}

interface ClientImageMeta {
  width: number;
  height: number;
}

const ACCEPTED_MIME_TYPES = new Set([
  'image/jpeg',
  'image/png',
  'image/webp',
  'image/bmp',
]);

const MAX_BYTES = 20 * 1024 * 1024;

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

export function DetectView({
  health,
  healthLoading,
  onRefreshHealth,
  onRecordActivity,
}: DetectViewProps) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [clientMeta, setClientMeta] = useState<ClientImageMeta | null>(null);
  const [dragActive, setDragActive] = useState(false);

  const [threshold, setThreshold] = useState<number>(0.65);
  const [showGrid, setShowGrid] = useState<boolean>(false);
  const [zoomActual, setZoomActual] = useState<boolean>(false);

  const [analyzing, setAnalyzing] = useState<boolean>(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [checkingHealth, setCheckingHealth] = useState<boolean>(false);
  const [healthCheckStatus, setHealthCheckStatus] = useState<{
    ok: boolean;
    message: string;
  } | null>(null);
  const [history, setHistory] = useState<ScanHistoryEntry[]>([]);

  const historyUrlsRef = useRef<Set<string>>(new Set());

  useEffect(() => {
    const urls = historyUrlsRef.current;
    return () => {
      urls.forEach((url) => URL.revokeObjectURL(url));
      urls.clear();
    };
  }, []);

  const revokeIfUnused = (url: string | null) => {
    if (url && !historyUrlsRef.current.has(url)) {
      URL.revokeObjectURL(url);
    }
  };

  const handleFileSelection = (file: File | undefined | null) => {
    if (!file) return;

    setErrorMsg(null);
    setHealthCheckStatus(null);
    setResult(null);

    if (file.size === 0) {
      setErrorMsg('Selected file is empty (0 bytes).');
      return;
    }

    if (file.size > MAX_BYTES) {
      setErrorMsg(
        `File size (${formatBytes(file.size)}) exceeds the 20 MB upload limit.`,
      );
      return;
    }

    const ext = file.name.split('.').pop()?.toLowerCase() || '';
    const validExt = ['jpg', 'jpeg', 'png', 'webp', 'bmp'].includes(ext);
    if (!ACCEPTED_MIME_TYPES.has(file.type) && !validExt) {
      setErrorMsg(
        `Unsupported file type (${file.type || ext || 'unknown'}). Use JPG, PNG, WEBP, or BMP.`,
      );
      return;
    }

    const objectUrl = URL.createObjectURL(file);
    const probe = new Image();
    probe.onload = () => {
      revokeIfUnused(previewUrl);
      setSelectedFile(file);
      setPreviewUrl(objectUrl);
      setClientMeta({
        width: probe.naturalWidth,
        height: probe.naturalHeight,
      });
      onRecordActivity?.({
        stage: 'upload',
        message: 'Image received for analysis.',
      });
    };
    probe.onerror = () => {
      URL.revokeObjectURL(objectUrl);
      setErrorMsg(
        `Could not decode "${file.name}". The file may be corrupted or not a valid image.`,
      );
    };
    probe.src = objectUrl;
  };

  const handleInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    handleFileSelection(e.target.files?.[0]);
    e.target.value = '';
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    handleFileSelection(e.dataTransfer.files?.[0]);
  };

  const handleClear = () => {
    revokeIfUnused(previewUrl);
    setSelectedFile(null);
    setPreviewUrl(null);
    setClientMeta(null);
    setResult(null);
    setErrorMsg(null);
    setHealthCheckStatus(null);
  };

  const handleDetect = async () => {
    if (!selectedFile || analyzing) return;
    setAnalyzing(true);
    setErrorMsg(null);
    setHealthCheckStatus(null);
    onRecordActivity?.({
      stage: 'preprocess',
      message: 'Image preprocessing completed.',
    });
    onRecordActivity?.({
      stage: 'inference',
      message: 'Deep learning inference initiated.',
    });

    try {
      const response = await predictImage(selectedFile, threshold);
      if (!response || !response.probabilities) {
        throw new Error('Received an incomplete prediction payload from backend.');
      }
      setResult(response);
      onRecordActivity?.({
        stage: 'classification',
        message: `Classification result generated (${response.verdict}).`,
      });
      onRecordActivity?.({
        stage: 'confidence',
        message: `Confidence score calculated (${(response.confidence_percentage ?? response.confidence * 100).toFixed(2)}%).`,
      });
      if (previewUrl) {
        historyUrlsRef.current.add(previewUrl);
        setHistory((prev) => [
          {
            id: `${Date.now()}-${selectedFile.name}`,
            previewUrl,
            result: response,
            timestamp: new Date().toLocaleTimeString(),
          },
          ...prev.slice(0, 7),
        ]);
      }
    } catch (err) {
      setResult(null);
      setErrorMsg(
        err instanceof Error ? err.message : 'Unexpected inference error.',
      );
    } finally {
      setAnalyzing(false);
    }
  };

  const handleCheckHealth = async () => {
    if (checkingHealth) return;
    setCheckingHealth(true);
    try {
      const res = await onRefreshHealth();
      const httpPart = res.http_status ? `HTTP ${res.http_status}` : 'No HTTP response';
      if (res.python_backend_reachable && res.model_loaded) {
        setHealthCheckStatus({
          ok: true,
          message: `Backend healthy (${httpPart}) · status: ${res.status} · model_loaded: true · device: ${res.device || 'cpu'}`,
        });
      } else if (res.python_backend_reachable) {
        setHealthCheckStatus({
          ok: false,
          message: `Backend reachable (${httpPart}), model_loaded: false (${res.model_error || res.status})`,
        });
      } else {
        setHealthCheckStatus({
          ok: false,
          message:
            res.model_error ||
            `Backend health check failed (${httpPart}). Verify the inference service is running.`,
        });
      }
    } catch (err) {
      setHealthCheckStatus({
        ok: false,
        message:
          err instanceof Error
            ? `Health check failed: ${err.message}`
            : 'Health check failed.',
      });
    } finally {
      setCheckingHealth(false);
    }
  };

  const renderVerdictIcon = (verdict: PredictionResponse['verdict']) => {
    if (verdict === 'REAL') {
      return <CheckCircle2 className="h-5 w-5 text-emerald-400 shrink-0" aria-hidden="true" />;
    }
    return <ShieldAlert className="h-5 w-5 text-rose-400 shrink-0" aria-hidden="true" />;
  };

  const verdictToneClass = (verdict: PredictionResponse['verdict']) => {
    if (verdict === 'REAL') return 'border-emerald-800/80 bg-emerald-950/25 text-emerald-300';
    return 'border-rose-800/80 bg-rose-950/25 text-rose-300';
  };

  return (
    <div className="mx-auto max-w-[1360px] px-6 py-8 md:py-10">
      {/* Workspace Telemetry Bar */}
      <div className="flex flex-wrap items-center justify-between gap-4 border border-slate-800 bg-[#0B0E17] px-5 py-3">
        <div className="flex flex-wrap items-center gap-4 text-xs">
          <span className="font-mono text-slate-400">
            Model: <span className="text-slate-200">{health?.architecture || 'EfficientNet-B0'}</span>
          </span>
          <span className="text-slate-700 hidden sm:inline" aria-hidden="true">
            |
          </span>
          <span className="font-mono text-slate-400 hidden sm:inline">
            Mapping: <span className="text-slate-200">fake=0 · real=1</span>
          </span>
        </div>

        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={() => onRefreshHealth()}
            className="inline-flex min-h-[36px] items-center gap-1.5 rounded border border-slate-800 px-3 py-1.5 text-xs font-medium text-slate-300 transition-colors hover:border-slate-700 hover:text-white whitespace-nowrap shrink-0"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${healthLoading ? 'animate-spin' : ''}`} />
            <span>Refresh Status</span>
          </button>
        </div>
      </div>

      {/* Main Asymmetric Split Console */}
      <div className="mt-6 grid grid-cols-1 gap-8 lg:grid-cols-12">
        {/* Left Stage: Upload, Preview & Controls */}
        <div className="flex flex-col border border-slate-800 bg-[#0B0E17] lg:col-span-7">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 px-6 py-4">
            <div>
              <h1 className="font-display text-lg font-semibold text-slate-50">
                Image Inspection Stage
              </h1>
              <p className="text-xs text-slate-400">
                Upload an image to run EfficientNet-B0 inference (JPG, PNG, WEBP, BMP · max 20 MB)
              </p>
            </div>

            {previewUrl && (
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => setShowGrid((prev) => !prev)}
                  className={`inline-flex items-center gap-1.5 rounded border px-2.5 py-1.5 text-xs font-medium transition-colors whitespace-nowrap shrink-0 ${
                    showGrid
                      ? 'border-cyan-500/70 bg-cyan-950/30 text-cyan-300'
                      : 'border-slate-700 text-slate-300 hover:border-slate-600'
                  }`}
                >
                  <Grid className="h-3.5 w-3.5" />
                  <span>224×224 Grid</span>
                </button>
                <button
                  type="button"
                  onClick={() => setZoomActual((prev) => !prev)}
                  className={`inline-flex items-center gap-1.5 rounded border px-2.5 py-1.5 text-xs font-medium transition-colors whitespace-nowrap shrink-0 ${
                    zoomActual
                      ? 'border-cyan-500/70 bg-cyan-950/30 text-cyan-300'
                      : 'border-slate-700 text-slate-300 hover:border-slate-600'
                  }`}
                >
                  <ZoomIn className="h-3.5 w-3.5" />
                  <span>{zoomActual ? 'Fit View' : '100% Scale'}</span>
                </button>
                <button
                  type="button"
                  onClick={handleClear}
                  className="inline-flex items-center gap-1.5 rounded border border-slate-700 px-2.5 py-1.5 text-xs font-medium text-slate-300 hover:border-rose-700 hover:text-rose-300 transition-colors whitespace-nowrap shrink-0"
                >
                  <Trash2 className="h-3.5 w-3.5" />
                  <span>Clear</span>
                </button>
              </div>
            )}
          </div>

          {/* Dropzone / Preview Canvas */}
          <div className="flex-1 p-6">
            <input
              ref={fileInputRef}
              type="file"
              accept=".jpg,.jpeg,.png,.webp,.bmp,image/jpeg,image/png,image/webp,image/bmp"
              onChange={handleInputChange}
              className="sr-only"
              aria-label="Upload image file"
            />

            {!previewUrl ? (
              <div
                onDragEnter={(e) => {
                  e.preventDefault();
                  setDragActive(true);
                }}
                onDragOver={(e) => {
                  e.preventDefault();
                  setDragActive(true);
                }}
                onDragLeave={(e) => {
                  e.preventDefault();
                  setDragActive(false);
                }}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    fileInputRef.current?.click();
                  }
                }}
                className={`flex min-h-[380px] cursor-pointer flex-col items-center justify-center border border-dashed p-8 text-center transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-500 ${
                  dragActive
                    ? 'border-cyan-400 bg-cyan-950/20'
                    : 'border-slate-700 bg-[#07090E] hover:border-slate-500'
                }`}
              >
                <div className="flex h-12 w-12 items-center justify-center rounded-md border border-slate-800 bg-[#0B0E17] text-cyan-400">
                  <Upload className="h-5 w-5" />
                </div>
                <p className="mt-4 font-display text-base font-medium text-slate-100">
                  Drag and drop an image here, or click to select
                </p>
                <p className="mt-1.5 text-xs text-slate-400">
                  Supports JPEG, PNG, WEBP, and BMP · Verified before 224×224 RGB preprocessing
                </p>
                <span className="mt-5 inline-flex items-center rounded border border-slate-700 px-4 py-2 text-xs font-medium text-slate-200">
                  Browse Local Image
                </span>
              </div>
            ) : (
              <div className="space-y-4">
                <div
                  className={`relative flex min-h-[380px] max-h-[520px] items-center justify-center border border-slate-800 bg-[#07090E] ${
                    zoomActual ? 'overflow-auto' : 'overflow-hidden'
                  }`}
                >
                  <img
                    src={previewUrl}
                    alt={selectedFile?.name || 'Uploaded preview'}
                    referrerPolicy="no-referrer"
                    className={
                      zoomActual
                        ? 'max-w-none cursor-crosshair'
                        : 'max-h-[480px] w-auto max-w-full object-contain cursor-crosshair'
                    }
                  />

                  {showGrid && (
                    <div
                      aria-hidden="true"
                      className="pointer-events-none absolute inset-0"
                      style={{
                        backgroundImage:
                          'linear-gradient(to right, rgba(6, 182, 212, 0.2) 1px, transparent 1px), linear-gradient(to bottom, rgba(6, 182, 212, 0.2) 1px, transparent 1px)',
                        backgroundSize: '32px 32px',
                      }}
                    />
                  )}
                </div>

                {/* Unboxed Image Metadata Strip */}
                {selectedFile && (
                  <div className="flex flex-wrap items-center justify-between gap-3 border-t border-slate-800 pt-3 text-xs text-slate-400">
                    <div className="flex flex-wrap items-center gap-2 font-mono">
                      <span className="text-slate-200 truncate max-w-[240px]">
                        {selectedFile.name}
                      </span>
                      <span aria-hidden="true">·</span>
                      <span className="tabular-nums">
                        {clientMeta ? `${clientMeta.width} × ${clientMeta.height} px` : '—'}
                      </span>
                      <span aria-hidden="true">·</span>
                      <span className="tabular-nums">{formatBytes(selectedFile.size)}</span>
                    </div>
                    <button
                      type="button"
                      disabled={analyzing}
                      onClick={() => fileInputRef.current?.click()}
                      className="text-xs font-medium text-cyan-400 hover:text-cyan-300 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                    >
                      Replace Image
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Control Footer */}
          <div className="flex flex-col gap-4 border-t border-slate-800 px-6 py-4 sm:flex-row sm:items-center sm:justify-between">
            <div className="flex flex-wrap items-center gap-6">
              <div className="flex flex-col gap-1.5 sm:min-w-[240px]">
                <div className="flex items-center justify-between text-xs">
                  <label htmlFor="threshold-slider" className="text-slate-300">
                    Low-Confidence Warning Cutoff
                  </label>
                  <span className="font-mono text-cyan-400 tabular-nums">
                    {(threshold * 100).toFixed(0)}%
                  </span>
                </div>
                <input
                  id="threshold-slider"
                  type="range"
                  min={0.55}
                  max={0.9}
                  step={0.05}
                  value={threshold}
                  onChange={(e) => setThreshold(parseFloat(e.target.value))}
                  className="h-1.5 w-full cursor-pointer accent-cyan-500"
                />
              </div>
            </div>

            <button
              type="button"
              disabled={!selectedFile || analyzing}
              onClick={handleDetect}
              className="inline-flex min-h-[44px] items-center justify-center gap-2 rounded-md bg-cyan-500 px-6 py-2.5 text-sm font-semibold text-slate-950 transition-colors hover:bg-cyan-400 disabled:cursor-not-allowed disabled:bg-slate-800 disabled:text-slate-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400 whitespace-nowrap shrink-0"
            >
              {analyzing ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span>Running Inference...</span>
                </>
              ) : (
                <span>Analyze Image</span>
              )}
            </button>
          </div>
        </div>

        {/* Right Stage: Real Prediction Readout */}
        <div className="flex flex-col border border-slate-800 bg-[#0B0E17] lg:col-span-5">
          <div className="flex items-center justify-between border-b border-slate-800 px-6 py-4">
            <h2 className="font-display text-lg font-semibold text-slate-50">
              Classification Output
            </h2>
            {result && (
              <span className="font-mono text-xs text-slate-400 tabular-nums">
                {result.inference_time_ms != null
                  ? `${result.inference_time_ms.toFixed(2)} ms`
                  : result.round_trip_ms != null
                    ? `${result.round_trip_ms.toFixed(0)} ms round-trip`
                    : '—'}
              </span>
            )}
          </div>

          <div className="flex-1 p-6">
            {analyzing && (
              <div
                role="status"
                aria-live="polite"
                className="flex h-full min-h-[300px] flex-col items-center justify-center text-center px-4"
              >
                <Loader2 className="h-7 w-7 animate-spin text-cyan-400" />
                <p className="mt-4 font-display text-base font-medium text-slate-100">
                  Analyzing Image with EfficientNet-B0
                </p>
                <p className="mt-1.5 text-xs leading-relaxed text-slate-400">
                  Uploading via multipart FormData and evaluating FAKE (0) / REAL (1) softmax probabilities...
                </p>
              </div>
            )}

            {!analyzing && errorMsg && (
              <div
                role="alert"
                className="border border-amber-800/80 bg-amber-950/25 p-5 text-sm text-amber-200"
              >
                <div className="flex items-start gap-3">
                  <AlertTriangle className="mt-0.5 h-5 w-5 text-amber-400 shrink-0" />
                  <div className="space-y-3 flex-1">
                    <div>
                      <p className="font-semibold text-amber-200">
                        Inference Could Not Complete
                      </p>
                      <p className="mt-1 text-xs leading-relaxed text-amber-100/90">
                        {errorMsg}
                      </p>
                    </div>

                    <div className="flex flex-wrap items-center gap-2">
                      {selectedFile && (
                        <button
                          type="button"
                          disabled={analyzing}
                          onClick={handleDetect}
                          className="inline-flex min-h-[34px] items-center gap-1.5 rounded bg-amber-500 px-3.5 py-1.5 text-xs font-semibold text-slate-950 transition-colors hover:bg-amber-400 disabled:cursor-not-allowed disabled:opacity-50"
                        >
                          <RefreshCw className="h-3.5 w-3.5" />
                          <span>Retry Analysis</span>
                        </button>
                      )}
                      <button
                        type="button"
                        disabled={checkingHealth}
                        onClick={handleCheckHealth}
                        className="inline-flex min-h-[34px] items-center gap-1.5 rounded border border-amber-700/70 px-3 py-1.5 text-xs font-medium text-amber-200 transition-colors hover:border-amber-500 hover:text-white disabled:cursor-not-allowed disabled:opacity-60"
                      >
                        {checkingHealth && (
                          <Loader2 className="h-3.5 w-3.5 animate-spin" />
                        )}
                        <span>
                          {checkingHealth ? 'Checking /health...' : 'Check Backend Health'}
                        </span>
                      </button>
                    </div>

                    {healthCheckStatus && (
                      <div
                        role="status"
                        className={`rounded border px-3 py-2 font-mono text-[11px] leading-relaxed ${
                          healthCheckStatus.ok
                            ? 'border-emerald-800/80 bg-emerald-950/30 text-emerald-200'
                            : 'border-rose-800/80 bg-rose-950/30 text-rose-200'
                        }`}
                      >
                        {healthCheckStatus.message}
                      </div>
                    )}

                    <div className="border-t border-amber-800/60 pt-3 text-xs text-slate-300 space-y-1.5">
                      <p className="font-medium text-slate-200">
                        Inference backend unavailable or unreachable.
                      </p>
                      <p className="text-[11px] leading-relaxed text-slate-400">
                        Use <span className="text-slate-200">Check Backend Health</span> to verify
                        service status, then click <span className="text-slate-200">Retry Analysis</span>.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {!analyzing && !errorMsg && !result && (
              <div className="flex h-full min-h-[300px] flex-col justify-center text-center px-4">
                <p className="font-display text-base font-medium text-slate-200">
                  {selectedFile
                    ? 'Ready for EfficientNet-B0 inference'
                    : 'No image loaded'}
                </p>
                <p className="mt-2 text-xs leading-relaxed text-slate-400">
                  {selectedFile
                    ? 'Click "Analyze Image" to run EfficientNet-B0 deepfake detection via the connected FastAPI inference backend.'
                    : 'Upload a portrait or scene image on the left to inspect its probabilities across class 0 (fake) and class 1 (real).'}
                </p>
              </div>
            )}

            {!analyzing && result && (
              <div className="space-y-6">
                {/* Primary Verdict Banner */}
                <div className={`border p-5 ${verdictToneClass(result.verdict)}`}>
                  <div className="flex items-center justify-between gap-4">
                    <div className="flex items-center gap-3">
                      {renderVerdictIcon(result.verdict)}
                      <div>
                        <p className="text-xs text-slate-300">Classification Result</p>
                        <div className="mt-1 flex items-center gap-2.5">
                          <span
                            className={`inline-flex items-center rounded border px-2.5 py-0.5 font-mono text-sm font-bold tracking-wider uppercase ${
                              result.verdict === 'REAL'
                                ? 'border-emerald-500/50 bg-emerald-500/20 text-emerald-300'
                                : 'border-rose-500/50 bg-rose-500/20 text-rose-300'
                            }`}
                          >
                            {result.verdict}
                          </span>
                          <span className="font-display text-lg font-semibold text-slate-100">
                            {result.verdict === 'REAL' ? 'Authentic Image' : 'AI-Generated / Fake'}
                          </span>
                        </div>
                      </div>
                    </div>

                    <div className="text-right">
                      <p className="text-xs text-slate-300">Confidence</p>
                      <p className="font-mono text-2xl font-bold tabular-nums">
                        {(
                          result.confidence_percentage ?? result.confidence * 100
                        ).toFixed(2)}
                        %
                      </p>
                    </div>
                  </div>

                  {result.confidence < threshold && (
                    <div
                      role="status"
                      className="mt-3 flex items-center gap-2 border-t border-slate-700/60 pt-2.5 text-xs font-medium text-amber-300"
                    >
                      <AlertTriangle
                        className="h-4 w-4 text-amber-400 shrink-0"
                        aria-hidden="true"
                      />
                      <span>
                        Low confidence – prediction may be unreliable.
                      </span>
                    </div>
                  )}
                </div>

                {/* Probability Breakdown */}
                <div className="space-y-4">
                  <div className="flex items-center justify-between text-xs text-slate-400">
                    <span>Class Probabilities (Softmax)</span>
                    <span className="font-mono tabular-nums">
                      Margin: {(result.margin * 100).toFixed(2)}%
                    </span>
                  </div>

                  {/* FAKE (Class 0) */}
                  <div className="space-y-1.5">
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-medium text-slate-200">
                        FAKE <span className="font-mono text-slate-500">(Class 0)</span>
                      </span>
                      <span className="font-mono font-semibold text-rose-300 tabular-nums">
                        {(result.probabilities.fake * 100).toFixed(2)}%
                      </span>
                    </div>
                    <div className="relative h-2.5 w-full overflow-hidden bg-slate-900 border border-slate-800">
                      <div
                        className="h-full bg-rose-500 transition-transform duration-300 origin-left"
                        style={{
                          width: `${result.probabilities.fake > 0 ? Math.max(result.probabilities.fake * 100, 0.5) : 0}%`,
                        }}
                      />
                    </div>
                  </div>

                  {/* REAL (Class 1) */}
                  <div className="space-y-1.5">
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-medium text-slate-200">
                        REAL <span className="font-mono text-slate-500">(Class 1)</span>
                      </span>
                      <span className="font-mono font-semibold text-emerald-300 tabular-nums">
                        {(result.probabilities.real * 100).toFixed(2)}%
                      </span>
                    </div>
                    <div className="relative h-2.5 w-full overflow-hidden bg-slate-900 border border-slate-800">
                      <div
                        className="h-full bg-emerald-500 transition-transform duration-300 origin-left"
                        style={{
                          width: `${result.probabilities.real > 0 ? Math.max(result.probabilities.real * 100, 0.5) : 0}%`,
                        }}
                      />
                    </div>
                  </div>
                </div>

                {/* Telemetry Readout Table */}
                <div className="divide-y divide-slate-800 border border-slate-800 bg-[#07090E] text-xs">
                  <div className="flex items-center justify-between px-4 py-2.5">
                    <span className="text-slate-400">
                      {result.inference_time_ms != null
                        ? 'Inference Latency'
                        : 'Request Latency'}
                    </span>
                    <span className="font-mono text-slate-200 tabular-nums">
                      {result.inference_time_ms != null
                        ? `${result.inference_time_ms.toFixed(2)} ms`
                        : result.round_trip_ms != null
                          ? `${result.round_trip_ms.toFixed(0)} ms round-trip`
                          : '—'}
                    </span>
                  </div>
                  {result.logits && (
                    <div className="flex items-center justify-between px-4 py-2.5">
                      <span className="text-slate-400">Raw Logits (fake / real)</span>
                      <span className="font-mono text-slate-200 tabular-nums">
                        {result.logits.fake.toFixed(3)} / {result.logits.real.toFixed(3)}
                      </span>
                    </div>
                  )}
                  <div className="flex items-center justify-between px-4 py-2.5">
                    <span className="text-slate-400">Predicted Class Index</span>
                    <span className="font-mono text-slate-200 tabular-nums">
                      class_id = {result.class_id ?? (result.raw_label === 'FAKE' ? 0 : 1)} (
                      {result.predicted_class || result.raw_label.toLowerCase()})
                    </span>
                  </div>
                  <div className="flex items-center justify-between px-4 py-2.5">
                    <span className="text-slate-400">Image Payload</span>
                    <span className="font-mono text-slate-200 tabular-nums">
                      {(
                        result.image_metadata.format ||
                        selectedFile?.type.replace('image/', '').toUpperCase() ||
                        'IMAGE'
                      )}{' '}
                      ·{' '}
                      {(result.image_metadata.width ?? clientMeta?.width) &&
                      (result.image_metadata.height ?? clientMeta?.height)
                        ? `${result.image_metadata.width ?? clientMeta?.width}×${result.image_metadata.height ?? clientMeta?.height}`
                        : formatBytes(result.image_metadata.size_bytes)}
                    </span>
                  </div>
                  <div className="flex items-center justify-between px-4 py-2.5">
                    <span className="text-slate-400">Model Architecture</span>
                    <span className="font-mono text-slate-200">
                      {result.model_info.architecture || 'EfficientNet-B0'}
                    </span>
                  </div>
                </div>

                {/* Grad-CAM Visual Explanation (from model.features[-1]) */}
                {result.gradcam_overlay && (
                  <div className="border border-slate-800 bg-[#07090E] p-4 space-y-2.5">
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-medium text-slate-200">
                        Grad-CAM Saliency Overlay
                      </span>
                      <span className="font-mono text-[11px] text-slate-400">
                        features[-1] · 224×224
                      </span>
                    </div>
                    <img
                      src={result.gradcam_overlay}
                      alt="Grad-CAM saliency overlay highlighting image regions influencing the EfficientNet-B0 prediction"
                      referrerPolicy="no-referrer"
                      className="w-56 h-56 object-contain border border-slate-800"
                    />
                  </div>
                )}

                {/* Honest Disclaimer */}
                <p className="text-xs leading-relaxed text-slate-400">
                  {result.disclaimer}
                </p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Session Inspection History (only real backend inferences from this session) */}
      {history.length > 0 && (
        <section className="mt-10 border border-slate-800 bg-[#0B0E17] p-6">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <h2 className="font-display text-base font-semibold text-slate-100">
              Session Inference Log
            </h2>
            <span className="font-mono text-xs text-slate-400 tabular-nums">
              {history.length} verified run{history.length === 1 ? '' : 's'}
            </span>
          </div>
          <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {history
              .filter((entry) => Boolean(entry?.result?.probabilities))
              .map((entry) => (
                <button
                  key={entry.id}
                  type="button"
                  onClick={() => setResult(entry.result)}
                  className="flex items-center gap-3 border border-slate-800 bg-[#07090E] p-3 text-left transition-colors hover:border-slate-600"
                >
                  <img
                    src={entry.previewUrl}
                    alt={entry.result?.image_metadata?.filename || 'Scanned image'}
                    referrerPolicy="no-referrer"
                    className="h-12 w-12 shrink-0 object-cover border border-slate-800"
                  />
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-xs font-medium text-slate-200">
                      {entry.result?.image_metadata?.filename || 'Uploaded image'}
                    </p>
                    <p className="mt-0.5 font-mono text-xs text-slate-400 tabular-nums">
                      {entry.result?.verdict || '—'} ·{' '}
                      {(
                        entry.result?.confidence_percentage ??
                        (entry.result?.confidence ?? 0) * 100
                      ).toFixed(1)}
                      %
                    </p>
                    <p className="font-mono text-[11px] text-slate-500 tabular-nums">
                      {entry.timestamp}
                    </p>
                  </div>
                </button>
              ))}
          </div>
        </section>
      )}
    </div>
  );
}

export type PageId = 'home' | 'detect' | 'about';

export interface ForensicActivityEntry {
  id: string;
  message: string;
  timestamp: string;
  stage: 'upload' | 'preprocess' | 'inference' | 'classification' | 'confidence';
}

export interface ClassMapping {
  fake: number;
  real: number;
}

export interface HealthResponse {
  status: 'ok' | 'healthy' | 'ready' | 'checkpoint_missing' | 'backend_unavailable' | string;
  python_backend_reachable: boolean;
  model_loaded: boolean;
  model_name?: string;
  model_error?: string | null;
  checkpoint_path: string;
  checkpoint_present_on_disk?: boolean;
  python_backend_url?: string;
  architecture: string;
  class_mapping: ClassMapping;
  image_size: number;
  device?: string;
  low_confidence_threshold?: number;
  uptime_seconds?: number;
  checked_url?: string;
  http_status?: number;
}

export interface PredictionResponse {
  success?: boolean;
  result?: 'REAL' | 'FAKE' | string;
  prediction?: 'REAL' | 'FAKE' | string;
  predicted_class?: string;
  class_id?: number;
  verdict: 'REAL' | 'FAKE';
  raw_label: 'REAL' | 'FAKE';
  confidence: number;
  confidence_percentage?: number;
  low_confidence?: boolean;
  confidence_warning?: string | null;
  threshold?: number;
  margin: number;
  inference_time?: number | null;
  inference_time_ms?: number | null;
  round_trip_ms?: number | null;
  model?: string;
  probabilities: {
    fake: number;
    real: number;
  };
  logits?: {
    fake: number;
    real: number;
  } | null;
  gradcam_overlay?: string | null;
  image_metadata: {
    filename: string;
    format?: string;
    width?: number;
    height?: number;
    mode?: string;
    size_bytes: number;
  };
  model_info: {
    architecture: string;
    checkpoint: string;
    device?: string;
    image_size: number;
    class_mapping: ClassMapping;
    low_confidence_threshold: number;
  };
  disclaimer: string;
}

export interface PerClassMetrics {
  class_id: number;
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export interface PerformanceResponse {
  evaluated: boolean;
  message?: string;
  expected_metrics_path?: string;
  confusion_matrix_image_available?: boolean;
  confusion_matrix_image_url?: string | null;
  architecture?: string;
  checkpoint?: string;
  dataset?: string;
  class_mapping?: ClassMapping;
  evaluated_at?: string;
  total_test_available?: number;
  evaluated_samples?: number;
  accuracy?: number;
  precision_macro?: number;
  recall_macro?: number;
  f1_macro?: number;
  roc_auc?: number | null;
  per_class?: {
    fake: PerClassMetrics;
    real: PerClassMetrics;
  };
  confusion_matrix?: number[][];
  avg_confidence?: number;
  avg_inference_ms_per_image?: number;
  brier_score?: number | null;
  threshold_diagnostics?: {
    threshold: number;
    low_confidence_count?: number;
    low_confidence_rate?: number;
    decided_accuracy: number | null;
    near_fifty_rate: number;
    mean_confidence_correct: number | null;
    mean_confidence_incorrect: number | null;
    inverted_labels_suspected: boolean;
  };
  device?: string;
}

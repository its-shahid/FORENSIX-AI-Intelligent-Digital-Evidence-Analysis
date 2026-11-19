import fs from 'fs';
import http from 'http';
import https from 'https';
import path from 'path';
import {fileURLToPath} from 'url';
import dotenv from 'dotenv';
import express from 'express';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT_DIR = __dirname;
const PORT = Number(process.env.PORT || 3000);
const PYTHON_BACKEND_URL = (
  process.env.PYTHON_BACKEND_URL ||
  process.env.VITE_PYTHON_BACKEND_URL ||
  'https://karaoke-collectors-look-keith.trycloudflare.com'
)
  .trim()
  .replace(/\/+$/, '');

const RESULTS_DIR = path.join(ROOT_DIR, 'results');
const METRICS_PATH = path.join(RESULTS_DIR, 'evaluation_metrics.json');
const CONFUSION_MATRIX_PATH = path.join(RESULTS_DIR, 'confusion_matrix.png');
const MODEL_PATH = path.join(ROOT_DIR, 'models', 'best_model.pth');
const MAX_UPLOAD_BYTES = 25 * 1024 * 1024;

function readRequestBody(req: express.Request): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let total = 0;

    req.on('data', (chunk: Buffer) => {
      total += chunk.length;
      if (total > MAX_UPLOAD_BYTES) {
        reject(new Error('Request payload exceeds 25 MB limit.'));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });

    req.on('end', () => {
      resolve(Buffer.concat(chunks));
    });

    req.on('error', (err) => {
      reject(err);
    });
  });
}

async function fetchJsonFromBackend(
  targetPath: string,
): Promise<{status: number; data: Record<string, unknown> | null}> {
  if (!PYTHON_BACKEND_URL) {
    return {status: 503, data: null};
  }
  const targetUrl = `${PYTHON_BACKEND_URL}${targetPath}`;
  const res = await fetch(targetUrl, {
    method: 'GET',
    headers: {
      Accept: 'application/json',
    },
    signal: AbortSignal.timeout(15000),
  });
  const text = await res.text();
  try {
    const parsed = JSON.parse(text) as Record<string, unknown>;
    return {status: res.status, data: parsed};
  } catch {
    return {status: res.status, data: null};
  }
}

async function forwardToPython(
  req: express.Request,
  res: express.Response,
  targetPath: string,
  onUnavailable: (reason?: string) => void,
): Promise<void> {
  if (!PYTHON_BACKEND_URL) {
    onUnavailable('PYTHON_BACKEND_URL / VITE_PYTHON_BACKEND_URL is not configured');
    return;
  }
  const targetUrl = new URL(`${PYTHON_BACKEND_URL}${targetPath}`);
  const client = targetUrl.protocol === 'https:' ? https : http;

  let bodyBuffer: Buffer | null = null;
  if (req.method !== 'GET' && req.method !== 'HEAD') {
    try {
      bodyBuffer = await readRequestBody(req);
    } catch (err) {
      if (!res.headersSent) {
        res.status(413).json({
          detail: {
            error: 'Invalid or oversized request payload.',
            message: err instanceof Error ? err.message : 'Failed to read upload stream.',
          },
        });
      }
      return;
    }
  }

  const headers: http.OutgoingHttpHeaders = {};
  if (req.headers['content-type']) {
    headers['content-type'] = req.headers['content-type'];
  }
  if (bodyBuffer !== null) {
    headers['content-length'] = String(bodyBuffer.length);
  }
  if (req.headers['accept']) {
    headers['accept'] = req.headers['accept'];
  }

  const proxyReq = client.request(
    {
      hostname: targetUrl.hostname,
      port: targetUrl.port || (targetUrl.protocol === 'https:' ? 443 : 80),
      path: `${targetUrl.pathname}${targetUrl.search}`,
      method: req.method,
      headers,
      timeout: 60000,
    },
    (proxyRes) => {
      res.status(proxyRes.statusCode || 502);
      Object.entries(proxyRes.headers).forEach(([key, value]) => {
        if (value !== undefined && key.toLowerCase() !== 'transfer-encoding') {
          res.setHeader(key, value);
        }
      });
      proxyRes.pipe(res);
    },
  );

  proxyReq.on('timeout', () => {
    proxyReq.destroy(new Error('Python backend request timed out after 60s.'));
  });

  proxyReq.on('error', (err) => {
    if (!res.headersSent) {
      onUnavailable(err.message);
    }
  });

  if (bodyBuffer !== null) {
    proxyReq.write(bodyBuffer);
  }
  proxyReq.end();
}

async function startServer() {
  const app = express();

  const handleHealth = async (_req: express.Request, res: express.Response) => {
    if (PYTHON_BACKEND_URL) {
      try {
        let resp = await fetchJsonFromBackend('/health');
        if (resp.status === 404) {
          resp = await fetchJsonFromBackend('/api/health');
        }

        if (resp.status >= 200 && resp.status < 300 && resp.data) {
          res.status(200).json({
            status: resp.data.status || 'ok',
            python_backend_reachable: true,
            model_loaded: Boolean(resp.data.model_loaded),
            model_name: resp.data.model_name || 'EfficientNet-B0',
            checkpoint_path: resp.data.checkpoint_path || 'models/best_model.pth',
            python_backend_url: PYTHON_BACKEND_URL,
            architecture: resp.data.architecture || 'EfficientNet-B0',
            class_mapping: resp.data.class_mapping || {fake: 0, real: 1},
            image_size: resp.data.image_size || 224,
            device: resp.data.device || 'cpu',
            model_error: resp.data.model_error || null,
          });
          return;
        }
      } catch {
        // Fall through to unavailable status
      }
    }

    const checkpointExists = fs.existsSync(MODEL_PATH);
    res.status(200).json({
      status: PYTHON_BACKEND_URL ? 'backend_unavailable' : 'backend_unconfigured',
      python_backend_reachable: false,
      model_loaded: false,
      checkpoint_present_on_disk: checkpointExists,
      checkpoint_path: 'models/best_model.pth',
      python_backend_url: PYTHON_BACKEND_URL || null,
      architecture: 'EfficientNet-B0',
      class_mapping: {fake: 0, real: 1},
      image_size: 224,
      model_error: PYTHON_BACKEND_URL
        ? `Python FastAPI server is unreachable at ${PYTHON_BACKEND_URL}.`
        : 'No cloud FastAPI backend URL is configured. Set VITE_PYTHON_BACKEND_URL (or PYTHON_BACKEND_URL) to your Render HTTPS URL, or click Backend Endpoint above.',
    });
  };

  app.get('/health', handleHealth);
  app.get('/api/health', handleHealth);

  app.get('/api/model-info', (req, res) => {
    void forwardToPython(req, res, '/api/model-info', () => {
      res.status(200).json({
        model_loaded: false,
        python_backend_reachable: false,
        architecture: 'EfficientNet-B0 (torchvision.models.efficientnet_b0)',
        classifier_head: 'nn.Linear(1280, 2)',
        checkpoint_path: 'models/best_model.pth',
        class_mapping: {fake: 0, real: 1},
        preprocessing: {
          resize: [224, 224],
          color_mode: 'RGB',
          normalize_mean: [0.485, 0.456, 0.406],
          normalize_std: [0.229, 0.224, 0.225],
        },
        device: 'unavailable',
        low_confidence_threshold: 0.65,
      });
    });
  });

  app.post('/api/predict', (req, res) => {
    const querySuffix = req.originalUrl.includes('?')
      ? req.originalUrl.slice(req.originalUrl.indexOf('?'))
      : '';
    void forwardToPython(req, res, `/api/predict${querySuffix}`, (reason) => {
      res.status(503).json({
        detail: {
          error: 'Python inference backend is unavailable.',
          message: PYTHON_BACKEND_URL
            ? `Could not connect to FastAPI server at ${PYTHON_BACKEND_URL}${reason ? ` (${reason})` : ''}.`
            : 'No FastAPI backend URL configured. Set VITE_PYTHON_BACKEND_URL / PYTHON_BACKEND_URL to your Render HTTPS service URL.',
          checkpoint_path: 'models/best_model.pth',
        },
      });
    });
  });

  app.get('/api/performance', async (_req, res) => {
    try {
      const resp = await fetchJsonFromBackend('/api/performance');
      if (resp.status >= 200 && resp.status < 300 && resp.data) {
        res.status(200).json(resp.data);
        return;
      }
    } catch {
      // Fall through to local results check
    }

    const hasCmImage = fs.existsSync(CONFUSION_MATRIX_PATH);
    if (fs.existsSync(METRICS_PATH)) {
      try {
        const parsed = JSON.parse(fs.readFileSync(METRICS_PATH, 'utf-8'));
        res.status(200).json({
          ...parsed,
          evaluated: true,
          confusion_matrix_image_available: hasCmImage,
          confusion_matrix_image_url: hasCmImage ? '/api/results/confusion_matrix.png' : null,
        });
        return;
      } catch {
        // Fall through to un-evaluated message
      }
    }

    res.status(200).json({
      evaluated: false,
      confusion_matrix_image_available: hasCmImage,
      confusion_matrix_image_url: hasCmImage ? '/api/results/confusion_matrix.png' : null,
      expected_metrics_path: 'results/evaluation_metrics.json',
      message:
        'No saved evaluation results found yet. Run `python scripts/evaluate_model.py` against your models/best_model.pth and Dataset/test to generate real metrics.',
    });
  });

  app.get('/api/results/confusion_matrix.png', (req, res) => {
    void forwardToPython(req, res, '/api/results/confusion_matrix.png', () => {
      if (fs.existsSync(CONFUSION_MATRIX_PATH)) {
        res.sendFile(CONFUSION_MATRIX_PATH);
        return;
      }
      res.status(404).json({
        detail: 'results/confusion_matrix.png has not been generated yet.',
      });
    });
  });

  // Never allow /api/* routes to fall through to Vite or static index.html
  app.all('/api/*', (req, res) => {
    res.status(404).json({
      detail: {
        error: 'API endpoint not found.',
        message: `No handler registered for ${req.method} ${req.originalUrl}.`,
      },
    });
  });

  if (process.env.NODE_ENV === 'production') {
    const distDir = path.join(ROOT_DIR, 'dist');
    app.use(express.static(distDir));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });
  } else {
    const {createServer: createViteServer} = await import('vite');
    const vite = await createViteServer({
      server: {middlewareMode: true},
      appType: 'spa',
    });
    app.use(vite.middlewares);
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Web server listening on http://0.0.0.0:${PORT}`);
    console.log(
      PYTHON_BACKEND_URL
        ? `Forwarding inference API to ${PYTHON_BACKEND_URL}`
        : 'No PYTHON_BACKEND_URL configured; configure VITE_PYTHON_BACKEND_URL or PYTHON_BACKEND_URL for cloud inference.',
    );
  });
}

startServer();

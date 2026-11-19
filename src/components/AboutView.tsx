import {
  ArrowDown,
  ArrowRight,
  CheckCircle2,
  Cpu,
  Eye,
  Gauge,
  Layers,
  MonitorSmartphone,
  ShieldAlert,
  ShieldCheck,
  SlidersHorizontal,
  Upload,
} from 'lucide-react';

const OBJECTIVES = [
  'Detect AI-generated and manipulated images.',
  'Analyze spatial image patterns using deep learning.',
  'Classify uploaded images as REAL or FAKE.',
  'Provide transparent prediction confidence scores.',
  'Offer an accessible digital forensic analysis interface.',
];

const WORKFLOW_STEPS = [
  {label: 'Image Upload', icon: Upload, tone: 'text-cyan-400'},
  {label: 'Preprocessing', icon: SlidersHorizontal, tone: 'text-cyan-300'},
  {label: 'Deep Learning Analysis', icon: Cpu, tone: 'text-violet-400'},
  {label: 'REAL / FAKE Classification', icon: ShieldCheck, tone: 'text-cyan-400'},
  {label: 'Confidence Score', icon: Gauge, tone: 'text-violet-300'},
];

const KEY_FEATURES = [
  {
    title: 'Image Upload and Preview',
    description:
      'Drag-and-drop or local selection for JPG, PNG, WEBP, and BMP images up to 20 MB with dimension checks, 100% zoom, and a 224×224 inspection grid.',
    icon: Eye,
  },
  {
    title: 'AI-Based Image Detection',
    description:
      'Sends uploaded images via multipart FormData to a Python FastAPI inference service running a trained PyTorch EfficientNet-B0 model.',
    icon: Cpu,
  },
  {
    title: 'REAL / FAKE Classification',
    description:
      'Evaluates binary class logits and assigns REAL when P(REAL) > P(FAKE), or FAKE when P(FAKE) ≥ P(REAL).',
    icon: ShieldCheck,
  },
  {
    title: 'Confidence Score',
    description:
      'Displays the winning class probability as a percentage alongside an adjustable threshold slider (default 65%) for low-confidence warnings.',
    icon: Gauge,
  },
  {
    title: 'Prediction Results',
    description:
      'Presents color-coded REAL or FAKE badges, class probability bars, raw logits, inference latency, image metadata, and session scan history.',
    icon: Layers,
  },
  {
    title: 'Responsive Interface',
    description:
      'Built with Tailwind CSS in a dark forensic theme that adapts cleanly across desktop, tablet, and mobile viewports.',
    icon: MonitorSmartphone,
  },
];

const TECH_STACK = [
  {
    category: 'Programming Languages',
    value: 'Python 3.10 · TypeScript',
  },
  {
    category: 'Frontend Technologies',
    value: 'React 19 · Vite · Tailwind CSS · Lucide Icons',
  },
  {
    category: 'Backend Framework',
    value: 'FastAPI · Uvicorn · Express',
  },
  {
    category: 'Deep Learning Framework',
    value: 'PyTorch · torchvision',
  },
  {
    category: 'Model Architecture',
    value: 'EfficientNet-B0 · nn.Linear(1280, 2)',
  },
  {
    category: 'Image Processing',
    value: 'Pillow (PIL) · NumPy · EXIF Normalization',
  },
  {
    category: 'Deployment & Hosting',
    value: 'Docker · Render · Node.js Web Server',
  },
];

const MODEL_SPECS = [
  {
    label: 'Model Architecture',
    value: 'EfficientNet-B0 (torchvision)',
  },
  {
    label: 'Input Image Dimensions',
    value: '224 × 224 px · RGB (3 channels)',
  },
  {
    label: 'Classification Classes',
    value: 'Class 0 = FAKE · Class 1 = REAL',
  },
  {
    label: 'Preprocessing Method',
    value: 'EXIF Transpose · RGB · Resize(224×224) · ImageNet Mean/Std',
  },
  {
    label: 'Normalization Parameters',
    value: 'Mean [0.485, 0.456, 0.406] · Std [0.229, 0.224, 0.225]',
  },
];

export function AboutView() {
  return (
    <div className="mx-auto max-w-[1360px] px-6 py-8 md:py-10 space-y-10">
      {/* A. About FORENSIX AI */}
      <section className="border border-slate-800 bg-[#0B0E17] p-6 md:p-8">
        <p className="font-mono text-xs font-semibold uppercase tracking-[0.18em] text-cyan-400">
          AI-POWERED DEEPFAKE IMAGE DETECTION
        </p>
        <h1 className="mt-2 font-display text-2xl font-bold uppercase tracking-wider text-white sm:text-3xl">
          About FORENSIX <span className="text-cyan-400">AI</span>
        </h1>
        <p className="mt-1 font-display text-sm font-medium text-violet-300 sm:text-base">
          Intelligent Digital Evidence Analysis
        </p>
        <p className="mt-4 max-w-[78ch] text-sm leading-relaxed text-slate-300">
          FORENSIX AI is a deepfake image detection platform that analyzes digital images to
          distinguish authentic photographs from AI-generated or manipulated visuals. It combines a
          trained PyTorch EfficientNet-B0 classifier served via FastAPI with an interactive web
          interface to deliver binary REAL or FAKE predictions and softmax confidence scores.
        </p>
      </section>

      {/* B. Problem Statement & C. Project Objectives */}
      <section className="grid grid-cols-1 gap-6 lg:grid-cols-12">
        <div className="border border-slate-800 bg-[#0B0E17] p-6 lg:col-span-6 space-y-3">
          <div className="flex items-center gap-2.5 border-b border-slate-800 pb-3">
            <ShieldAlert className="h-4 w-4 text-violet-400 shrink-0" aria-hidden="true" />
            <h2 className="font-display text-lg font-semibold text-white">
              Problem Statement
            </h2>
          </div>
          <p className="text-sm leading-relaxed text-slate-300">
            Advances in generative AI and image synthesis have made fabricated faces and manipulated
            photographs nearly indistinguishable to the human eye. As synthetic media spreads across
            digital channels, relying on manual visual inspection alone increases the risk of
            misinformation, impersonation, and compromised visual evidence.
          </p>
        </div>

        <div className="border border-slate-800 bg-[#0B0E17] p-6 lg:col-span-6 space-y-3">
          <div className="flex items-center gap-2.5 border-b border-slate-800 pb-3">
            <CheckCircle2 className="h-4 w-4 text-cyan-400 shrink-0" aria-hidden="true" />
            <h2 className="font-display text-lg font-semibold text-white">
              Project Objectives
            </h2>
          </div>
          <ul className="space-y-2 text-sm text-slate-300">
            {OBJECTIVES.map((item) => (
              <li key={item} className="flex items-start gap-2.5">
                <span
                  aria-hidden="true"
                  className="mt-2 h-1.5 w-1.5 rounded-full bg-cyan-400 shrink-0"
                />
                <span>{item}</span>
              </li>
            ))}
          </ul>
        </div>
      </section>

      {/* E. How the System Works */}
      <section aria-labelledby="about-workflow-heading" className="space-y-4">
        <div className="border-b border-slate-800 pb-3">
          <h2
            id="about-workflow-heading"
            className="font-display text-xl font-semibold text-white"
          >
            How the System Works
          </h2>
        </div>

        <div className="border border-slate-800 bg-[#0B0E17] p-5">
          <div className="flex flex-col items-stretch justify-between gap-2 md:flex-row md:items-center">
            {WORKFLOW_STEPS.map((step, index) => {
              const Icon = step.icon;
              return (
                <div
                  key={step.label}
                  className="flex flex-col items-center gap-2 md:flex-1 md:flex-row"
                >
                  <div className="flex w-full items-center gap-2.5 border border-slate-800 bg-[#07090E] px-3.5 py-3 md:flex-1">
                    <Icon className={`h-4 w-4 shrink-0 ${step.tone}`} aria-hidden="true" />
                    <span className="text-xs font-medium text-slate-100 leading-snug">
                      {step.label}
                    </span>
                  </div>
                  {index < WORKFLOW_STEPS.length - 1 && (
                    <>
                      <ArrowRight
                        className="hidden h-4 w-4 shrink-0 text-cyan-500/80 md:block"
                        aria-hidden="true"
                      />
                      <ArrowDown
                        className="h-4 w-4 shrink-0 text-cyan-500/80 md:hidden"
                        aria-hidden="true"
                      />
                    </>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* F. Key Features */}
      <section aria-labelledby="about-features-heading" className="space-y-4">
        <div className="border-b border-slate-800 pb-3">
          <h2
            id="about-features-heading"
            className="font-display text-xl font-semibold text-white"
          >
            Key Features
          </h2>
        </div>

        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {KEY_FEATURES.map((feature) => {
            const Icon = feature.icon;
            return (
              <div
                key={feature.title}
                className="border border-slate-800 bg-[#0B0E17] p-5 transition-colors hover:border-slate-700"
              >
                <div className="flex items-center gap-2.5">
                  <div className="flex h-8 w-8 items-center justify-center rounded border border-slate-800 bg-[#07090E] text-cyan-400 shrink-0">
                    <Icon className="h-4 w-4" aria-hidden="true" />
                  </div>
                  <h3 className="font-display text-sm font-semibold text-white">
                    {feature.title}
                  </h3>
                </div>
                <p className="mt-3 text-xs leading-relaxed text-slate-400">
                  {feature.description}
                </p>
              </div>
            );
          })}
        </div>
      </section>

      {/* D. Technology Stack & G. Model Information */}
      <section className="grid grid-cols-1 gap-6 lg:grid-cols-12">
        <div className="border border-slate-800 bg-[#0B0E17] p-6 lg:col-span-6 space-y-4">
          <h2 className="font-display text-lg font-semibold text-white">
            Technology Stack
          </h2>
          <div className="divide-y divide-slate-800 border border-slate-800 bg-[#07090E] text-xs">
            {TECH_STACK.map((row) => (
              <div
                key={row.category}
                className="flex flex-col gap-1 px-4 py-2.5 sm:flex-row sm:items-center sm:justify-between"
              >
                <span className="text-slate-400">{row.category}</span>
                <span className="font-mono text-slate-200 sm:text-right">{row.value}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="border border-slate-800 bg-[#0B0E17] p-6 lg:col-span-6 space-y-4">
          <h2 className="font-display text-lg font-semibold text-white">
            Model Information
          </h2>
          <div className="divide-y divide-slate-800 border border-slate-800 bg-[#07090E] text-xs">
            {MODEL_SPECS.map((spec) => (
              <div
                key={spec.label}
                className="flex flex-col gap-1 px-4 py-2.5 sm:flex-row sm:items-center sm:justify-between"
              >
                <span className="text-slate-400">{spec.label}</span>
                <span className="font-mono text-slate-200 sm:text-right">{spec.value}</span>
              </div>
            ))}
          </div>
          <p className="text-xs leading-relaxed text-slate-400">
            During inference, raw model logits pass through a softmax layer to produce{' '}
            <code className="font-mono text-rose-300">P(FAKE)</code> and{' '}
            <code className="font-mono text-emerald-300">P(REAL)</code>. Images are classified as{' '}
            <strong className="text-emerald-300">REAL</strong> if{' '}
            <code className="font-mono text-slate-200">P(REAL) &gt; P(FAKE)</code> and{' '}
            <strong className="text-rose-300">FAKE</strong> otherwise, with the winning probability
            reported as confidence and a low-confidence warning displayed below the selected cutoff.
          </p>
        </div>
      </section>

      {/* H. Project Significance */}
      <section className="border border-slate-800 bg-[#0B0E17] p-6 space-y-3">
        <h2 className="font-display text-lg font-semibold text-white">
          Project Significance
        </h2>
        <p className="text-sm leading-relaxed text-slate-300">
          AI-assisted image forensics supports digital content verification, media authenticity
          screening, and public awareness of synthetic visual manipulation. By combining automated
          deep learning inference with transparent probability breakdowns, FORENSIX AI offers a
          practical screening aid for evaluating suspicious imagery.
        </p>
        <p className="border-l-2 border-violet-500/70 bg-[#07090E] px-4 py-3 text-xs leading-relaxed text-slate-300">
          <strong className="text-violet-300">Analytical Disclaimer:</strong> Predictions are
          probabilistic outputs generated by a neural network trained on specific image distributions
          and must not be treated as definitive legal or forensic proof.
        </p>
      </section>
    </div>
  );
}

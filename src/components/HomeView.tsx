import {
  Activity,
  ArrowDown,
  ArrowRight,
  CheckCircle2,
  Cpu,
  Gauge,
  ShieldCheck,
  SlidersHorizontal,
  Upload,
} from 'lucide-react';
import {useState} from 'react';
import activityVisualImg from '../assets/images/forensic_activity_visual_1795081681605.jpg';
import heroBannerImg from '../assets/images/forensix_hero_banner_1795115606174.jpg';
import type {ForensicActivityEntry, PageId} from '../types/forensics';

interface HomeViewProps {
  activityLog: ForensicActivityEntry[];
  onNavigate: (page: PageId) => void;
}

const ARCHITECTURE_FLOW = [
  {label: 'Image Upload', icon: Upload, tone: 'text-cyan-400'},
  {label: 'Image Preprocessing', icon: SlidersHorizontal, tone: 'text-cyan-300'},
  {label: 'AI Model Analysis', icon: Cpu, tone: 'text-violet-400'},
  {label: 'REAL / FAKE Prediction', icon: ShieldCheck, tone: 'text-cyan-400'},
  {label: 'Confidence Score', icon: Gauge, tone: 'text-violet-300'},
];

const DETECTION_STEPS = [
  {
    number: '01',
    title: 'Image Upload',
    description: 'Upload an image for forensic analysis.',
    icon: Upload,
  },
  {
    number: '02',
    title: 'Preprocessing',
    description: 'The image is resized and normalized.',
    icon: SlidersHorizontal,
  },
  {
    number: '03',
    title: 'AI Analysis',
    description: 'The trained deep learning model examines image patterns.',
    icon: Cpu,
  },
  {
    number: '04',
    title: 'Prediction',
    description: 'Displays REAL or FAKE with a confidence percentage.',
    icon: CheckCircle2,
  },
];

const SAMPLE_ACTIVITY_ENTRIES: Array<{
  id: string;
  stage: string;
  message: string;
}> = [
  {
    id: 'sample-1',
    stage: 'Sample Step 01',
    message: 'Image received for analysis.',
  },
  {
    id: 'sample-2',
    stage: 'Sample Step 02',
    message: 'Image preprocessing completed.',
  },
  {
    id: 'sample-3',
    stage: 'Sample Step 03',
    message: 'Deep learning inference initiated.',
  },
  {
    id: 'sample-4',
    stage: 'Sample Step 04',
    message: 'Classification result generated.',
  },
  {
    id: 'sample-5',
    stage: 'Sample Step 05',
    message: 'Confidence score calculated.',
  },
];

export function HomeView({activityLog, onNavigate}: HomeViewProps) {
  const [heroImgFailed, setHeroImgFailed] = useState(false);
  const [activityImgFailed, setActivityImgFailed] = useState(false);

  const hasLiveActivity = activityLog.length > 0;

  return (
    <div className="mx-auto max-w-[1360px] px-6 py-10 md:py-14 space-y-14">
      {/* 1. Hero Banner Section */}
      <section className="relative overflow-hidden border border-slate-800 bg-[#07090E]">
        {/* Background Visual: Integrated into banner on desktop, positioned cleanly on mobile */}
        <div className="relative h-56 w-full overflow-hidden md:absolute md:inset-0 md:h-full">
          {!heroImgFailed ? (
            <img
              src={heroBannerImg}
              alt="Futuristic AI deepfake forensic face scan with glowing cyan wireframe mesh and neural network patterns"
              referrerPolicy="no-referrer"
              onError={() => setHeroImgFailed(true)}
              className="h-full w-full object-cover object-[78%_22%] md:object-right"
            />
          ) : (
            <div
              aria-hidden="true"
              className="h-full w-full bg-gradient-to-br from-[#07090E] via-[#0B1221] to-[#131129]"
            />
          )}

          {/* Directional contrast scrims for sharp readability */}
          <div
            aria-hidden="true"
            className="pointer-events-none absolute inset-0 bg-gradient-to-t from-[#07090E] via-[#07090E]/25 to-transparent md:bg-gradient-to-r md:from-[#07090E] md:via-[#07090E]/85 md:to-transparent"
          />
          <div
            aria-hidden="true"
            className="pointer-events-none hidden md:block absolute inset-0 bg-gradient-to-t from-[#07090E]/50 via-transparent to-[#07090E]/30"
          />
        </div>

        {/* Left-aligned Foreground HTML Content */}
        <div className="relative z-10 max-w-2xl px-6 pb-7 pt-2 text-left sm:px-8 sm:pb-9 md:px-12 md:py-12">
          <p className="font-mono text-xs font-semibold uppercase tracking-[0.18em] text-cyan-400">
            AI-POWERED DEEPFAKE IMAGE DETECTION
          </p>

          <h1 className="mt-2 font-display text-3xl font-bold uppercase tracking-wider text-white sm:text-4xl md:text-5xl leading-tight">
            FORENSIX <span className="text-cyan-400">AI</span>
          </h1>

          <p className="mt-1.5 inline-block bg-gradient-to-r from-violet-400 to-cyan-400 bg-clip-text font-display text-base font-medium text-transparent sm:text-lg md:text-xl">
            Intelligent Digital Evidence Analysis
          </p>

          <p className="mt-3.5 max-w-[58ch] text-sm leading-relaxed text-white/90 sm:text-[15px]">
            FORENSIX AI is an intelligent digital forensics platform designed to investigate image
            authenticity and identify AI-generated or manipulated visuals using deep learning and
            computer vision.
          </p>

          <div className="mt-5">
            <button
              type="button"
              onClick={() => onNavigate('detect')}
              className="inline-flex min-h-[42px] items-center gap-2 rounded-md bg-cyan-500 px-5 py-2.5 text-sm font-semibold text-slate-950 transition-colors hover:bg-cyan-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400 whitespace-nowrap shrink-0"
            >
              <span>Launch Image Detector</span>
              <ArrowRight className="h-4 w-4" aria-hidden="true" />
            </button>
          </div>
        </div>
      </section>

      {/* 2. Inference Execution Architecture (Single-Line Flow) */}
      <section aria-labelledby="architecture-heading" className="space-y-5">
        <div className="border-b border-slate-800 pb-3">
          <h2
            id="architecture-heading"
            className="font-display text-2xl font-semibold text-white"
          >
            Inference Execution Architecture
          </h2>
        </div>

        <div className="border border-slate-800 bg-[#0B0E17] p-5 md:p-6">
          <div className="flex flex-col items-stretch justify-between gap-2 md:flex-row md:items-center">
            {ARCHITECTURE_FLOW.map((node, index) => {
              const Icon = node.icon;
              return (
                <div
                  key={node.label}
                  className="flex flex-col items-center gap-2 md:flex-1 md:flex-row"
                >
                  <div className="flex w-full items-center gap-2.5 border border-slate-800 bg-[#07090E] px-3.5 py-3 transition-colors hover:border-cyan-500/40 md:flex-1">
                    <Icon className={`h-4 w-4 shrink-0 ${node.tone}`} aria-hidden="true" />
                    <span className="text-xs font-medium text-slate-100 leading-snug">
                      {node.label}
                    </span>
                  </div>

                  {index < ARCHITECTURE_FLOW.length - 1 && (
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

      {/* 3. How Detection Works (4 Compact Steps) */}
      <section aria-labelledby="how-detection-heading" className="space-y-5">
        <div className="flex flex-col gap-2 border-b border-slate-800 pb-3 sm:flex-row sm:items-end sm:justify-between">
          <h2
            id="how-detection-heading"
            className="font-display text-2xl font-semibold text-white"
          >
            How Detection Works
          </h2>
          <button
            type="button"
            onClick={() => onNavigate('detect')}
            className="text-sm font-medium text-cyan-400 transition-colors hover:text-cyan-300 whitespace-nowrap shrink-0 text-left"
          >
            Analyze an image →
          </button>
        </div>

        <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
          {DETECTION_STEPS.map((step) => {
            const Icon = step.icon;
            return (
              <div
                key={step.number}
                className="flex flex-col justify-between border border-slate-800 bg-[#0B0E17] p-5 transition-colors hover:border-slate-700"
              >
                <div>
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-xs font-medium text-violet-300 tabular-nums">
                      Step {step.number}
                    </span>
                    <Icon className="h-4 w-4 text-cyan-400" aria-hidden="true" />
                  </div>
                  <h3 className="mt-3 font-display text-base font-semibold text-white">
                    {step.title}
                  </h3>
                  <p className="mt-1.5 text-sm leading-relaxed text-slate-400">
                    {step.description}
                  </p>
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* 4. Digital Forensic Activity */}
      <section aria-labelledby="activity-heading" className="space-y-5">
        <div className="border-b border-slate-800 pb-3">
          <h2
            id="activity-heading"
            className="font-display text-2xl font-semibold text-white"
          >
            Digital Forensic Activity
          </h2>
        </div>

        <div className="grid grid-cols-1 items-stretch gap-8 lg:grid-cols-12">
          {/* Forensic Investigation Visual */}
          <div className="relative overflow-hidden border border-slate-800 bg-[#0B0E17] lg:col-span-5 min-h-[260px]">
            {!activityImgFailed ? (
              <img
                src={activityVisualImg}
                alt="Layered digital forensic pixel authenticity and frequency spectrum analysis visual"
                referrerPolicy="no-referrer"
                onError={() => setActivityImgFailed(true)}
                className="h-full w-full object-cover"
              />
            ) : (
              <div className="flex h-full min-h-[260px] flex-col items-center justify-center bg-gradient-to-br from-[#0B0E17] via-[#0F1629] to-[#17112B] p-8 text-center">
                <Activity className="h-10 w-10 text-violet-400" aria-hidden="true" />
                <p className="mt-3 font-display text-sm font-semibold text-slate-200">
                  Digital Forensic Workflow
                </p>
              </div>
            )}
            <div
              aria-hidden="true"
              className="pointer-events-none absolute inset-0 bg-gradient-to-t from-[#07090E]/60 via-transparent to-transparent"
            />
          </div>

          {/* Compact Activity Log */}
          <div className="flex flex-col justify-between border border-slate-800 bg-[#0B0E17] p-6 lg:col-span-7">
            <div>
              <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 pb-3">
                <span className="font-display text-sm font-semibold text-white">
                  {hasLiveActivity
                    ? 'Session Analysis Activity'
                    : 'Illustrative Sample Workflow'}
                </span>
                <span className="font-mono text-xs text-slate-400">
                  {hasLiveActivity
                    ? `${activityLog.length} recorded event${activityLog.length === 1 ? '' : 's'}`
                    : 'Sample preview — upload an image to record live events'}
                </span>
              </div>

              <ul className="mt-3 divide-y divide-slate-800/80">
                {hasLiveActivity
                  ? activityLog.slice(0, 5).map((entry) => (
                      <li
                        key={entry.id}
                        className="flex items-center justify-between gap-4 py-2.5 text-sm"
                      >
                        <span className="text-slate-200">{entry.message}</span>
                        <span className="font-mono text-xs text-cyan-400 tabular-nums shrink-0">
                          {entry.timestamp}
                        </span>
                      </li>
                    ))
                  : SAMPLE_ACTIVITY_ENTRIES.map((sample) => (
                      <li
                        key={sample.id}
                        className="flex items-center justify-between gap-4 py-2.5 text-sm"
                      >
                        <span className="text-slate-300">{sample.message}</span>
                        <span className="font-mono text-xs text-violet-300/90 shrink-0">
                          {sample.stage}
                        </span>
                      </li>
                    ))}
              </ul>
            </div>

            <div className="mt-5 flex items-center justify-between border-t border-slate-800/80 pt-4 text-xs text-slate-400">
              <span>
                {hasLiveActivity
                  ? 'Recorded from your active browser session.'
                  : 'No image analyzed in this session yet.'}
              </span>
              <button
                type="button"
                onClick={() => onNavigate('detect')}
                className="font-medium text-cyan-400 transition-colors hover:text-cyan-300 whitespace-nowrap shrink-0"
              >
                Open Image Detection →
              </button>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}

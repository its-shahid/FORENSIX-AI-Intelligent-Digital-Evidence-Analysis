/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import {useCallback, useEffect, useState} from 'react';
import {AboutView} from './components/AboutView';
import {DetectView} from './components/DetectView';
import {Header} from './components/Header';
import {HomeView} from './components/HomeView';
import {fetchHealth} from './services/api';
import type {
  ForensicActivityEntry,
  HealthResponse,
  PageId,
} from './types/forensics';

export default function App() {
  const [activePage, setActivePage] = useState<PageId>('home');
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthLoading, setHealthLoading] = useState<boolean>(true);
  const [activityLog, setActivityLog] = useState<ForensicActivityEntry[]>([]);

  const refreshHealth = useCallback(async (): Promise<HealthResponse> => {
    setHealthLoading(true);
    const res = await fetchHealth();
    setHealth(res);
    setHealthLoading(false);
    return res;
  }, []);

  const recordActivity = useCallback(
    (entry: Omit<ForensicActivityEntry, 'id' | 'timestamp'>) => {
      setActivityLog((prev) => [
        {
          ...entry,
          id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
          timestamp: new Date().toLocaleTimeString(),
        },
        ...prev.slice(0, 9),
      ]);
    },
    [],
  );

  useEffect(() => {
    refreshHealth();
  }, [refreshHealth]);

  const handleNavigate = (page: PageId) => {
    setActivePage(page);
    window.scrollTo({top: 0, behavior: 'smooth'});
  };

  return (
    <div className="min-h-screen flex flex-col bg-[#07090E] text-slate-100">
      <Header activePage={activePage} onNavigate={handleNavigate} />

      <main className="flex-1">
        {activePage === 'home' && (
          <HomeView
            activityLog={activityLog}
            onNavigate={handleNavigate}
          />
        )}
        {activePage === 'detect' && (
          <DetectView
            health={health}
            healthLoading={healthLoading}
            onRefreshHealth={refreshHealth}
            onRecordActivity={recordActivity}
          />
        )}
        {activePage === 'about' && <AboutView />}
      </main>

      <footer className="mt-12 border-t border-slate-800/80 bg-[#07090E]">
        <div className="mx-auto flex max-w-[1360px] flex-col gap-3 px-6 py-6 text-xs text-slate-500 sm:flex-row sm:items-center sm:justify-between">
          <p>
            <span className="font-semibold uppercase tracking-wider text-slate-300">
              FORENSIX AI
            </span>{' '}
            · Intelligent Digital Evidence Analysis
          </p>
          <div className="flex items-center gap-5">
            <button
              type="button"
              onClick={() => handleNavigate('detect')}
              className="hover:text-slate-300 transition-colors"
            >
              Image Detection
            </button>
            <button
              type="button"
              onClick={() => handleNavigate('about')}
              className="hover:text-slate-300 transition-colors"
            >
              Architecture &amp; Limitations
            </button>
          </div>
        </div>
      </footer>
    </div>
  );
}

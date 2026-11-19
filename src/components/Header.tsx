import {Menu, X} from 'lucide-react';
import {useState} from 'react';
import type {PageId} from '../types/forensics';

interface HeaderProps {
  activePage: PageId;
  onNavigate: (page: PageId) => void;
}

const NAV_ITEMS: Array<{id: PageId; label: string}> = [
  {id: 'home', label: 'Home'},
  {id: 'detect', label: 'Image Detection'},
  {id: 'about', label: 'About'},
];

export function Header({activePage, onNavigate}: HeaderProps) {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const handleNav = (page: PageId) => {
    onNavigate(page);
    setMobileMenuOpen(false);
  };

  return (
    <header className="sticky top-0 z-40 border-b border-slate-800/90 bg-[#07090E]/95 backdrop-blur-sm">
      <div className="mx-auto flex max-w-[1360px] items-center justify-between gap-8 px-6 py-3.5">
        <button
          type="button"
          onClick={() => handleNav('home')}
          className="text-left font-display text-lg font-bold uppercase tracking-wider text-slate-50 transition-colors hover:text-cyan-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-500 whitespace-nowrap shrink-0"
        >
          FORENSIX <span className="text-cyan-400">AI</span>
        </button>

        <nav aria-label="Primary Navigation" className="hidden md:flex items-center gap-7">
          {NAV_ITEMS.map((item) => {
            const isActive = activePage === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => handleNav(item.id)}
                className={`relative py-1 text-sm font-medium transition-colors whitespace-nowrap shrink-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-500 ${
                  isActive
                    ? 'text-cyan-400'
                    : 'text-slate-400 hover:text-slate-100'
                }`}
              >
                {item.label}
                {isActive && (
                  <span
                    aria-hidden="true"
                    className="absolute inset-x-0 -bottom-[15px] h-[2px] bg-cyan-400"
                  />
                )}
              </button>
            );
          })}
        </nav>

        <div className="flex items-center gap-3 shrink-0">
          <button
            type="button"
            onClick={() => handleNav('detect')}
            className="hidden sm:inline-flex items-center justify-center rounded-md bg-cyan-500 px-4 py-2 text-xs font-semibold text-slate-950 transition-colors hover:bg-cyan-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400 whitespace-nowrap shrink-0"
          >
            Detect Image
          </button>

          <button
            type="button"
            onClick={() => setMobileMenuOpen((prev) => !prev)}
            aria-label={mobileMenuOpen ? 'Close navigation menu' : 'Open navigation menu'}
            className="inline-flex h-10 w-10 items-center justify-center rounded-md border border-slate-800 text-slate-300 hover:border-slate-700 hover:text-white md:hidden"
          >
            {mobileMenuOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
          </button>
        </div>
      </div>

      {mobileMenuOpen && (
        <nav
          aria-label="Mobile Navigation"
          className="border-t border-slate-800 bg-[#0B0E17] px-6 py-3 md:hidden"
        >
          <div className="flex flex-col gap-1">
            {NAV_ITEMS.map((item) => {
              const isActive = activePage === item.id;
              return (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => handleNav(item.id)}
                  className={`flex min-h-[44px] items-center justify-between rounded-md px-3 text-sm font-medium transition-colors ${
                    isActive
                      ? 'bg-slate-800/80 text-cyan-400'
                      : 'text-slate-300 hover:bg-slate-900 hover:text-white'
                  }`}
                >
                  <span>{item.label}</span>
                  {isActive && <span className="text-xs font-mono text-cyan-400">Active</span>}
                </button>
              );
            })}
          </div>
        </nav>
      )}
    </header>
  );
}

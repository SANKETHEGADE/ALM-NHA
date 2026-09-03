import React from 'react';
import { Cpu, Activity } from 'lucide-react';

/**
 * Stage 3: Console Header
 * Typographic harmony with landing page: Space Grotesk display headers, JetBrains Mono tracking-wider metadata.
 */
export function ConsoleHeader({ activeSession }) {
  const formatDateTime = (isoString) => {
    try {
      return new Date(isoString).toLocaleString([], {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
      });
    } catch (e) {
      return 'Active';
    }
  };

  return (
    <header className="h-16 px-6 bg-gradient-to-r from-[#121215] via-[#151518] to-[#121215] border-b border-white/[0.08] flex items-center justify-between shrink-0 shadow-sm z-10 select-none">
      {/* Session Title & Forensic Metadata */}
      <div className="flex items-center gap-3.5 min-w-0">
        <div className="flex flex-col min-w-0 gap-0.5">
          <div className="flex items-center gap-2.5">
            <h2 className="font-['Space_Grotesk'] text-sm sm:text-base font-bold text-white tracking-tight uppercase truncate">
              {activeSession?.label || 'Active Forensic Session'}
            </h2>
            <span className="px-2 py-0.5 rounded-full bg-white/[0.06] border border-white/[0.12] text-[10px] font-mono font-bold tracking-wider text-neutral-300 shrink-0 shadow-2xs">
              ID: {activeSession?.id || 'sess_active'}
            </span>
          </div>
          <span className="text-[10px] font-mono font-medium tracking-wider text-neutral-400 uppercase flex items-center gap-1.5">
            <span>Captured:</span>
            <span className="text-neutral-200">{formatDateTime(activeSession?.createdAt)}</span>
          </span>
        </div>
      </div>

      {/* Top Status Bar: Matching Landing Page Pill Style */}
      <div className="flex items-center gap-2.5">
        {/* Telemetry Status (Green Accent Only) */}
        <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-white/[0.04] border border-white/[0.1] text-xs font-mono shadow-xs backdrop-blur-xs">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
            <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.7)]" />
          </span>
          <span className="text-[10px] font-mono font-bold tracking-wider text-neutral-200 uppercase">
            TELEMETRY: CONNECTED
          </span>
        </div>

        {/* Engine Spec Pill (Monochrome) */}
        <div className="hidden sm:flex items-center gap-1.5 px-3 py-1 rounded-full bg-white/[0.04] border border-white/[0.1] text-xs font-mono text-neutral-400 shadow-xs">
          <Cpu className="w-3.5 h-3.5 text-neutral-400" />
          <span className="text-[10px] font-mono font-semibold tracking-wider text-neutral-300 uppercase">
            MULTIMODAL FUSION 2026.4
          </span>
        </div>
      </div>
    </header>
  );
}

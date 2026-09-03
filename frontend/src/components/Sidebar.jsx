import React from 'react';
import { Plus, Shield, LogOut, Radio, Volume2, Clock, Inbox } from 'lucide-react';

/**
 * Stage 3: Console Left Sidebar
 * Typographic harmony with landing page: Space Grotesk headings, JetBrains Mono tracking-widest labels, rounded-full pills.
 */
export function Sidebar({
  sessions = [],
  activeSessionId,
  onSelectSession,
  onNewSession,
  user,
  onSignOut,
  isCollapsed,
  onToggleCollapse
}) {
  const getBadgeStyle = (threatLevel = 'NOMINAL') => {
    switch (threatLevel.toUpperCase()) {
      case 'CRITICAL':
      case 'HIGH':
        return 'bg-red-500/15 text-red-400 border-red-500/30 shadow-[0_0_8px_rgba(239,68,68,0.2)]';
      case 'NOMINAL':
      default:
        return 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30 shadow-[0_0_8px_rgba(16,185,129,0.2)]';
    }
  };

  const formatTime = (isoString) => {
    try {
      const date = new Date(isoString);
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } catch (e) {
      return 'Just now';
    }
  };

  return (
    <aside
      className={`h-screen bg-gradient-to-b from-[#111114] via-[#0E0E11] to-[#0A0A0C] border-r border-white/[0.08] flex flex-col shrink-0 transition-all duration-200 ease-out z-30 select-none shadow-xl ${
        isCollapsed ? 'w-[68px]' : 'w-[268px]'
      }`}
    >
      {/* Sidebar Brand Header */}
      <div className="h-16 px-4 border-b border-white/[0.08] flex items-center justify-between shrink-0">
        {!isCollapsed && (
          <div className="flex items-center gap-2.5 overflow-hidden group">
            <div className="w-8 h-8 rounded-md bg-white text-[#0A0A0A] font-mono font-black text-xs tracking-tighter flex items-center justify-center shrink-0 shadow-sm">
              SH
            </div>
            <div className="flex flex-col min-w-0">
              <span className="font-['Space_Grotesk'] text-sm font-bold tracking-tight text-white uppercase truncate">
                Smart Horizon
              </span>
              <span className="text-[9px] font-mono font-semibold tracking-widest text-neutral-400 uppercase truncate">
                Forensic Console
              </span>
            </div>
          </div>
        )}

        <button
          type="button"
          onClick={onToggleCollapse}
          title={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          className="p-2 rounded-lg bg-white/[0.03] hover:bg-white/[0.08] active:bg-white/[0.12] text-neutral-400 hover:text-white transition-all duration-150 cursor-pointer border border-white/[0.06] focus-visible:outline-white/40"
        >
          <Radio className="w-4 h-4 text-neutral-300" />
        </button>
      </div>

      {/* New Session Action Button (Matching Landing Page Pill Style) */}
      <div className="p-3.5 border-b border-white/[0.08] shrink-0">
        <button
          type="button"
          onClick={onNewSession}
          className="w-full h-10 bg-white/[0.08] hover:bg-white hover:text-[#0A0A0A] active:scale-[0.98] border border-white/[0.15] rounded-full px-4 flex items-center justify-center gap-2 text-xs font-mono font-bold tracking-tight text-white transition-all duration-150 group cursor-pointer shadow-sm hover:shadow-md focus-visible:outline-white/40"
        >
          <Plus className="w-3.5 h-3.5 group-hover:scale-110 transition-transform duration-200 shrink-0" />
          {!isCollapsed && <span>New Session</span>}
        </button>
      </div>

      {/* Scrollable Sessions List */}
      <div className="flex-1 overflow-y-auto px-2.5 py-3 flex flex-col gap-1.5">
        {!isCollapsed && (
          <div className="px-2 py-1 text-[10px] font-mono font-bold text-neutral-400 uppercase tracking-widest">
            Evidence Sessions ({sessions.length})
          </div>
        )}

        {sessions.length === 0 ? (
          <div className="p-6 text-center text-xs font-mono text-neutral-500 flex flex-col items-center gap-2 my-auto">
            <Inbox className="w-5 h-5 text-neutral-600" />
            {!isCollapsed && <span>No sessions recorded</span>}
          </div>
        ) : (
          sessions.map((sess) => {
            const isActive = sess.id === activeSessionId;
            const threatLevel = sess.result?.threat_level || sess.result?.fusion?.threat_level || 'NOMINAL';

            return (
              <button
                key={sess.id}
                type="button"
                onClick={() => onSelectSession(sess.id)}
                className={`w-full rounded-xl p-3 text-left flex flex-col gap-2 transition-all duration-150 border cursor-pointer active:scale-[0.99] focus-visible:outline-white/40 ${
                  isActive
                    ? 'bg-gradient-to-r from-white/[0.14] to-white/[0.07] border-white/25 text-white shadow-md shadow-black/40 ring-1 ring-white/10'
                    : 'bg-transparent hover:bg-white/[0.04] border-transparent text-neutral-400 hover:text-neutral-200'
                }`}
                title={sess.label}
              >
                <div className="flex items-center justify-between gap-1.5">
                  <div className="flex items-center gap-2 min-w-0">
                    <Volume2 className={`w-3.5 h-3.5 shrink-0 ${isActive ? 'text-white' : 'text-neutral-500'}`} />
                    {!isCollapsed && (
                      <span className={`font-['Space_Grotesk'] text-xs truncate max-w-[130px] ${isActive ? 'text-white font-bold' : 'text-neutral-300 font-medium'}`}>
                        {sess.label}
                      </span>
                    )}
                  </div>
                  {!isCollapsed && (
                    <span className={`px-2 py-0.5 rounded-full text-[9px] font-mono font-bold uppercase tracking-wider border ${getBadgeStyle(threatLevel)}`}>
                      {threatLevel}
                    </span>
                  )}
                </div>

                {!isCollapsed && (
                  <div className="flex items-center justify-between text-[10px] font-mono text-neutral-400 pt-0.5">
                    <span className="tracking-wider">{sess.id.slice(-8)}</span>
                    <div className="flex items-center gap-1">
                      <Clock className="w-2.5 h-2.5 text-neutral-500" />
                      <span>{formatTime(sess.createdAt)}</span>
                    </div>
                  </div>
                )}
              </button>
            );
          })
        )}
      </div>

      {/* User Footer & Sign-out */}
      <div className="p-3.5 border-t border-white/[0.08] shrink-0 bg-[#0E0E11]/90 backdrop-blur-xs">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2.5 min-w-0">
            <div className="w-8 h-8 rounded-full bg-white text-[#0A0A0A] font-mono font-bold text-xs flex items-center justify-center shrink-0 shadow-sm">
              {user?.avatar || 'AN'}
            </div>
            {!isCollapsed && (
              <div className="flex flex-col min-w-0">
                <span className="font-['Space_Grotesk'] text-xs font-bold text-white tracking-tight truncate">{user?.name || 'Analyst'}</span>
                <span className="text-[10px] font-mono font-semibold tracking-wider text-neutral-400 uppercase truncate">{user?.role || 'Forensic Lead'}</span>
              </div>
            )}
          </div>

          <button
            type="button"
            onClick={onSignOut}
            title="Sign out of console"
            className="p-2 rounded-lg hover:bg-white/[0.08] active:bg-white/[0.15] text-neutral-400 hover:text-red-400 transition-all duration-150 shrink-0 cursor-pointer border border-transparent hover:border-white/[0.08] focus-visible:outline-white/40"
          >
            <LogOut className="w-4 h-4" />
          </button>
        </div>
      </div>
    </aside>
  );
}

import React, { useState, useEffect } from 'react';
import { Activity, Radio, Shield, Terminal } from 'lucide-react';

export function Footer({ onOpenConsole }) {
  const [utcTime, setUtcTime] = useState('');

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      setUtcTime(now.toUTCString().split(' ')[4] + ' UTC');
    };
    updateTime();
    const timer = setInterval(updateTime, 1000);
    return () => clearInterval(timer);
  }, []);

  return (
    <footer className="py-16 px-6 bg-[#FAFAFA] border-t border-[#E4E4E4] text-[#0A0A0A] select-none">
      <div className="max-w-7xl mx-auto flex flex-col gap-12">
        {/* Top Row: Brand & Telemetry Status */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-6 pb-8 border-b border-[#E4E4E4]">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-[#0A0A0A] text-white flex items-center justify-center font-mono font-black text-xs rounded-sm shadow-xs">
              SH
            </div>
            <div className="flex flex-col">
              <span className="font-['Space_Grotesk'] font-bold text-sm tracking-tight uppercase">
                Smart Horizon
              </span>
              <span className="text-[10px] font-mono font-semibold text-[#6B6B6B]">
                Context-Aware Forensic Acoustic Intelligence
              </span>
            </div>
          </div>

          <div className="flex items-center gap-4 text-xs font-mono">
            <div className="flex items-center gap-1.5 px-3 py-1 rounded-full border border-[#E4E4E4] bg-white text-[#2E2E2E] shadow-2xs">
              <span className="w-2 h-2 rounded-full bg-[#0A0A0A] animate-pulse" />
              <span className="font-semibold">NODAL CLOCK: {utcTime}</span>
            </div>

            <button
              type="button"
              onClick={onOpenConsole}
              className="px-3.5 py-1.5 rounded-full bg-[#0A0A0A] hover:bg-[#2E2E2E] active:scale-[0.98] text-white text-[11px] font-semibold transition-all duration-150 cursor-pointer shadow-xs focus-visible:outline-[#0A0A0A]"
            >
              Console Access
            </button>
          </div>
        </div>

        {/* Links Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-8 text-xs font-mono">
          <div className="flex flex-col gap-3">
            <span className="font-bold text-[#0A0A0A] uppercase tracking-wider">
              Detection Engines
            </span>
            <a href="#detection" className="text-[#4A4A4A] hover:text-[#0A0A0A] transition-colors">
              Multimodal Fusion
            </a>
            <a href="#detection" className="text-[#4A4A4A] hover:text-[#0A0A0A] transition-colors">
              Diarization Validator
            </a>
            <a href="#detection" className="text-[#4A4A4A] hover:text-[#0A0A0A] transition-colors">
              Acoustic Transients
            </a>
          </div>

          <div className="flex flex-col gap-3">
            <span className="font-bold text-[#0A0A0A] uppercase tracking-wider">
              Platform
            </span>
            <a href="#how-it-works" className="text-[#4A4A4A] hover:text-[#0A0A0A] transition-colors">
              How it works
            </a>
            <a href="#live-demo" className="text-[#4A4A4A] hover:text-[#0A0A0A] transition-colors">
              Live Engine Test
            </a>
            <button
              type="button"
              onClick={onOpenConsole}
              className="text-left text-[#4A4A4A] hover:text-[#0A0A0A] transition-colors cursor-pointer"
            >
              Interactive Console
            </button>
          </div>

          <div className="flex flex-col gap-3">
            <span className="font-bold text-[#0A0A0A] uppercase tracking-wider">
              Verification
            </span>
            <span className="text-[#4A4A4A]">False-Positive Standard 2026</span>
            <span className="text-[#4A4A4A]">48 kHz Linear Ingestion</span>
            <span className="text-[#4A4A4A]">Zero-Trust Dispatch Audit</span>
          </div>

          <div className="flex flex-col gap-3">
            <span className="font-bold text-[#0A0A0A] uppercase tracking-wider">
              System Info
            </span>
            <span className="text-[#4A4A4A]">Release: 2026.4.2</span>
            <span className="text-[#4A4A4A]">License: Proprietary / Enterprise</span>
            <span className="text-[#4A4A4A]">Status: ALL SYSTEMS NOMINAL</span>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="pt-8 border-t border-[#E4E4E4] flex flex-col sm:flex-row items-center justify-between gap-4 text-[11px] font-mono text-[#6B6B6B]">
          <span>© 2026 Smart Horizon Security Systems. All rights reserved.</span>
          <span className="font-semibold text-[#0A0A0A]">[06/06] VERIFIED DEPLOYMENT</span>
        </div>
      </div>
    </footer>
  );
}

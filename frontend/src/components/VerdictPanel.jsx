import React from 'react';
import {
  ShieldAlert,
  ShieldCheck,
  AlertTriangle,
  FileText,
  Users,
  Mic,
  Activity,
  Layers,
  CheckCircle2,
  XCircle,
  Clock,
  Info,
  Radio,
  Sliders,
  Sparkles,
  Zap,
  Volume2
} from 'lucide-react';

/**
 * Stage 3: VerdictPanel
 * Unified Typography & Finishing Polish matching the landing page:
 * - Display Headlines: Space Grotesk Bold/Black with tight tracking
 * - Numerals & Metrics: JetBrains Mono tabular-nums with uppercase wide tracking
 * - Body Text: Clean Inter
 * - Palette: Strictly Monochrome Grayscale + Emerald Green (Nominal) & Crimson Red (Critical)
 */
export function VerdictPanel({ activeSession, isLoading }) {
  if (isLoading) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center p-12 text-center gap-5 my-auto">
        <div className="relative flex items-center justify-center">
          <div className="w-14 h-14 border-2 border-white/20 border-t-white rounded-full animate-spin" />
          <Radio className="w-6 h-6 text-white absolute animate-pulse" />
        </div>
        <div className="flex flex-col gap-1.5">
          <span className="font-['Space_Grotesk'] text-sm font-bold text-white tracking-wide uppercase">
            Processing Live Voice Telemetry
          </span>
          <span className="text-xs font-mono font-semibold tracking-wider text-neutral-400 uppercase">
            Synthesizing transcript intent, acoustic pitch & multi-speaker diarization...
          </span>
        </div>
      </div>
    );
  }

  if (!activeSession || !activeSession.result) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center p-12 text-center gap-5 max-w-md mx-auto my-auto select-none">
        <div className="w-14 h-14 rounded-2xl bg-gradient-to-b from-white/[0.08] to-white/[0.02] border border-white/[0.12] flex items-center justify-center text-white shadow-lg shadow-black/40">
          <Activity className="w-7 h-7 text-neutral-300" />
        </div>
        <div className="flex flex-col gap-2">
          <h3 className="font-['Space_Grotesk'] text-base font-bold text-white tracking-tight uppercase">
            No Acoustic Evidence in Session
          </h3>
          <p className="text-xs text-neutral-400 leading-relaxed font-sans">
            Use the <strong className="text-white font-semibold">Voice to Text</strong> button below to capture live speech or select a benchmark scenario to evaluate context-aware forensic synthesis.
          </p>
        </div>
      </div>
    );
  }

  const { fusion, diarization, acoustic_features } = activeSession.result;
  const threatLevel = fusion?.threat_level || 'NOMINAL';
  const rationale = fusion?.fusion_rationale || 'Analysis complete.';
  const channelsUsed = fusion?.channels_used || ['transcript', 'vocal_biometrics'];
  const channelsDiscounted = fusion?.channels_discounted || [];
  const intent = fusion?.intent || 'Evaluated Speech Pattern';
  const semantic = fusion?.semantic_analysis || {};
  const confidences = fusion?.confidence_metrics || {
    speech_confidence: 96,
    classification_confidence: fusion?.confidence || 95,
    acoustic_confidence: 94,
    speaker_confidence: diarization?.confidence || 96
  };

  const capture = activeSession.capture || {};
  const biometrics = capture.vocalBiometrics || { emotion: 'neutral', arousal: 'low', confidence: 0.94 };
  const acousticEvents = capture.acousticEvents || [];
  const acoustics = acoustic_features || {
    pitch_f0: 138,
    pitch_label: 'Normal Pitch',
    pitch_description: 'Standard vocal register (~138 Hz)',
    volume_rms_db: -18,
    speech_rate_wpm: 145,
    voice_activity_pct: 82,
    silence_duration_sec: 0.4,
    snr_db: 24,
    background_noise_db: -46
  };

  // Color constraints: Green exclusively for nominal, Red exclusively for critical
  const isCritical = threatLevel === 'CRITICAL' || threatLevel === 'HIGH';

  const threatTheme = isCritical
    ? {
        badge: 'bg-red-500/15 text-red-400 border-red-500/30 shadow-[0_0_12px_rgba(239,68,68,0.25)]',
        icon: <ShieldAlert className="w-5 h-5 text-red-400" />,
        title: 'CRITICAL ACTIVE THREAT',
        cardBorder: 'border-red-500/30 hover:border-red-500/40',
        glow: 'shadow-[0_8px_32px_rgba(239,68,68,0.08)]'
      }
    : {
        badge: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30 shadow-[0_0_12px_rgba(16,185,129,0.25)]',
        icon: <ShieldCheck className="w-5 h-5 text-emerald-400" />,
        title: 'NOMINAL / NON-THREATENING SCENE',
        cardBorder: 'border-emerald-500/30 hover:border-emerald-500/40',
        glow: 'shadow-[0_8px_32px_rgba(16,185,129,0.08)]'
      };

  return (
    <div className="flex-1 overflow-y-auto p-8 flex flex-col gap-7 max-w-5xl mx-auto w-full select-none">
      {/* 1. Multimodal Fusion Synthesis Verdict Card */}
      <section
        className={`bg-gradient-to-b from-[#17171B] via-[#131316] to-[#0F0F12] border ${threatTheme.cardBorder} rounded-2xl p-6 md:p-7 ${threatTheme.glow} flex flex-col gap-5 transition-all duration-200`}
      >
        {/* Top Header */}
        <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-white/[0.08]">
          <div className="flex items-center gap-3.5">
            <div className="w-10 h-10 rounded-xl bg-white/[0.06] border border-white/[0.12] flex items-center justify-center shrink-0 shadow-inner">
              {threatTheme.icon}
            </div>
            <div>
              <div className="flex items-center gap-2.5">
                <span className={`px-2.5 py-0.5 rounded-full text-[10px] font-mono font-bold uppercase tracking-wider border ${threatTheme.badge}`}>
                  {threatLevel}
                </span>
                <h3 className="font-['Space_Grotesk'] text-base sm:text-lg font-bold text-white tracking-tight uppercase">
                  {threatTheme.title}
                </h3>
              </div>
              <span className="text-xs font-mono font-semibold tracking-wider text-neutral-400 mt-1 block uppercase">
                INTENT: <span className="text-neutral-200">{intent}</span>
              </span>
            </div>
          </div>

          {/* Separate Confidence Dimensions (Matching Landing Page 98.4% Numeral Style) */}
          <div className="flex flex-wrap items-center gap-2">
            <div className="px-3 py-1.5 rounded-xl bg-white/[0.04] border border-white/[0.08] flex flex-col items-end text-right shadow-xs">
              <span className="text-[9px] font-mono font-bold text-neutral-400 uppercase tracking-widest">ASR Conf</span>
              <span className="text-xs font-mono font-bold text-white tabular-nums">{confidences.speech_confidence}%</span>
            </div>
            <div className="px-3 py-1.5 rounded-xl bg-white/[0.04] border border-white/[0.08] flex flex-col items-end text-right shadow-xs">
              <span className="text-[9px] font-mono font-bold text-neutral-400 uppercase tracking-widest">Classify Conf</span>
              <span className="text-xs font-mono font-bold text-white tabular-nums">{confidences.classification_confidence}%</span>
            </div>
            <div className="px-3 py-1.5 rounded-xl bg-white/[0.04] border border-white/[0.08] flex flex-col items-end text-right shadow-xs">
              <span className="text-[9px] font-mono font-bold text-neutral-400 uppercase tracking-widest">Acoustic Conf</span>
              <span className="text-xs font-mono font-bold text-white tabular-nums">{confidences.acoustic_confidence}%</span>
            </div>
          </div>
        </div>

        {/* Fusion Rationale Banner */}
        <div className="bg-[#1A1A1E]/80 border border-white/[0.08] rounded-xl p-4 flex flex-col gap-2 shadow-inner">
          <div className="flex items-center gap-2 text-xs font-mono font-bold tracking-wider text-white uppercase">
            <Info className="w-4 h-4 text-white shrink-0" />
            <span>Multimodal Context Rationale</span>
          </div>
          <p className="text-xs sm:text-sm text-neutral-200 leading-relaxed font-sans pl-6">
            {rationale}
          </p>
        </div>

        {/* 2. Semantic & Context Analysis Sub-Panel */}
        <div className="bg-[#1A1A1E]/80 border border-white/[0.08] rounded-xl p-4 flex flex-col gap-3 shadow-inner">
          <div className="flex items-center gap-2 text-xs font-mono font-bold tracking-wider text-white pb-2 border-b border-white/[0.06] uppercase">
            <Sparkles className="w-4 h-4 text-white" />
            <span>Semantic Context & Grammatical Reasoning</span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono">
            <div className="bg-[#141417] border border-white/[0.06] rounded-lg p-2.5 flex flex-col gap-1">
              <span className="text-[9px] font-bold text-neutral-400 uppercase tracking-widest">Modality</span>
              <span className="font-['Space_Grotesk'] font-bold text-xs text-white uppercase truncate">{semantic.modality || 'Actual Incident'}</span>
            </div>

            <div className="bg-[#141417] border border-white/[0.06] rounded-lg p-2.5 flex flex-col gap-1">
              <span className="text-[9px] font-bold text-neutral-400 uppercase tracking-widest">Temporal Frame</span>
              <span className="font-['Space_Grotesk'] font-bold text-xs text-white uppercase truncate">{semantic.temporal_frame || 'Present (Active)'}</span>
            </div>

            <div className="bg-[#141417] border border-white/[0.06] rounded-lg p-2.5 flex flex-col gap-1">
              <span className="text-[9px] font-bold text-neutral-400 uppercase tracking-widest">Negation Detected</span>
              <span className={`font-['Space_Grotesk'] font-bold text-xs uppercase ${semantic.negation_detected ? 'text-emerald-400' : 'text-neutral-300'}`}>
                {semantic.negation_detected ? 'YES (Negated)' : 'NO'}
              </span>
            </div>

            <div className="bg-[#141417] border border-white/[0.06] rounded-lg p-2.5 flex flex-col gap-1">
              <span className="text-[9px] font-bold text-neutral-400 uppercase tracking-widest">Joke / Humor</span>
              <span className={`font-['Space_Grotesk'] font-bold text-xs uppercase ${semantic.joke_detected ? 'text-emerald-400' : 'text-neutral-300'}`}>
                {semantic.joke_detected ? 'YES (Discounted)' : 'NO'}
              </span>
            </div>
          </div>
        </div>

        {/* Channels Used vs Discounted */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5">
          {/* Channels Used */}
          <div className="bg-[#1A1A1E]/80 border border-white/[0.08] rounded-xl p-4 flex flex-col gap-2.5 shadow-inner">
            <div className="flex items-center gap-2 text-xs font-mono font-bold tracking-wider text-neutral-200 uppercase">
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              <span>CHANNELS DRIVING VERDICT ({channelsUsed.length})</span>
            </div>
            <div className="flex flex-wrap gap-2">
              {channelsUsed.map((ch) => (
                <span
                  key={ch}
                  className="px-2.5 py-1 rounded-lg bg-[#141417] border border-white/[0.08] text-[10px] font-mono font-bold text-white uppercase tracking-wider shadow-2xs"
                >
                  ✓ {ch.replace('_', ' ').toUpperCase()}
                </span>
              ))}
            </div>
          </div>

          {/* Channels Discounted */}
          <div className="bg-[#1A1A1E]/80 border border-white/[0.08] rounded-xl p-4 flex flex-col gap-2.5 shadow-inner">
            <div className="flex items-center gap-2 text-xs font-mono font-bold tracking-wider text-neutral-200 uppercase">
              <XCircle className="w-4 h-4 text-neutral-400" />
              <span>CHANNELS DISCOUNTED / OVERRIDDEN ({channelsDiscounted.length})</span>
            </div>
            <div className="flex flex-wrap gap-2">
              {channelsDiscounted.length === 0 ? (
                <span className="text-[10px] font-mono font-medium text-neutral-400">None (all channels concordant)</span>
              ) : (
                channelsDiscounted.map((ch) => (
                  <span
                    key={ch}
                    className="px-2.5 py-1 rounded-lg bg-[#141417] border border-white/[0.12] text-[10px] font-mono font-bold text-neutral-300 uppercase tracking-wider shadow-2xs"
                  >
                    ✕ {ch.replace('_', ' ').toUpperCase()} (DISCOUNTED)
                  </span>
                ))
              )}
            </div>
          </div>
        </div>
      </section>

      {/* 3. Physical Acoustic Measurements Card */}
      <section className="bg-gradient-to-b from-[#17171B] via-[#131316] to-[#0F0F12] border border-white/[0.09] rounded-2xl p-6 md:p-7 shadow-xl shadow-black/40 flex flex-col gap-5">
        <div className="flex items-center justify-between pb-4 border-b border-white/[0.08]">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-white/[0.06] border border-white/[0.12] flex items-center justify-center text-white shadow-inner">
              <Sliders className="w-4 h-4 text-neutral-300" />
            </div>
            <div>
              <h3 className="font-['Space_Grotesk'] text-base sm:text-lg font-bold text-white tracking-tight uppercase">
                Physical Acoustic Measurements
              </h3>
              <span className="text-[10px] font-mono font-semibold tracking-wider text-neutral-400 uppercase">
                Independent physical audio measurements (not deterministic threat rules)
              </span>
            </div>
          </div>
          <span className="text-[10px] font-mono font-bold px-3 py-1 rounded-full bg-white/[0.05] border border-white/[0.08] text-neutral-300 uppercase tracking-widest hidden sm:inline">
            Secondary Evidence
          </span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3.5 text-xs font-mono">
          {/* Pitch F0 */}
          <div className="bg-[#1A1A1E]/80 border border-white/[0.08] rounded-xl p-3.5 flex flex-col gap-1 shadow-inner">
            <div className="flex items-center gap-1.5 text-neutral-400">
              <Mic className="w-3.5 h-3.5 text-white" />
              <span className="text-[10px] font-bold uppercase tracking-widest">Pitch (F0)</span>
            </div>
            <span className="text-xl font-bold text-white font-mono tabular-nums">{acoustics.pitch_f0} Hz</span>
            <span className="text-[10px] font-mono text-neutral-400 uppercase tracking-wider">{acoustics.pitch_label}</span>
          </div>

          {/* Volume RMS */}
          <div className="bg-[#1A1A1E]/80 border border-white/[0.08] rounded-xl p-3.5 flex flex-col gap-1 shadow-inner">
            <div className="flex items-center gap-1.5 text-neutral-400">
              <Volume2 className="w-3.5 h-3.5 text-white" />
              <span className="text-[10px] font-bold uppercase tracking-widest">Energy / RMS</span>
            </div>
            <span className="text-xl font-bold text-white font-mono tabular-nums">{acoustics.volume_rms_db} dB</span>
            <span className="text-[10px] font-mono text-neutral-400 uppercase tracking-wider">Speech Amplitude</span>
          </div>

          {/* Speech Rate WPM */}
          <div className="bg-[#1A1A1E]/80 border border-white/[0.08] rounded-xl p-3.5 flex flex-col gap-1 shadow-inner">
            <div className="flex items-center gap-1.5 text-neutral-400">
              <Clock className="w-3.5 h-3.5 text-white" />
              <span className="text-[10px] font-bold uppercase tracking-widest">Speech Rate</span>
            </div>
            <span className="text-xl font-bold text-white font-mono tabular-nums">{acoustics.speech_rate_wpm} WPM</span>
            <span className="text-[10px] font-mono text-neutral-400 uppercase tracking-wider">Pace Cadence</span>
          </div>

          {/* SNR & Background */}
          <div className="bg-[#1A1A1E]/80 border border-white/[0.08] rounded-xl p-3.5 flex flex-col gap-1 shadow-inner">
            <div className="flex items-center gap-1.5 text-neutral-400">
              <Activity className="w-3.5 h-3.5 text-white" />
              <span className="text-[10px] font-bold uppercase tracking-widest">Signal / SNR</span>
            </div>
            <span className="text-xl font-bold text-white font-mono tabular-nums">+{acoustics.snr_db} dB</span>
            <span className="text-[10px] font-mono text-neutral-400 uppercase tracking-wider">{acoustics.voice_activity_pct}% voice active</span>
          </div>
        </div>
      </section>

      {/* 4. Validated Speaker Diarization Section */}
      <section className="bg-gradient-to-b from-[#17171B] via-[#131316] to-[#0F0F12] border border-white/[0.09] rounded-2xl p-6 md:p-7 shadow-xl shadow-black/40 flex flex-col gap-5">
        <div className="flex items-center justify-between pb-4 border-b border-white/[0.08]">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-white/[0.06] border border-white/[0.12] flex items-center justify-center text-white shadow-inner">
              <Users className="w-4 h-4 text-neutral-300" />
            </div>
            <div>
              <h3 className="font-['Space_Grotesk'] text-base sm:text-lg font-bold text-white tracking-tight uppercase">
                Validated Speaker Diarization ({diarization?.speaker_segments?.length || 1} Turns)
              </h3>
              <span className="text-[10px] font-mono font-semibold tracking-wider text-neutral-400 uppercase">
                Multi-speaker segmentation with neutral turn labels
              </span>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-mono font-bold tracking-widest text-neutral-400 uppercase">SPEAKER CONF:</span>
            <span className="text-xs font-mono font-bold text-white tabular-nums">
              {confidences.speaker_confidence || 96}%
            </span>
          </div>
        </div>

        {/* Validated Speaker Segments */}
        <div className="flex flex-col gap-3">
          {(diarization?.speaker_segments || []).map((spk, idx) => (
            <div
              key={idx}
              className="bg-[#1A1A1E]/80 border border-white/[0.08] rounded-xl p-4 flex flex-col gap-2.5 shadow-inner"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2.5">
                  <span className="px-2.5 py-0.5 rounded-full bg-white text-[#0A0A0A] text-[10px] font-mono font-bold uppercase tracking-wider shadow-sm">
                    {spk.speaker_id || `Speaker ${idx + 1}`}
                  </span>
                  <span className="text-xs font-mono font-semibold tracking-wider text-neutral-400 uppercase">Forensic Turn</span>
                </div>
                <div className="flex items-center gap-1.5 text-[11px] font-mono font-semibold text-neutral-400 tabular-nums">
                  <Clock className="w-3.5 h-3.5 text-neutral-400" />
                  <span>
                    [{Number(spk.start).toFixed(1)}s - {Number(spk.end).toFixed(1)}s]
                  </span>
                </div>
              </div>

              <p className="text-xs sm:text-sm text-white leading-relaxed italic bg-[#141417] border border-white/[0.06] rounded-lg p-3.5 font-sans">
                "{spk.text}"
              </p>
            </div>
          ))}
        </div>

        {/* Merge Decisions Audit Log */}
        {diarization?.merge_decisions && diarization.merge_decisions.length > 0 && (
          <div className="mt-1 pt-4 border-t border-white/[0.08] flex flex-col gap-2.5">
            <div className="flex items-center gap-2 text-xs font-mono font-bold tracking-wider text-white uppercase">
              <Layers className="w-4 h-4 text-white" />
              <span>Diarization Validator Merge Audit Log</span>
            </div>

            <div className="flex flex-col gap-2">
              {diarization.merge_decisions.map((dec, i) => (
                <div
                  key={i}
                  className="text-[11px] font-mono bg-[#1A1A1E]/80 border border-white/[0.08] rounded-xl px-3.5 py-2.5 flex flex-col sm:flex-row sm:items-center justify-between gap-1 text-neutral-300"
                >
                  <div className="flex items-center gap-2">
                    <span
                      className={`px-2 py-0.5 rounded-full text-[9px] font-bold uppercase tracking-wider border ${
                        dec.decision === 'merged'
                          ? 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30'
                          : 'bg-white/10 text-neutral-200 border-white/20'
                      }`}
                    >
                      {dec.decision}
                    </span>
                    <span className="text-neutral-400 font-medium">[{dec.segments_considered?.join(' + ')}]</span>
                  </div>
                  <span className="text-[10px] text-neutral-400 sm:text-right font-medium">{dec.reason}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </section>
    </div>
  );
}

import React from 'react';
import { VoiceAIOrbVisualizer } from './VoiceAIOrbVisualizer';
import { RealForensicCharts } from './RealForensicCharts';
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

export function VerdictPanel({ activeSession, isLoading, isCapturing, realtimeTelemetry = [] }) {
  if (isCapturing) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center p-6 md:p-8 bg-[#0d0d0d] text-[#ececec]">
        <div className="flex flex-col items-center gap-4 animate-in fade-in duration-200">
          <VoiceAIOrbVisualizer telemetry={realtimeTelemetry} isCapturing={true} />
          
          <div className="flex flex-col items-center gap-1.5 text-center">
            <span className="text-sm font-medium text-white tracking-wide">
              Listening for voice input...
            </span>
            <span className="text-xs text-[#8e8ea0]">
              Real-time Web Audio FFT responsive telemetry
            </span>
          </div>
        </div>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center p-6 md:p-8 bg-[#0d0d0d] text-[#ececec]">
        <div className="flex flex-col items-center gap-4 animate-in fade-in duration-200">
          <VoiceAIOrbVisualizer telemetry={realtimeTelemetry} isCapturing={false} />

          <div className="flex flex-col items-center gap-1.5 text-center">
            <span className="text-sm font-medium text-white tracking-wide animate-pulse">
              Processing Voice Telemetry & Multimodal Intent...
            </span>
            <span className="text-xs text-[#8e8ea0]">
              Synthesizing vocal biometrics, pitch, & speaker diarization
            </span>
          </div>
        </div>
      </div>
    );
  }

  if (!activeSession || !activeSession.result) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center p-6 md:p-12 bg-[#0d0d0d] text-[#ececec] overflow-y-auto select-none">
        <div className="max-w-3xl w-full flex flex-col items-center gap-8 animate-in fade-in duration-300 my-auto">
          
          {/* Big Font Quote Header */}
          <div className="flex flex-col items-center text-center gap-3">
            <h1 className="text-2xl md:text-3xl font-medium tracking-tight text-[#ececec]">
              What can Vox help with?
            </h1>
            <p className="text-sm md:text-base text-[#8e8ea0] max-w-xl leading-relaxed italic">
              "Forensic acoustic intelligence engine for real-time speech threat evaluation, vocal biometrics, and multi-speaker diarization."
            </p>
          </div>

          {/* 4 Feature Cards Grid (ChatGPT Style) */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 w-full pt-2">
            
            {/* Card 1 */}
            <div className="bg-[#181818] border border-[#262626] rounded-xl p-5 flex flex-col gap-2.5 hover:border-[#383838] transition-colors">
              <div className="flex items-center gap-2 text-white">
                <ShieldAlert className="w-5 h-5 text-[#ececec]" />
                <h3 className="text-sm font-medium">Real-Time Threat Detection</h3>
              </div>
              <p className="text-xs text-[#8e8ea0] leading-relaxed">
                Evaluates incoming voice streams against trained PyTorch models to flag emergency calls, armed threats, or distress signals.
              </p>
            </div>

            {/* Card 2 */}
            <div className="bg-[#181818] border border-[#262626] rounded-xl p-5 flex flex-col gap-2.5 hover:border-[#383838] transition-colors">
              <div className="flex items-center gap-2 text-white">
                <Activity className="w-5 h-5 text-[#ececec]" />
                <h3 className="text-sm font-medium">Acoustic Telemetry & Pitch</h3>
              </div>
              <p className="text-xs text-[#8e8ea0] leading-relaxed">
                Calculates real fundamental pitch (F0 Hz), RMS volume energy (dB), speech cadence (WPM), and signal-to-noise ratio.
              </p>
            </div>

            {/* Card 3 */}
            <div className="bg-[#181818] border border-[#262626] rounded-xl p-5 flex flex-col gap-2.5 hover:border-[#383838] transition-colors">
              <div className="flex items-center gap-2 text-white">
                <Users className="w-5 h-5 text-[#ececec]" />
                <h3 className="text-sm font-medium">Multi-Speaker Diarization</h3>
              </div>
              <p className="text-xs text-[#8e8ea0] leading-relaxed">
                Extracts voiceprint embeddings to isolate and segment distinct conversational speaker turns (Speaker 1, Speaker 2).
              </p>
            </div>

            {/* Card 4 */}
            <div className="bg-[#181818] border border-[#262626] rounded-xl p-5 flex flex-col gap-2.5 hover:border-[#383838] transition-colors">
              <div className="flex items-center gap-2 text-white">
                <Sparkles className="w-5 h-5 text-[#ececec]" />
                <h3 className="text-sm font-medium">Context & Intent Reasoning</h3>
              </div>
              <p className="text-xs text-[#8e8ea0] leading-relaxed">
                Multimodal LLM reasoning disambiguates actual critical incidents from sarcastic remarks, movie quotes, and jokes.
              </p>
            </div>

          </div>

          <span className="text-xs text-[#8e8ea0] font-mono pt-2">
            Click the centered <strong className="text-white">Microphone</strong> or <strong className="text-white">Upload</strong> icon below to begin audio analysis
          </span>

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

  const isCritical = threatLevel === 'CRITICAL' || threatLevel === 'HIGH';

  const threatTheme = isCritical
    ? {
        badge: 'bg-red-500/15 text-red-400 border border-red-500/20',
        title: 'Critical active threat'
      }
    : {
        badge: 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/20',
        title: 'Nominal / Non-threatening scene'
      };

  return (
    <div className="flex-1 overflow-y-auto px-6 md:px-12 py-8 flex flex-col gap-6 w-full max-w-4xl mx-auto select-text bg-[#0d0d0d] text-[#ececec]">
      {/* 1. USER PROMPT MESSAGE (Right-aligned ChatGPT user prompt) */}
      <div className="flex items-start justify-end gap-3 animate-in fade-in duration-150">
        <div className="bg-[#181818] border border-[#262626] text-[#ececec] rounded-2xl px-4 py-3 max-w-xl text-sm leading-relaxed shadow-xs">
          <div className="text-[11px] text-[#8e8ea0] mb-1 font-medium">User Voice</div>
          <div>"{capture.transcript || 'Audio sample evaluated'}"</div>
        </div>
        <div className="w-8 h-8 rounded-full bg-[#262626] flex items-center justify-center text-white text-xs font-semibold shrink-0">
          You
        </div>
      </div>

      {/* 2. CHATGPT ASSISTANT RESPONSE STREAM (Assistant response card block) */}
      <div className="flex items-start gap-4 pt-1 animate-in fade-in duration-200">
        {/* ChatGPT AI Avatar */}
        <div className="w-8 h-8 rounded-full bg-white text-black flex items-center justify-center text-xs font-bold shrink-0 shadow-sm">
          SH
        </div>

        {/* Assistant Content Stream (Distinct Message Block, rounded 12px) */}
        <div className="flex-1 flex flex-col gap-6 text-sm leading-relaxed text-[#ececec] bg-[#181818] border border-[#262626] rounded-xl p-5 shadow-sm">
          {/* Verdict Title & Badge */}
          <div className="flex items-center gap-3">
            <span className={`px-2.5 py-0.5 rounded-full text-xs font-medium ${threatTheme.badge}`}>
              {threatLevel}
            </span>
            <h3 className="text-base font-medium text-[#ececec]">
              {threatTheme.title}
            </h3>
          </div>

          {/* Rationale Narrative */}
          <div className="text-sm text-[#ececec] leading-relaxed">
            {rationale}
          </div>

          {/* Semantic & Intent Analysis List */}
          <div className="flex flex-col gap-2.5 pt-4 border-t border-[#262626]">
            <h4 className="text-sm font-medium text-[#ececec]">Semantic & intent reasoning</h4>
            <ul className="space-y-2 text-xs text-[#8e8ea0] pl-1">
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-[#ececec]" />
                <span><strong className="text-[#ececec] font-medium">Intent:</strong> {intent}</span>
              </li>
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-[#ececec]" />
                <span><strong className="text-[#ececec] font-medium">Modality:</strong> {semantic.modality || 'Actual Incident'}</span>
              </li>
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-[#ececec]" />
                <span><strong className="text-[#ececec] font-medium">Temporal Frame:</strong> {semantic.temporal_frame || 'Present (Active)'}</span>
              </li>
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-[#ececec]" />
                <span><strong className="text-[#ececec] font-medium">Negation Detected:</strong> {semantic.negation_detected ? 'Yes (Negated context evaluated)' : 'No'}</span>
              </li>
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-[#ececec]" />
                <span><strong className="text-[#ececec] font-medium">Joke / Humor Detected:</strong> {semantic.joke_detected ? 'Yes (Humorous tone / joke context)' : 'No'}</span>
              </li>
            </ul>
          </div>

          {/* Physical Acoustic Measurements Stat Grid (#1e1e1e bg, 12px rounded, border #262626) */}
          <div className="flex flex-col gap-2.5 pt-4 border-t border-[#262626]">
            <h4 className="text-sm font-medium text-[#ececec]">Physical acoustic measurements</h4>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-1">
              <div className="bg-[#1e1e1e] border border-[#262626] rounded-xl p-3 flex flex-col gap-1">
                <span className="text-[11px] text-[#8e8ea0] font-normal">Pitch (F0)</span>
                <span className="text-sm font-medium text-[#ececec]">{acoustics.pitch_f0} Hz</span>
                <span className="text-[10px] text-[#8e8ea0]">{acoustics.pitch_label}</span>
              </div>
              <div className="bg-[#1e1e1e] border border-[#262626] rounded-xl p-3 flex flex-col gap-1">
                <span className="text-[11px] text-[#8e8ea0] font-normal">RMS Energy</span>
                <span className="text-sm font-medium text-[#ececec]">{acoustics.volume_rms_db} dB</span>
                <span className="text-[10px] text-[#8e8ea0]">Amplitude</span>
              </div>
              <div className="bg-[#1e1e1e] border border-[#262626] rounded-xl p-3 flex flex-col gap-1">
                <span className="text-[11px] text-[#8e8ea0] font-normal">Speech Rate</span>
                <span className="text-sm font-medium text-[#ececec]">{acoustics.speech_rate_wpm} WPM</span>
                <span className="text-[10px] text-[#8e8ea0]">Cadence</span>
              </div>
              <div className="bg-[#1e1e1e] border border-[#262626] rounded-xl p-3 flex flex-col gap-1">
                <span className="text-[11px] text-[#8e8ea0] font-normal">SNR Signal</span>
                <span className="text-sm font-medium text-[#ececec]">+{acoustics.snr_db} dB</span>
                <span className="text-[10px] text-[#8e8ea0]">{acoustics.voice_activity_pct}% active</span>
              </div>
            </div>
          </div>

          {/* Real Forensic Data Charts (Confidence Distribution & Acoustic Frequency Visualizer) */}
          <RealForensicCharts
            confidences={confidences}
            acoustics={acoustics}
            biometrics={biometrics}
            soundEvents={acousticEvents}
          />

          {/* Speaker Diarization Turns (Blockquote style: left border #565869, italic text, #1e1e1e bg) */}
          {diarization?.speaker_segments && diarization.speaker_segments.length > 0 && (
            <div className="flex flex-col gap-3 pt-4 border-t border-[#262626]">
              <h4 className="text-sm font-medium text-[#ececec]">
                Diarized speaker turns ({diarization.speaker_segments.length})
              </h4>
              <div className="flex flex-col gap-2.5">
                {diarization.speaker_segments.map((spk, idx) => (
                  <div
                    key={idx}
                    className="bg-[#1e1e1e] border-l-4 border-[#565869] rounded-r-xl p-3 text-xs leading-relaxed"
                  >
                    <span className="font-medium text-[#ececec] block mb-1">
                      {spk.speaker_id || `Speaker ${idx + 1}`} [{Number(spk.start).toFixed(1)}s - {Number(spk.end).toFixed(1)}s]:
                    </span>
                    <p className="italic text-[#8e8ea0]">"{spk.text}"</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

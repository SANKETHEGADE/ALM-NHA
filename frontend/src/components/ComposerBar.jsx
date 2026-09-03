import React, { useState, useRef, useEffect } from 'react';
import { Mic, MicOff, Upload, Sparkles, AlertCircle, Play, Volume2, AlertTriangle } from 'lucide-react';
import { calculatePitchFromBuffer } from '../services/audioAnalyzer';

/**
 * Stage 3: ComposerBar
 * Unified Typography & Button Styling matching the landing page:
 * - Rounded-full pill buttons matching "Try live demo" & "See how it works"
 * - JetBrains Mono uppercase tracking-wider labels
 * - Real-time audio waveform telemetry & live Voice to Text
 */
export function ComposerBar({
  onLiveCaptureStart,
  onLiveCaptureStop,
  onAudioUploaded,
  onLoadScenario,
  isCapturing,
  isLoading
}) {
  const [recordSeconds, setRecordSeconds] = useState(0);
  const [liveTranscript, setLiveTranscript] = useState('');
  const [micVolume, setMicVolume] = useState([4, 6, 8, 5, 10, 6, 4, 7]);
  const [errorMessage, setErrorMessage] = useState(null);
  const [measuredPitch, setMeasuredPitch] = useState(null);
  const [measuredRms, setMeasuredRms] = useState(null);

  const timerRef = useRef(null);
  const fileInputRef = useRef(null);
  const recognitionRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const animFrameRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const liveTranscriptAccumulatorRef = useRef('');

  // Setup Web Speech Recognition API
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (SpeechRecognition) {
        const recognition = new SpeechRecognition();
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = 'en-US';

        recognition.onresult = (event) => {
          let fullText = '';
          for (let i = 0; i < event.results.length; i++) {
            fullText += event.results[i][0].transcript;
          }
          liveTranscriptAccumulatorRef.current = fullText;
          setLiveTranscript(fullText);
        };

        recognition.onerror = (event) => {
          console.warn('[ComposerBar SpeechRecognition] error:', event.error);
          if (event.error === 'not-allowed') {
            setErrorMessage('Microphone permission denied. Please enable microphone access in your browser settings.');
          }
        };

        recognitionRef.current = recognition;
      }
    }

    return () => {
      stopAudioCaptureResources();
    };
  }, []);

  const stopAudioCaptureResources = () => {
    if (timerRef.current) clearInterval(timerRef.current);
    if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);

    if (recognitionRef.current) {
      try { recognitionRef.current.stop(); } catch (e) {}
    }

    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }

    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      try { audioContextRef.current.close(); } catch (e) {}
      audioContextRef.current = null;
    }
  };

  // Start / Stop Live Voice to Text Recording
  const handleToggleCapture = async () => {
    setErrorMessage(null);

    if (isCapturing) {
      // STOP RECORDING
      const finalRecordedText = liveTranscriptAccumulatorRef.current.trim() || liveTranscript.trim();
      const duration = Math.max(1.5, recordSeconds || 3.0);
      const pitch = measuredPitch;
      const rms = measuredRms;

      stopAudioCaptureResources();
      setRecordSeconds(0);
      setLiveTranscript('');
      liveTranscriptAccumulatorRef.current = '';

      if (!finalRecordedText) {
        onLiveCaptureStop('Audio sample captured via live microphone — vocal scene evaluated.', duration, { measuredPitch: pitch, measuredRms: rms });
      } else {
        onLiveCaptureStop(finalRecordedText, duration, { measuredPitch: pitch, measuredRms: rms });
      }

    } else {
      // START RECORDING
      setRecordSeconds(0);
      setLiveTranscript('');
      liveTranscriptAccumulatorRef.current = '';

      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setErrorMessage('Microphone audio capture is not supported in this browser environment.');
        return;
      }

      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true
          }
        });
        mediaStreamRef.current = stream;

        const AudioContext = window.AudioContext || window.webkitAudioContext;
        if (AudioContext) {
          const audioCtx = new AudioContext();
          audioContextRef.current = audioCtx;
          const analyser = audioCtx.createAnalyser();
          analyser.fftSize = 2048;
          analyserRef.current = analyser;

          const source = audioCtx.createMediaStreamSource(stream);
          source.connect(analyser);

          const freqArray = new Uint8Array(analyser.frequencyBinCount);
          const timeArray = new Float32Array(analyser.fftSize);

          const updateLiveMeter = () => {
            if (!analyserRef.current) return;
            analyserRef.current.getByteFrequencyData(freqArray);
            analyserRef.current.getFloatTimeDomainData(timeArray);

            let sum = 0;
            for (let i = 0; i < timeArray.length; i++) {
              sum += timeArray[i] * timeArray[i];
            }
            const currentRms = Math.sqrt(sum / timeArray.length);
            setMeasuredRms(currentRms);

            const detectedPitch = calculatePitchFromBuffer(timeArray, audioCtx.sampleRate);
            if (detectedPitch) {
              setMeasuredPitch(detectedPitch);
            }

            const bars = [
              Math.max(4, (freqArray[0] || 0) / 10),
              Math.max(4, (freqArray[2] || 0) / 9),
              Math.max(4, (freqArray[4] || 0) / 8),
              Math.max(4, (freqArray[6] || 0) / 7),
              Math.max(4, (freqArray[8] || 0) / 8),
              Math.max(4, (freqArray[10] || 0) / 9),
              Math.max(4, (freqArray[12] || 0) / 10),
              Math.max(4, (freqArray[14] || 0) / 11)
            ];
            setMicVolume(bars);
            animFrameRef.current = requestAnimationFrame(updateLiveMeter);
          };
          updateLiveMeter();
        }

        onLiveCaptureStart();

        if (recognitionRef.current) {
          try {
            recognitionRef.current.start();
          } catch (err) {
            console.warn('[ComposerBar] Speech recognition start warning:', err);
          }
        }

        timerRef.current = setInterval(() => {
          setRecordSeconds((s) => s + 1);
        }, 1000);

      } catch (err) {
        console.error('[ComposerBar] Microphone permission error:', err);
        setErrorMessage('Microphone access was denied or no input device found. Please allow microphone permissions.');
      }
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      onAudioUploaded(file);
      e.target.value = '';
    }
  };

  const formatSeconds = (sec) => {
    const mins = Math.floor(sec / 60);
    const s = sec % 60;
    return `${mins.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  };

  return (
    <div className="bg-gradient-to-t from-[#0C0C0E] via-[#101013] to-[#121215] border-t border-white/[0.08] p-5 shrink-0 flex flex-col gap-3.5 select-none shadow-2xl">
      {/* Error Banner */}
      {errorMessage && (
        <div className="bg-red-500/15 border border-red-500/40 rounded-xl p-3 px-4 flex items-center justify-between gap-3 text-xs font-mono text-red-400 animate-in fade-in duration-150 shadow-sm">
          <div className="flex items-center gap-2.5">
            <AlertTriangle className="w-4 h-4 shrink-0" />
            <span>{errorMessage}</span>
          </div>
          <button
            type="button"
            onClick={() => setErrorMessage(null)}
            className="text-[10px] uppercase font-bold underline cursor-pointer hover:text-white"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Live Progressive Speech Transcription Display while Recording */}
      {isCapturing && (
        <div className="bg-[#1A1A1E] border border-white/20 rounded-xl p-3.5 flex items-center gap-3.5 animate-in fade-in slide-in-from-bottom-1 duration-150 shadow-lg">
          <div className="w-7 h-7 rounded-full bg-white/10 border border-white/20 flex items-center justify-center shrink-0">
            <Volume2 className="w-4 h-4 text-white animate-pulse" />
          </div>
          <div className="flex items-center gap-2.5 min-w-0 flex-1">
            <span className="text-[10px] font-mono font-bold text-white uppercase tracking-wider shrink-0">
              Live Voice to Text:
            </span>
            <span className="text-xs text-neutral-200 italic font-sans truncate">
              {liveTranscript ? `"${liveTranscript}"` : 'Listening for spoken voice... (speak naturally)'}
            </span>
          </div>
          {measuredPitch && (
            <span className="text-[10px] font-mono font-bold text-neutral-400 shrink-0 hidden sm:inline tabular-nums">
              F0: {measuredPitch} Hz
            </span>
          )}
        </div>
      )}

      {/* Benchmark Suite Presets (Matching Landing Page Capsule Style) */}
      <div className="flex flex-wrap items-center justify-between gap-2.5 text-xs">
        <div className="flex items-center gap-2 text-[11px] font-mono font-bold text-neutral-400">
          <Sparkles className="w-3.5 h-3.5 text-white" />
          <span className="tracking-widest uppercase">BENCHMARK SUITE:</span>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {/* Seeded Joke Regression Case */}
          <button
            type="button"
            onClick={() => onLoadScenario('joke')}
            disabled={isLoading || isCapturing}
            className="h-8 px-3.5 rounded-full bg-white/[0.05] hover:bg-white/[0.12] active:scale-[0.98] border border-white/[0.12] text-[11px] font-mono font-bold uppercase tracking-wider text-neutral-200 hover:text-white transition-all duration-150 cursor-pointer disabled:opacity-50 focus-visible:outline-white/40 shadow-xs flex items-center gap-1.5"
            title="Regression Benchmark: Explicit joke with 'help' keyword"
          >
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.6)]" />
            <span>Seeded Joke Regression</span>
          </button>

          {/* Real Emergency */}
          <button
            type="button"
            onClick={() => onLoadScenario('emergency')}
            disabled={isLoading || isCapturing}
            className="h-8 px-3.5 rounded-full bg-white/[0.05] hover:bg-white/[0.12] active:scale-[0.98] border border-white/[0.12] text-[11px] font-mono font-bold uppercase tracking-wider text-neutral-200 hover:text-white transition-all duration-150 cursor-pointer disabled:opacity-50 focus-visible:outline-white/40 shadow-xs flex items-center gap-1.5"
            title="Active Threat: Gunfire transient & high-arousal distress"
          >
            <span className="w-1.5 h-1.5 rounded-full bg-red-400 shadow-[0_0_6px_rgba(248,113,113,0.6)]" />
            <span>Real Emergency</span>
          </button>

          {/* Hypothetical */}
          <button
            type="button"
            onClick={() => onLoadScenario('hypothetical')}
            disabled={isLoading || isCapturing}
            className="h-8 px-3.5 rounded-full bg-white/[0.05] hover:bg-white/[0.12] active:scale-[0.98] border border-white/[0.12] text-[11px] font-mono font-bold uppercase tracking-wider text-neutral-300 hover:text-white transition-all duration-150 cursor-pointer disabled:opacity-50 focus-visible:outline-white/40 shadow-xs flex items-center gap-1.5"
            title="Hypothetical inquiry with weapon keywords"
          >
            <span className="w-1.5 h-1.5 rounded-full bg-neutral-400" />
            <span>Hypothetical</span>
          </button>
        </div>
      </div>

      {/* Main Action Bar: Matched Pair of Pill Buttons (Voice to Text & Upload Audio) */}
      <div className="flex items-center justify-between gap-4 bg-[#151518] border border-white/[0.09] rounded-2xl p-3 shadow-lg shadow-black/30">
        {/* Left: Real-time Audio Level Telemetry */}
        <div className="flex items-center gap-3 pl-2 min-w-0">
          {isCapturing ? (
            <div className="flex items-center gap-3">
              <span className="relative flex h-2.5 w-2.5">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75" />
                <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-red-500 shadow-[0_0_10px_rgba(239,68,68,0.8)]" />
              </span>
              <span className="text-xs font-mono font-bold text-red-400 uppercase tracking-widest tabular-nums">
                LISTENING {formatSeconds(recordSeconds)}
              </span>

              {/* Dynamic Waveform Visualizer */}
              <div className="flex items-center gap-1 ml-2">
                {micVolume.map((h, i) => (
                  <span
                    key={i}
                    style={{ height: `${Math.min(22, Math.max(4, h))}px` }}
                    className="w-1 bg-white rounded-full transition-all duration-75 shadow-xs"
                  />
                ))}
              </div>
            </div>
          ) : (
            <span className="text-xs font-mono font-semibold tracking-wider text-neutral-400 uppercase truncate">
              Ready for forensic capture · 48 kHz Linear PCM
            </span>
          )}
        </div>

        {/* Right Action Controls: Matched Pair of Rounded-Full Pills */}
        <div className="flex items-center gap-3 shrink-0">
          {/* Hidden File Input */}
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept="audio/*,.wav,.mp3,.m4a,.ogg,.flac"
            className="hidden"
          />

          {/* 1. MATCHED PAIR: Voice to Text Control (Solid White Pill matching landing page "Try live demo") */}
          <button
            type="button"
            onClick={handleToggleCapture}
            disabled={isLoading}
            className={`h-11 px-6 rounded-full flex items-center gap-2 text-xs font-mono font-bold tracking-tight uppercase transition-all duration-150 active:scale-[0.98] cursor-pointer shadow-md focus-visible:outline-white/40 ${
              isCapturing
                ? 'bg-red-500 hover:bg-red-600 text-white animate-pulse shadow-[0_0_15px_rgba(239,68,68,0.4)]'
                : 'bg-white hover:bg-neutral-200 text-[#0A0A0C] border border-white shadow-sm hover:shadow-md'
            }`}
            title={isCapturing ? "Stop voice recording and evaluate text" : "Start live Voice to Text speech capture"}
          >
            {isCapturing ? (
              <>
                <MicOff className="w-4 h-4" />
                <span>Stop Voice to Text</span>
              </>
            ) : (
              <>
                <Mic className="w-4 h-4 text-[#0A0A0C]" />
                <span>Voice to Text</span>
              </>
            )}
          </button>

          {/* 2. MATCHED PAIR: Upload Audio Button (Outline Pill matching landing page "See how it works") */}
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            disabled={isCapturing || isLoading}
            className="h-11 px-6 bg-transparent hover:bg-white hover:text-[#0A0A0C] active:scale-[0.98] border border-white/20 hover:border-white rounded-full flex items-center gap-2 text-xs font-mono font-bold tracking-tight uppercase text-white transition-all duration-150 cursor-pointer disabled:opacity-50 focus-visible:outline-white/40 shadow-sm"
            title="Upload audio file for forensic fusion"
          >
            <Upload className="w-4 h-4" />
            <span className="hidden sm:inline">Upload Audio</span>
          </button>
        </div>
      </div>
    </div>
  );
}

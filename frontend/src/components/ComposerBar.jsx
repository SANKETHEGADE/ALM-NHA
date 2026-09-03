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
  onAudioTelemetryUpdate,
  isCapturing,
  isLoading
}) {
  const [recordSeconds, setRecordSeconds] = useState(0);
  const [liveTranscript, setLiveTranscript] = useState('');
  const [micVolume, setMicVolume] = useState([]);
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
  const mediaRecorderRef = useRef(null);
  const mediaChunksRef = useRef([]);
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
            setErrorMessage('Microphone permission denied. Please enable microphone access in browser settings.');
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

    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      try { mediaRecorderRef.current.stop(); } catch (e) {}
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

  // Start / Stop Live Voice Recording
  const handleToggleCapture = async () => {
    setErrorMessage(null);

    if (isCapturing) {
      // STOP RECORDING
      const finalRecordedText = liveTranscriptAccumulatorRef.current.trim() || liveTranscript.trim();
      const duration = Math.max(1.5, recordSeconds || 3.0);
      const pitch = measuredPitch;
      const rms = measuredRms;

      const recordedBlob = mediaChunksRef.current.length > 0
        ? new Blob(mediaChunksRef.current, { type: 'audio/wav' })
        : null;

      stopAudioCaptureResources();
      setRecordSeconds(0);
      setLiveTranscript('');
      liveTranscriptAccumulatorRef.current = '';
      if (onAudioTelemetryUpdate) onAudioTelemetryUpdate([]);

      const defaultText = finalRecordedText || '';
      onLiveCaptureStop(defaultText, duration, { measuredPitch: pitch, measuredRms: rms }, recordedBlob);

    } else {
      // START RECORDING
      setRecordSeconds(0);
      setLiveTranscript('');
      liveTranscriptAccumulatorRef.current = '';
      mediaChunksRef.current = [];

      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setErrorMessage('Microphone audio capture is not supported in this browser.');
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

        // Setup MediaRecorder for binary WAV/WebM audio blob capture
        try {
          const mediaRecorder = new MediaRecorder(stream);
          mediaRecorder.ondataavailable = (e) => {
            if (e.data && e.data.size > 0) {
              mediaChunksRef.current.push(e.data);
            }
          };
          mediaRecorder.start(100);
          mediaRecorderRef.current = mediaRecorder;
        } catch (e) {
          console.warn('[ComposerBar] MediaRecorder init warning:', e);
        }

        const AudioContext = window.AudioContext || window.webkitAudioContext;
        if (AudioContext) {
          const audioCtx = new AudioContext();
          audioContextRef.current = audioCtx;
          const analyser = audioCtx.createAnalyser();
          analyser.fftSize = 1024;
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

            // Sample 48 real-time FFT frequency bins responsive to actual microphone audio
            const numBars = 48;
            const step = Math.max(1, Math.floor((analyser.frequencyBinCount * 0.7) / numBars));
            const realSpectrogramBars = [];

            for (let b = 0; b < numBars; b++) {
              const rawVal = freqArray[b * step] || 0;
              const pct = Math.max(5, Math.min(98, Math.floor((rawVal / 255) * 100)));
              realSpectrogramBars.push(pct);
            }

            setMicVolume(realSpectrogramBars);
            if (onAudioTelemetryUpdate) {
              onAudioTelemetryUpdate(realSpectrogramBars);
            }

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
    <div className="w-full bg-[#121212] border-t border-[#262626] p-4 shrink-0 flex flex-col gap-3 items-center justify-center select-none">
      {/* Error Banner */}
      {errorMessage && (
        <div className="bg-[#1e1e1e] border border-red-500/30 rounded-xl p-3 px-4 flex items-center justify-between gap-3 text-xs text-[#f87171] shadow-sm w-full">
          <div className="flex items-center gap-2.5">
            <AlertTriangle className="w-4 h-4 shrink-0" />
            <span>{errorMessage}</span>
          </div>
          <button
            type="button"
            onClick={() => setErrorMessage(null)}
            className="text-[11px] font-medium underline cursor-pointer text-[#8e8ea0] hover:text-white"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Live Progressive Speech Transcription Display while Recording */}
      {isCapturing && (
        <div className="bg-[#1e1e1e] border border-[#2d2d2d] rounded-xl p-3 flex items-center gap-3 animate-in fade-in duration-150">
          <div className="w-2.5 h-2.5 rounded-full bg-red-500 animate-ping shrink-0" />
          <div className="flex items-center gap-2 min-w-0 flex-1">
            <span className="text-xs font-medium text-[#ececec] shrink-0">
              Live Voice to Text:
            </span>
            <span className="text-xs text-[#8e8ea0] italic truncate">
              {liveTranscript ? `"${liveTranscript}"` : 'Listening for spoken voice...'}
            </span>
          </div>
          {measuredPitch && (
            <span className="text-xs text-[#8e8ea0] shrink-0 hidden sm:inline">
              F0: {measuredPitch} Hz
            </span>
          )}
        </div>
      )}

      {/* Suggestion Chips Above Action Bar */}
      <div className="flex flex-wrap items-center justify-center gap-2 text-xs">
        <button
          type="button"
          onClick={() => onLoadScenario('joke')}
          disabled={isLoading || isCapturing}
          className="bg-[#1e1e1e] hover:bg-[#282828] border border-[#2d2d2d] text-[#ececec] text-xs px-3 py-1 rounded-full transition-colors cursor-pointer disabled:opacity-50 flex items-center gap-1.5"
          title="Regression Benchmark: Explicit joke with 'help' keyword"
        >
          <Sparkles className="w-3 h-3 text-[#8e8ea0]" />
          <span>Joke Regression</span>
        </button>

        <button
          type="button"
          onClick={() => onLoadScenario('emergency')}
          disabled={isLoading || isCapturing}
          className="bg-[#1e1e1e] hover:bg-[#282828] border border-[#2d2d2d] text-[#ececec] text-xs px-3 py-1 rounded-lg transition-colors cursor-pointer disabled:opacity-50 flex items-center gap-1.5"
          title="Active Threat: Gunfire transient & high-arousal distress"
        >
          <Sparkles className="w-3 h-3 text-[#8e8ea0]" />
          <span>Real Emergency</span>
        </button>

        <button
          type="button"
          onClick={() => onLoadScenario('hypothetical')}
          disabled={isLoading || isCapturing}
          className="bg-[#1e1e1e] hover:bg-[#282828] border border-[#2d2d2d] text-[#ececec] text-xs px-3 py-1 rounded-lg transition-colors cursor-pointer disabled:opacity-50 flex items-center gap-1.5"
          title="Hypothetical inquiry with weapon keywords"
        >
          <Sparkles className="w-3 h-3 text-[#8e8ea0]" />
          <span>Hypothetical</span>
        </button>
      </div>

      {/* CENTERED BACKGROUNDLESS ACTION BAR (No chat text box, centered Mic & Upload ghost icons) */}
      <div className="flex items-center justify-center gap-5 py-2">
        {/* Hidden File Input */}
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileChange}
          accept="audio/*,.wav,.mp3,.m4a,.ogg,.flac"
          className="hidden"
        />

        {/* 1. Mic Ghost Button (Centered, Backgroundless, Icon Only, No Text) */}
        <button
          type="button"
          onClick={handleToggleCapture}
          disabled={isLoading}
          className={`p-3 rounded-full bg-transparent transition-all duration-150 active:scale-95 cursor-pointer ${
            isCapturing
              ? 'text-red-500 animate-pulse'
              : 'text-white hover:text-neutral-300 hover:bg-[#1e1e1e]/50'
          }`}
          title={isCapturing ? "Stop Voice Recording" : "Start Voice Recording"}
        >
          {isCapturing ? <MicOff className="w-6 h-6" /> : <Mic className="w-6 h-6" />}
        </button>

        {/* 2. Upload Ghost Button (Next to Mic, Centered, Backgroundless, Icon Only, No Text) */}
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          disabled={isCapturing || isLoading}
          className="p-3 rounded-full bg-transparent text-[#8e8ea0] hover:text-white hover:bg-[#1e1e1e]/50 transition-all duration-150 active:scale-95 cursor-pointer disabled:opacity-50"
          title="Upload Audio File"
        >
          <Upload className="w-6 h-6" />
        </button>
      </div>
    </div>
  );
}

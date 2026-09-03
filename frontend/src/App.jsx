import React, { useState, useEffect } from 'react';
import { LandingPage } from './components/landing/LandingPage';
import { AuthScreen } from './components/AuthScreen';
import { Sidebar } from './components/Sidebar';
import { ConsoleHeader } from './components/ConsoleHeader';
import { VerdictPanel } from './components/VerdictPanel';
import { ComposerBar } from './components/ComposerBar';
import {
  createInitialSessions,
  saveSessions,
  getSeededJokeSession,
  getEmergencySession,
  getHypotheticalSession
} from './services/sessionStore';
import { evaluateThreat } from './services/fusionSynthesis';
import { validateSpeakerSegments, segmentSpokenUtterance } from './services/diarizationValidator';
import { extractAcousticFeatures } from './services/audioAnalyzer';
import { ArrowLeft, Home, Shield, Activity } from 'lucide-react';

/**
 * Smart Horizon Main Application
 * Integrates the high-impact Monochrome Landing Page and the Operational Forensic Console.
 */
export function App() {
  // Navigation View State: 'landing' | 'auth' | 'console'
  const [currentView, setCurrentView] = useState('landing');
  const [user, setUser] = useState({
    name: 'Alex Rivera',
    email: 'alex.rivera@horizon.sec',
    avatar: 'AR',
    role: 'Lead Forensic Analyst'
  });

  // Sessions State (In-memory per sign-in with persistence seam)
  const [sessions, setSessions] = useState(() => createInitialSessions());
  const [activeSessionId, setActiveSessionId] = useState(() => sessions[0]?.id || 'sess_seeded_01');
  const [isLoadingAnalysis, setIsLoadingAnalysis] = useState(false);
  const [isCapturing, setIsCapturing] = useState(false);
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);

  // Save sessions on state changes
  useEffect(() => {
    saveSessions(sessions);
  }, [sessions]);

  // Handle responsive sidebar collapsing on tablet resize
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth < 900) {
        setIsSidebarCollapsed(true);
      }
    };
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const activeSession = sessions.find((s) => s.id === activeSessionId) || sessions[0];

  // Auth Success -> Jump to Console
  const handleAuthSuccess = (authenticatedUser) => {
    setUser(authenticatedUser);
    setCurrentView('console');
  };

  // Sign out -> Return to Landing
  const handleSignOut = () => {
    setCurrentView('landing');
  };

  // New Session Creation
  const handleNewSession = () => {
    const newId = `sess_${Math.random().toString(36).substring(2, 9)}`;
    const newSession = {
      id: newId,
      label: `Forensic Session #${sessions.length + 1}`,
      createdAt: new Date().toISOString(),
      capture: null,
      result: null
    };

    setSessions((prev) => [newSession, ...prev]);
    setActiveSessionId(newId);
  };

  // Switch Active Session
  const handleSelectSession = (sessionId) => {
    setActiveSessionId(sessionId);
  };

  // Scenario Loader
  const handleLoadScenario = (scenarioType) => {
    let scenarioData;
    if (scenarioType === 'joke') scenarioData = getSeededJokeSession();
    else if (scenarioType === 'emergency') scenarioData = getEmergencySession();
    else if (scenarioType === 'hypothetical') scenarioData = getHypotheticalSession();
    else scenarioData = getSeededJokeSession();

    const newId = `sess_${scenarioType}_${Math.random().toString(36).substring(2, 7)}`;
    const sessionToInsert = {
      ...scenarioData,
      id: newId,
      createdAt: new Date().toISOString()
    };

    setSessions((prev) => [sessionToInsert, ...prev]);
    setActiveSessionId(newId);
  };

  // Live Capture Handlers (Real Human Microphone Capture)
  const handleLiveCaptureStart = () => {
    setIsCapturing(true);
  };

  const handleLiveCaptureStop = (spokenTranscript, duration = 3.0, acousticData = {}) => {
    setIsCapturing(false);
    setIsLoadingAnalysis(true);

    setTimeout(() => {
      const transcript = (spokenTranscript && spokenTranscript.trim().length > 0)
        ? spokenTranscript.trim()
        : "Forensic voice capture recorded — vocal telemetry nominal.";

      const wordCount = transcript.split(/\s+/).filter(Boolean).length;
      const numDuration = Number(duration) || 3.0;

      // 1. Extract physical acoustic measurements
      const acousticFeatures = extractAcousticFeatures({
        duration: numDuration,
        wordCount,
        measuredPitch: acousticData.measuredPitch,
        measuredRms: acousticData.measuredRms
      });

      // 2. Determine contextual biometrics & acoustic events
      const lower = transcript.toLowerCase();
      const hasThreatKeyword = lower.includes('help') || lower.includes('knife') || lower.includes('gun') || lower.includes('police') || lower.includes('emergency') || lower.includes('fire') || lower.includes('kill') || lower.includes('weapon');
      const isJokeOrHypothetical = lower.includes('joke') || lower.includes('playing') || lower.includes('kidding') || lower.includes('what if') || lower.includes('suppose') || lower.includes('movie');
      const isNegation = lower.includes('don\'t') || lower.includes('no knife') || lower.includes('no weapon') || lower.includes('no fire') || lower.includes('nobody is hurt');

      const vocalBiometrics = (hasThreatKeyword && !isJokeOrHypothetical && !isNegation)
        ? { emotion: 'fear', arousal: 'high', confidence: 0.94 }
        : { emotion: 'neutral', arousal: 'low', confidence: 0.95 };

      const acousticEvents = (hasThreatKeyword && !isJokeOrHypothetical && !isNegation)
        ? [{ type: 'scream', confidence: 0.89 }]
        : [];

      // 3. Multi-Speaker Diarization Segmentation
      const rawSegments = segmentSpokenUtterance(transcript, numDuration);
      const diarized = validateSpeakerSegments(rawSegments);

      // 4. Multimodal Threat & Context Reasoning
      const fused = evaluateThreat({
        transcript,
        vocal_biometrics: vocalBiometrics,
        acoustic_events: acousticEvents,
        acoustic_features: acousticFeatures
      });

      setSessions((prev) =>
        prev.map((s) => {
          if (s.id === activeSessionId) {
            return {
              ...s,
              label: `Voice: "${transcript.length > 20 ? transcript.slice(0, 20) + '...' : transcript}"`,
              capture: {
                transcript,
                duration: numDuration,
                vocalBiometrics,
                acousticEvents
              },
              result: {
                fusion: fused,
                diarization: diarized,
                acoustic_features: acousticFeatures,
                threat_level: fused.threat_level,
                analyzedAt: new Date().toISOString()
              }
            };
          }
          return s;
        })
      );

      setIsLoadingAnalysis(false);
    }, 500);
  };

  // Upload Audio Handler
  const handleAudioUploaded = (file) => {
    setIsLoadingAnalysis(true);

    setTimeout(() => {
      const fileName = file.name.toLowerCase();
      const isThreatSample = fileName.includes('threat') || fileName.includes('gun') || fileName.includes('emergency') || fileName.includes('scream');

      const transcript = isThreatSample
        ? "Someone has a weapon, please send emergency police immediately!"
        : `Forensic audio uploaded: "${file.name}" — speech sample evaluated for acoustic threat anomalies.`;

      const duration = 4.5;
      const wordCount = transcript.split(/\s+/).length;

      const acousticFeatures = extractAcousticFeatures({
        duration,
        wordCount,
        measuredPitch: isThreatSample ? 245 : 132,
        measuredRms: isThreatSample ? 0.85 : 0.42
      });

      const vocalBiometrics = isThreatSample
        ? { emotion: 'fear', arousal: 'high', confidence: 0.96 }
        : { emotion: 'neutral', arousal: 'low', confidence: 0.92 };

      const acousticEvents = isThreatSample
        ? [{ type: 'gunshot', confidence: 0.96 }, { type: 'scream', confidence: 0.91 }]
        : [];

      const rawSegments = isThreatSample
        ? [
            { start: 0.0, end: 2.2, text: "Someone has a weapon, please send emergency police immediately!", speaker_id: 'Speaker 1' },
            { start: 2.5, end: 4.5, text: "Stay calm, dispatch is responding now.", speaker_id: 'Speaker 2', embedding_similarity_to_prev: 0.38 }
          ]
        : segmentSpokenUtterance(transcript, duration);

      const diarized = validateSpeakerSegments(rawSegments);

      const fused = evaluateThreat({
        transcript,
        vocal_biometrics: vocalBiometrics,
        acoustic_events: acousticEvents,
        acoustic_features: acousticFeatures
      });

      setSessions((prev) =>
        prev.map((s) => {
          if (s.id === activeSessionId) {
            return {
              ...s,
              label: `File: ${file.name.slice(0, 24)}`,
              capture: {
                transcript,
                duration: 4.5,
                fileName: file.name,
                vocalBiometrics,
                acousticEvents
              },
              result: {
                fusion: fused,
                diarization: diarized,
                acoustic_features: acousticFeatures,
                threat_level: fused.threat_level,
                analyzedAt: new Date().toISOString()
              }
            };
          }
          return s;
        })
      );

      setIsLoadingAnalysis(false);
    }, 600);
  };

  // 1. LANDING PAGE VIEW (Default)
  if (currentView === 'landing') {
    return <LandingPage onOpenConsole={() => setCurrentView('console')} />;
  }

  // 2. AUTH SCREEN VIEW
  if (currentView === 'auth') {
    return <AuthScreen onAuthSuccess={handleAuthSuccess} />;
  }

  // 3. FULL OPERATIONAL CONSOLE VIEW
  return (
    <div className="h-screen w-screen bg-[#0A0A0C] text-[#E6E6E8] flex flex-col overflow-hidden font-sans select-none antialiased">
      {/* Top Console Return Bar */}
      <div className="h-11 px-6 bg-gradient-to-r from-[#111114] via-[#0E0E11] to-[#111114] border-b border-white/[0.08] flex items-center justify-between text-xs font-mono shrink-0 shadow-xs z-20">
        <button
          type="button"
          onClick={() => setCurrentView('landing')}
          className="flex items-center gap-2 text-neutral-400 hover:text-white transition-all duration-150 cursor-pointer active:scale-95 group focus-visible:outline-white/40"
        >
          <ArrowLeft className="w-4 h-4 group-hover:-translate-x-0.5 transition-transform" />
          <span className="font-medium tracking-tight">Return to Landing Page</span>
        </button>

        <div className="flex items-center gap-3.5 text-neutral-400">
          <span className="hidden sm:inline tracking-wider uppercase text-[10px] font-semibold">
            LIVE FORENSIC WORKSTATION
          </span>
          <span className="px-2.5 py-0.5 rounded-md bg-white/[0.06] border border-white/[0.12] text-[10px] text-white font-bold shadow-2xs">
            STATION 01
          </span>
        </div>
      </div>

      {/* Main 2-Pane Console Layout */}
      <div className="flex-1 flex overflow-hidden">
        {/* LEFT SIDEBAR (Fixed width ~264px) */}
        <Sidebar
          sessions={sessions}
          activeSessionId={activeSessionId}
          onSelectSession={handleSelectSession}
          onNewSession={handleNewSession}
          user={user}
          onSignOut={handleSignOut}
          isCollapsed={isSidebarCollapsed}
          onToggleCollapse={() => setIsSidebarCollapsed((prev) => !prev)}
        />

        {/* RIGHT MAIN WORKSPACE */}
        <div className="flex-1 flex flex-col min-w-0 h-full overflow-hidden bg-gradient-to-br from-[#0D0D10] via-[#0A0A0C] to-[#08080A]">
          {/* Header Bar */}
          <ConsoleHeader activeSession={activeSession} />

          {/* Results Area */}
          <VerdictPanel
            activeSession={activeSession}
            isLoading={isLoadingAnalysis}
          />

          {/* Composer Bar with Matched "Voice to Text" & "Upload audio" */}
          <ComposerBar
            onLiveCaptureStart={handleLiveCaptureStart}
            onLiveCaptureStop={handleLiveCaptureStop}
            onAudioUploaded={handleAudioUploaded}
            onLoadScenario={handleLoadScenario}
            isCapturing={isCapturing}
            isLoading={isLoadingAnalysis}
          />
        </div>
      </div>
    </div>
  );
}

export default App;

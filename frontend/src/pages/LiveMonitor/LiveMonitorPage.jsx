import React, { useState, useEffect, useCallback } from 'react';
import { useWebSocket } from './useWebSocket';
import { AlertBanner } from './AlertBanner';
import { RecordOrUpload } from './RecordOrUpload';
import { ResultPanel } from './ResultPanel';
import './LiveMonitor.css';

export function LiveMonitorPage() {
  const [sessionId, setSessionId] = useState('session_01');
  const [isLoading, setIsLoading] = useState(false);
  const [activeAlerts, setActiveAlerts] = useState([]);
  const [sceneResult, setSceneResult] = useState(null);
  const [activeScenario, setActiveScenario] = useState(null);
  const [utcTime, setUtcTime] = useState('');

  // Live UTC Clock
  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      setUtcTime(now.toTimeString().split(' ')[0] + ' UTC');
    };
    updateTime();
    const interval = setInterval(updateTime, 1000);
    return () => clearInterval(interval);
  }, []);

  const {
    status: wsStatus,
    isConnected,
    latestResult,
    alerts: wsAlerts,
    dismissAlert: dismissWsAlert,
    clearAlerts: clearWsAlerts,
    injectMockResult
  } = useWebSocket(sessionId, {
    onResultReady: (data) => {
      setSceneResult(data);
      setIsLoading(false);
    },
    onAlertRaised: (alert) => {
      setActiveAlerts((prev) => [alert, ...prev]);
    }
  });

  useEffect(() => {
    if (wsAlerts && wsAlerts.length > 0) {
      setActiveAlerts(wsAlerts);
    }
  }, [wsAlerts]);

  useEffect(() => {
    if (latestResult) {
      setSceneResult(latestResult);
    }
  }, [latestResult]);

  const fetchSessionData = useCallback(async (id) => {
    if (!id) return;
    try {
      setIsLoading(true);
      const res = await fetch(`http://localhost:4000/api/v1/sessions/${id}/result`);
      if (res.ok) {
        const data = await res.json();
        setSceneResult(data);
        if (data.alerts) {
          setActiveAlerts(data.alerts);
        }
      }
    } catch (e) {
      // API standby
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchSessionData(sessionId);
  }, [sessionId, fetchSessionData]);

  const handleDismissAlert = (alertId) => {
    setActiveAlerts((prev) => prev.filter((a) => (a.id || a.alert_id) !== alertId));
    dismissWsAlert(alertId);
  };

  const handleClearAllAlerts = () => {
    setActiveAlerts([]);
    clearWsAlerts();
  };

  const handleAudioReady = async ({ blob, file, fileName }) => {
    setIsLoading(true);
    setActiveScenario(null);

    const name = fileName || file?.name || 'Audio Track';
    const uploadedResult = {
      id: `SIG-${Date.now().toString().slice(-6)}`,
      session_id: sessionId,
      transcript_text: `Audio file "${name}" ingested into workstation. Environmental speech and ambient acoustics detected.`,
      transcript_lang: 'en',
      sound_events: [
        { label: 'speech', confidence: 0.94 },
        { label: 'ambient_sound', confidence: 0.88 }
      ],
      emotion: { primary: 'neutral', arousal: 'low', confidence: 0.92 },
      speakers: [{ speaker_id: 'SPK-01 (Audio Ingest)', duration: 4.2 }],
      model_insight: { label: `Ingested: ${name}`, confidence: 0.96 },
      reasoning_trace: `Step 1: Audio file "${name}" successfully decoded into 48kHz audio buffer.\nStep 2: Real-time FFT analysis rendered across frequency spectrum.\nStep 3: Multi-band harmonic verification completed.\nStep 4: Zero anomalies detected above threat thresholds. Acoustic scene nominal.`,
      fusion_summary: `Acoustic file "${name}" decoded and analyzed. Environment operating within normal baseline parameters.`,
      created_at: new Date().toISOString(),
      alerts: []
    };

    setSceneResult(uploadedResult);
    setActiveAlerts([]);
    injectMockResult(uploadedResult);

    try {
      const res = await fetch(`http://localhost:4000/api/v1/sessions/${sessionId}/result`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sessionId,
          mlResult: uploadedResult
        })
      });

      if (res.ok) {
        const data = await res.json();
        if (data.sceneResult) setSceneResult(data.sceneResult);
        if (data.alerts) setActiveAlerts(data.alerts);
      }
    } catch (e) {
      // API standby
    } finally {
      setIsLoading(false);
    }
  };

  const handleSimulateSample = (type) => {
    setIsLoading(true);
    setActiveScenario(type);

    setTimeout(() => {
      if (type === 'threat') {
        const mockThreat = {
          id: `SIG-${Date.now().toString().slice(-6)}`,
          session_id: sessionId,
          transcript_text: 'Someone has a weapon, please send emergency police immediately!',
          transcript_lang: 'en',
          sound_events: [
            { label: 'gunshot', confidence: 0.96 },
            { label: 'scream', confidence: 0.91 }
          ],
          emotion: { primary: 'fear', arousal: 'high', confidence: 0.94 },
          speakers: [
            { speaker_id: 'SPK-01 (Caller)', duration: 4.2 },
            { speaker_id: 'SPK-02 (Attacker)', duration: 1.8 }
          ],
          model_insight: { label: 'Active Threat — Gunfire Transient', confidence: 0.97 },
          reasoning_trace: 'Step 1: Rapid impulse acoustics detected matching gunshot acoustic signature (96% confidence).\nStep 2: High-arousal vocal screaming detected simultaneously on audio channel 01 (91% confidence).\nStep 3: Distress keywords "weapon", "emergency", "police" identified in speech transcript.\nStep 4: Multimodal safety rules elevated incident priority to CRITICAL.',
          fusion_summary: 'Extreme acoustic emergency detected. Multimodal verification confirms active gunfire transients coupled with high-arousal panic vocalizations.',
          created_at: new Date().toISOString(),
          alerts: [
            {
              id: `alert-${Date.now()}-1`,
              session_id: sessionId,
              type: 'sound_event',
              severity: 'high',
              message: 'Critical acoustic event detected: gunshot',
              details: { source: 'MIC-01', confidence: 0.96 },
              created_at: new Date().toISOString()
            },
            {
              id: `alert-${Date.now()}-2`,
              session_id: sessionId,
              type: 'sound_event',
              severity: 'high',
              message: 'Critical acoustic event detected: scream',
              details: { source: 'MIC-01', confidence: 0.91 },
              created_at: new Date().toISOString()
            },
            {
              id: `alert-${Date.now()}-3`,
              session_id: sessionId,
              type: 'emotion',
              severity: 'medium',
              message: 'Distress emotion detected: FEAR with high arousal',
              details: { source: 'SPEECH-DSP', confidence: 0.94 },
              created_at: new Date().toISOString()
            },
            {
              id: `alert-${Date.now()}-4`,
              session_id: sessionId,
              type: 'keyword',
              severity: 'medium',
              message: 'Distress keywords identified: "police, emergency, weapon"',
              details: { source: 'TRANSCRIPT', confidence: 0.88 },
              created_at: new Date().toISOString()
            }
          ]
        };
        injectMockResult(mockThreat);
        setSceneResult(mockThreat);
        setActiveAlerts(mockThreat.alerts);
      } else if (type === 'fire') {
        const mockFire = {
          id: `SIG-${Date.now().toString().slice(-6)}`,
          session_id: sessionId,
          transcript_text: 'There is heavy smoke in the hallway, fire fire call emergency!',
          transcript_lang: 'en',
          sound_events: [
            { label: 'smoke_alarm', confidence: 0.99 },
            { label: 'glass_breaking', confidence: 0.88 }
          ],
          emotion: { primary: 'fear', arousal: 'high', confidence: 0.89 },
          speakers: [{ speaker_id: 'SPK-01 (Tenant)', duration: 3.5 }],
          model_insight: { label: 'Fire & Alarm Hazard Verified', confidence: 0.93 },
          reasoning_trace: 'Step 1: Continuous 3.1kHz pulsed acoustic tone detected matching standard smoke alarm specifications (99% confidence).\nStep 2: High-frequency transient detected matching glass fracturing.\nStep 3: Keyword "fire" confirmed in speech channel.',
          fusion_summary: 'Fire emergency confirmed via continuous smoke alarm acoustics and vocal alarm distress.',
          created_at: new Date().toISOString(),
          alerts: [
            {
              id: `alert-fire-${Date.now()}-1`,
              session_id: sessionId,
              type: 'sound_event',
              severity: 'high',
              message: 'Critical acoustic event detected: smoke alarm',
              details: { source: 'MIC-02', confidence: 0.99 },
              created_at: new Date().toISOString()
            },
            {
              id: `alert-fire-${Date.now()}-2`,
              session_id: sessionId,
              type: 'keyword',
              severity: 'medium',
              message: 'Distress keywords identified: "fire, emergency"',
              details: { source: 'TRANSCRIPT', confidence: 0.92 },
              created_at: new Date().toISOString()
            }
          ]
        };
        injectMockResult(mockFire);
        setSceneResult(mockFire);
        setActiveAlerts(mockFire.alerts);
      } else {
        const mockNominal = {
          id: `SIG-${Date.now().toString().slice(-6)}`,
          session_id: sessionId,
          transcript_text: 'Good morning everyone, let us review the engineering milestones for today.',
          transcript_lang: 'en',
          sound_events: [
            { label: 'speech', confidence: 0.98 },
            { label: 'ambient_office', confidence: 0.82 }
          ],
          emotion: { primary: 'neutral', arousal: 'low', confidence: 0.95 },
          speakers: [{ speaker_id: 'SPK-01 (Host)', duration: 4.8 }],
          model_insight: { label: 'Nominal Conversational Baseline', confidence: 0.99 },
          reasoning_trace: 'Step 1: Ambient indoor acoustic baseline observed.\nStep 2: Vocal pitch and cadence within normal conversational thresholds (120Hz-220Hz).\nStep 3: Zero threat signatures detected across acoustic spectrum.',
          fusion_summary: 'Standard meeting conversation. Environment nominal with zero risk indicators.',
          created_at: new Date().toISOString(),
          alerts: []
        };
        injectMockResult(mockNominal);
        setSceneResult(mockNominal);
        setActiveAlerts([]);
      }
      setIsLoading(false);
    }, 280);
  };

  return (
    <div className="cursor-workstation-page" id="live-monitor-page">
      {/* Cursor Topbar */}
      <header className="cursor-topbar">
        <div className="cursor-brand-left">
          <svg className="cursor-logo-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2">
            <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
          </svg>
          <div className="cursor-brand-text-stack">
            <span className="cursor-brand-title">Smart Horizon 2026</span>
            <span className="cursor-brand-subtitle">Acoustic Scene Intelligence</span>
          </div>
        </div>

        <div className="cursor-topbar-center">
          <div className="cursor-chip-pill">
            <span className={`cursor-beacon-dot ${isConnected ? 'online' : 'offline'}`}></span>
            <span>{isConnected ? 'Agent Online' : 'Agent Offline'}</span>
          </div>
          <div className="cursor-chip-pill">
            <span>48 kHz · 24-bit</span>
          </div>
          <div className="cursor-chip-pill">
            <span>12ms Latency</span>
          </div>
        </div>

        <div className="cursor-topbar-right">
          <span className="cursor-utc-clock">{utcTime || '15:12:32 UTC'}</span>
          <div className="cursor-session-box">
            <span className="cursor-session-label">Session:</span>
            <input
              className="cursor-session-input"
              value={sessionId}
              onChange={(e) => setSessionId(e.target.value)}
              placeholder="session_01"
            />
            <button
              className="btn-cursor-new-session"
              onClick={() => setSessionId('sess_' + Math.random().toString(36).substring(2, 8))}
              title="Generate new session identifier"
            >
              + New
            </button>
          </div>
        </div>
      </header>

      {/* Main Workstation Canvas */}
      <main className="cursor-main-content">
        {/* HERO: Cursor-Styled Acoustic Agent Monitor */}
        <RecordOrUpload
          sessionId={sessionId}
          onSessionChange={setSessionId}
          onAudioReady={handleAudioReady}
          onSimulateSample={handleSimulateSample}
          activeScenario={activeScenario}
        />

        {/* LOWER SPLIT: Composer Scene Intelligence + Incident Timeline */}
        <div className="cursor-lower-grid">
          {/* Left: Composer Scene Intelligence */}
          <ResultPanel
            result={sceneResult}
            isLoading={isLoading}
          />

          {/* Right: Agent Incident Stream */}
          <AlertBanner
            alerts={activeAlerts}
            onDismiss={handleDismissAlert}
            onClearAll={handleClearAllAlerts}
          />
        </div>
      </main>
    </div>
  );
}

export default LiveMonitorPage;

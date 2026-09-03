import React, { useState } from 'react';

export function ResultPanel({ result, isLoading = false }) {
  const [showRawJson, setShowRawJson] = useState(false);
  const [traceExpanded, setTraceExpanded] = useState(true);

  if (isLoading) {
    return (
      <section className="cursor-scene-panel" id="panel-scene-loading">
        <div className="cursor-panel-header">
          <div className="cursor-panel-title">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ color: 'var(--cursor-copper)' }}>
              <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
            </svg>
            <span>Composer · Scene Intelligence</span>
          </div>
        </div>
        <div className="cursor-standby-box" style={{ minHeight: '340px' }}>
          <div style={{ width: '24px', height: '24px', border: '2px solid rgba(255,255,255,0.1)', borderTopColor: 'var(--cursor-copper)', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }} />
          <span className="cursor-standby-headline">Fusing Multimodal Acoustics</span>
          <span className="cursor-standby-caption">
            Executing acoustic classification, speech diarization, and rule engine...
          </span>
        </div>
      </section>
    );
  }

  if (!result) {
    return (
      <section className="cursor-scene-panel" id="panel-scene-empty">
        <div className="cursor-panel-header">
          <div className="cursor-panel-title">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ color: 'var(--cursor-copper)' }}>
              <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
            </svg>
            <span>Composer · Scene Intelligence</span>
          </div>
          <span style={{ fontSize: '0.72rem', color: 'var(--text-cursor-muted)' }}>
            Standby
          </span>
        </div>
        <div className="cursor-standby-box" style={{ minHeight: '340px' }}>
          <svg className="cursor-standby-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6">
            <circle cx="12" cy="12" r="10" />
            <polygon points="12 8 8 12 12 16 16 12 12 8" />
          </svg>
          <span className="cursor-standby-headline">Composer Workspace Ready</span>
          <span className="cursor-standby-caption">
            Awaiting acoustic stream. Start live capture or select a verification mode above to generate scene intelligence forensics.
          </span>
        </div>
      </section>
    );
  }

  const transcriptText =
    result.transcript_text ||
    (typeof result.transcript === 'string' ? result.transcript : result.transcript?.text) ||
    'No speech detected in acoustic stream.';
  const transcriptLang = (result.transcript_lang || result.transcript?.lang || 'en').toUpperCase();

  const soundEvents = Array.isArray(result.sound_events)
    ? result.sound_events
    : Array.isArray(result.soundEvents)
    ? result.soundEvents
    : [];

  const emotion = result.emotion || { primary: 'neutral', arousal: 'low', confidence: 0 };
  const speakers = Array.isArray(result.speakers) ? result.speakers : [];
  const modelInsight = result.model_insight || result.modelInsight || null;
  const reasoningTrace = result.reasoning_trace || result.reasoningTrace || '';
  const fusionSummary = result.fusion_summary || result.fusionSummary || '';

  const isCriticalSound = (label) => {
    const critical = ['gunshot', 'explosion', 'smoke_alarm', 'glass_breaking', 'scream'];
    return critical.some((c) => (label || '').toLowerCase().includes(c));
  };

  return (
    <section className="cursor-scene-panel" id="panel-scene-intelligence">
      <div className="cursor-panel-header">
        <div className="cursor-panel-title">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ color: 'var(--cursor-copper)' }}>
            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
          </svg>
          <span>Composer · Scene Intelligence</span>
        </div>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.72rem', color: 'var(--cursor-copper)' }}>
          {result.id || 'Live Telemetry'}
        </span>
      </div>

      <div className="cursor-panel-body">
        {/* Composer Plan Banner */}
        {fusionSummary && (
          <div className="cursor-composer-plan-banner">
            <div className="cursor-plan-lead-row">
              <span className="cursor-plan-badge">Review Plan: Multimodal Fusion Synthesis</span>
              <span className="cursor-plan-timestamp">
                {result.created_at ? new Date(result.created_at).toLocaleTimeString() : 'Recent'}
              </span>
            </div>
            <p className="cursor-plan-copy">{fusionSummary}</p>
          </div>
        )}

        {/* Speech Transcript & Diarization */}
        <div className="cursor-transcript-container">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span className="cursor-section-label">Speech Transcript & Diarization</span>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.68rem', color: 'var(--text-cursor-muted)' }}>
              Dialect: {transcriptLang}-US
            </span>
          </div>

          <div className="cursor-transcript-box">"{transcriptText}"</div>

          {speakers.length > 0 && (
            <div className="cursor-speaker-chips">
              {speakers.map((spk, idx) => (
                <div key={idx} className="cursor-speaker-pill">
                  <strong>{spk.speaker_id || `SPK-${idx + 1}`}</strong>
                  {spk.duration && <span>[{spk.duration}s]</span>}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Acoustic Classification Matrix Table */}
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
            <span className="cursor-section-label">Acoustic Classification ({soundEvents.length})</span>
          </div>

          {soundEvents.length === 0 ? (
            <div style={{ padding: '10px 14px', background: 'var(--bg-cursor-elevated)', border: '1px solid var(--border-cursor)', borderRadius: 'var(--radius-sm)', fontSize: '0.8rem', color: 'var(--text-cursor-muted)' }}>
              No acoustic anomalies identified above detection floor.
            </div>
          ) : (
            <div className="cursor-classification-wrapper">
              <table className="cursor-classification-table">
                <thead>
                  <tr>
                    <th>Acoustic Label</th>
                    <th>Classification Confidence</th>
                    <th>Severity Flag</th>
                  </tr>
                </thead>
                <tbody>
                  {soundEvents.map((evt, idx) => {
                    const label = typeof evt === 'string' ? evt : evt.label || evt.name || 'sound';
                    const confidence =
                      typeof evt.confidence === 'number'
                        ? evt.confidence
                        : typeof evt.score === 'number'
                        ? evt.score
                        : 1.0;
                    const critical = isCriticalSound(label);

                    return (
                      <tr key={idx}>
                        <td style={{ fontWeight: 600, fontFamily: 'var(--font-mono)' }}>
                          {label.replace(/_/g, ' ')}
                        </td>
                        <td>
                          <div className="cursor-confidence-bar-unit">
                            <div className="cursor-confidence-track-bg">
                              <div
                                className={`cursor-confidence-track-fill ${critical ? 'critical' : ''}`}
                                style={{ width: `${Math.min(100, Math.round(confidence * 100))}%` }}
                              />
                            </div>
                            <span className="cursor-confidence-digits">{Math.round(confidence * 100)}%</span>
                          </div>
                        </td>
                        <td>
                          <span className={`cursor-diff-tag ${critical ? 'critical' : 'nominal'}`}>
                            {critical ? 'CRITICAL' : 'NOMINAL'}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Two-Column Telemetry: Vocal Dynamics & Classifier Insight */}
        <div className="cursor-telemetry-subgrid">
          {/* Vocal & Emotion Dynamics */}
          <div className="cursor-vocal-card">
            <span className="cursor-section-label">Vocal & Emotion Dynamics</span>
            <div className="cursor-vocal-grid">
              <div className="cursor-vocal-box">
                <span className="cursor-vocal-caption">Emotion</span>
                <span
                  className="cursor-vocal-val"
                  style={{
                    color:
                      emotion.primary === 'fear' || emotion.primary === 'anger'
                        ? 'var(--cursor-red)'
                        : 'var(--text-cursor-primary)'
                  }}
                >
                  {(emotion.primary || 'Neutral').toUpperCase()}
                </span>
              </div>

              <div className="cursor-vocal-box">
                <span className="cursor-vocal-caption">Arousal</span>
                <span
                  className="cursor-vocal-val"
                  style={{
                    color: emotion.arousal === 'high' ? 'var(--cursor-red)' : 'var(--text-cursor-primary)'
                  }}
                >
                  {(emotion.arousal || 'Low').toUpperCase()}
                </span>
              </div>

              <div className="cursor-vocal-box">
                <span className="cursor-vocal-caption">Confidence</span>
                <span className="cursor-vocal-val">
                  {Math.round((emotion.confidence || 0) * 100)}%
                </span>
              </div>
            </div>
          </div>

          {/* Jay's Custom Classifier Card */}
          <div className="cursor-classifier-card">
            <div className="cursor-classifier-header">
              <span className="cursor-section-label">Custom Classifier</span>
              <span className="cursor-classifier-tag">Jay's Classifier</span>
            </div>

            {modelInsight ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                <span className="cursor-classifier-title">{modelInsight.label || 'Verified Safe'}</span>
                {typeof modelInsight.confidence === 'number' && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <div className="cursor-confidence-track-bg" style={{ height: '5px' }}>
                      <div
                        className="cursor-confidence-track-fill"
                        style={{ width: `${Math.round(modelInsight.confidence * 100)}%` }}
                      />
                    </div>
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.72rem', color: 'var(--text-cursor-muted)' }}>
                      {Math.round(modelInsight.confidence * 100)}% Match
                    </span>
                  </div>
                )}
              </div>
            ) : (
              <span style={{ fontSize: '0.78rem', color: 'var(--text-cursor-muted)' }}>
                Model telemetry standby.
              </span>
            )}
          </div>
        </div>

        {/* Cursor Agent Reasoning Chain */}
        {reasoningTrace && (
          <div className="cursor-reasoning-card">
            <div
              className="cursor-reasoning-header"
              onClick={() => setTraceExpanded((prev) => !prev)}
            >
              <span className="cursor-section-label">Agent Reasoning Chain</span>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.68rem', color: 'var(--cursor-copper)' }}>
                {traceExpanded ? 'COLLAPSE' : 'EXPAND'}
              </span>
            </div>

            {traceExpanded && (
              <div className="cursor-reasoning-steps">
                {reasoningTrace.split(/(?=Step \d+:|\n- |\n\d+\.)/).map((step, idx) => (
                  <div key={idx} className="cursor-step-row">
                    <span className="cursor-step-badge">0{idx + 1}</span>
                    <span className="cursor-step-text">{step.trim()}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Raw Telemetry Inspector */}
        <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '4px' }}>
          <button
            className="btn-cursor-new-session"
            onClick={() => setShowRawJson((prev) => !prev)}
          >
            {showRawJson ? 'Hide Raw Telemetry' : 'Inspect Raw JSON Payload'}
          </button>
        </div>

        {showRawJson && (
          <div style={{ background: '#090805', padding: '14px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-cursor)', overflowX: 'auto', fontFamily: 'var(--font-mono)', fontSize: '0.74rem', color: '#f59e0b', maxHeight: '360px' }}>
            <pre>{JSON.stringify(result, null, 2)}</pre>
          </div>
        )}
      </div>
    </section>
  );
}

export default ResultPanel;

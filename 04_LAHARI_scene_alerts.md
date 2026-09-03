# Lahari — Backend + DB: Scene Intelligence + Alerts, and Live Monitor Page

## Role
Own persistence of every ML result, the alert rule engine (including severity logic), WebSocket broadcasting, and the Live Monitor frontend page where all of this surfaces live.

## What you own
- `backend/src/modules/scene/` — persist ML results
- `backend/src/modules/alerts/` — rule engine, severity logic
- `backend/src/db/migrations/002_scene_alerts.sql`
- `backend/src/websocket/` — broadcast to connected clients
- `frontend/src/pages/LiveMonitor/` — the real-time result view

## Your DB tables

```sql
CREATE TABLE scene_results (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id UUID REFERENCES audio_sessions(id),
  transcript_text TEXT,
  transcript_lang TEXT,
  sound_events JSONB,
  emotion JSONB,
  speakers JSONB,
  model_insight JSONB,      -- Jay's trained model output: { label, confidence }
  reasoning_trace TEXT,
  fusion_summary TEXT,
  raw_ml_response JSONB,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE alerts (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id UUID REFERENCES audio_sessions(id),
  type TEXT NOT NULL,          -- 'sound_event', 'emotion', 'keyword'
  severity TEXT CHECK (severity IN ('low','medium','high')),
  message TEXT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT now()
);
```

## Your contract — endpoints you build

```
GET /api/v1/sessions/:id/result → 200 full ML result (as stored) + alerts: [Alert]

GET /api/v1/alerts?session_id= (optional filter)
  → 200 { alerts: [{ alert_id, session_id, type, severity, message, created_at }] }

WS /ws/sessions/:id
  → server pushes { event: "result_ready" | "alert_raised", payload } as they occur
```

## Alert rule engine — trigger logic (this is business logic, lives here, not in ML)

Trigger an alert when:
- `sound_events` contains any of: `gunshot, explosion, smoke_alarm, glass_breaking, scream`
- OR `emotion.primary in ["fear","anger"]` AND `emotion.arousal == "high"` AND `emotion.confidence > 0.7`
- OR `transcript.text` contains distress keywords (config list: "help", "emergency", "fire", "police", ...)

Severity: `high` if sound-event match, `medium` if emotion/keyword only.

If Jay's `model_insight.label` is used to adjust severity (e.g. downgrade/confirm based on his classifier), coordinate the exact rule with Jay once you know his label vocabulary — don't hardcode assumptions about his output.

## What you DO NOT touch
- `backend/src/modules/auth/`, `backend/src/modules/sessions/` (Anwin's)
- `backend/src/modules/analytics/` (Anoop's)
- ML service internals

## File structure

```
backend/src/modules/scene/
├── routes.js
├── controller.js
├── persistScene.js       # writes ML result to scene_results
└── scene.test.js

backend/src/modules/alerts/
├── routes.js
├── ruleEngine.js          # evaluate(sceneResult) -> Alert[] | []
└── alerts.test.js

backend/src/websocket/
├── server.js
└── broadcast.js           # emitResultReady(), emitAlertRaised()

frontend/src/pages/LiveMonitor/
├── LiveMonitorPage.jsx
├── RecordOrUpload.jsx     # MediaRecorder API + file upload
├── ResultPanel.jsx         # transcript, sound tags, emotion badge, reasoning trace
├── AlertBanner.jsx
└── useWebSocket.js
```

## Local setup

```bash
cd backend
npm install
npx sequelize-cli db:migrate    # runs 002_scene_alerts.sql, needs 001 already applied
npm run dev

cd ../frontend
npm install
npm run dev
```

Test the rule engine standalone:
```bash
node -e "const { evaluate } = require('./src/modules/alerts/ruleEngine'); \
console.log(evaluate({ sound_events: [{label:'smoke_alarm'}], emotion:{primary:'fear',arousal:'high',confidence:0.9}, transcript:{text:'help fire'} }))"
```

## Dependencies on others
- **Waits on:** nothing to start — build against mocked ML result JSON (exact shape in `docs/SRS.md`) from hour 2.
- **Anwin hands you:** the async ML result once his session endpoint gets it back — agree on the internal hook (event emitter, queue, or direct function call) at hour 0–2.
- **Jay's `model_insight` label vocabulary:** needed before you finalize any severity logic that depends on it — confirm by hour 16.

## Timeline checkpoints
| Hour | You should have |
|---|---|
| 0–2 | DB schema migrated (after Anwin's 001), contract confirmed |
| 2–10 | Results/alerts endpoints working against mocked data, rule engine unit-tested |
| 10–16 | Live Monitor UI built against fixtures |
| 16–22 | Real ML results flowing in from Anwin, WebSocket broadcasting live |
| 22–24 | Full integration — real audio → real alert banner firing on Live Monitor |
| 30–38 | Calibrate thresholds, polish reasoning trace + alert banner UI, error states |

# Anoop — Backend + Frontend: History, Session Detail, Analytics

## Role
Own read/aggregate access across sessions — no new DB tables, you query what Anwin and Lahari have already written. Own the three "look back at past sessions" pages.

## What you own
- `backend/src/modules/analytics/` — list, detail, and summary endpoints (read-only queries against `audio_sessions`, `scene_results`, `alerts`)
- `frontend/src/pages/History/`
- `frontend/src/pages/SessionDetail/`
- `frontend/src/pages/Analytics/`

## What you DO NOT touch
- No new tables — you read from `audio_sessions` (Anwin's), `scene_results` and `alerts` (Lahari's). Never write to them.
- `backend/src/modules/auth/`, `sessions/`, `scene/`, `alerts/` — read-only queries into their tables via your own module, not by editing their route files.

## Your contract — endpoints you build

```
GET /api/v1/sessions?page=1&limit=10
  → 200 { sessions: [{ session_id, created_at, status, has_alert, summary }], total }

GET /api/v1/sessions/:id/result
  → 200 full ML result (as stored) + alerts: [Alert]
  (NOTE: this route is shared conceptually with Lahari's contract — confirm at hour 2
   whether you or Lahari owns the actual route file; recommend Lahari owns the write-path
   version, you own a read-only aggregation view if they diverge)

GET /api/v1/analytics/summary
  → 200 {
      total_sessions,
      total_alerts,
      alerts_by_type: {...},
      top_sound_events: [{ label, count }]
    }
```

## Frontend pages you build

- **History/Dashboard** — `GET /sessions?page=` paginated table/cards, status badges, "has alert" flag, click-through to detail
- **Session Detail** — `GET /sessions/:id/result` — full transcript, sound-event timeline (visual), emotion, reasoning trace, alerts list
- **Analytics** — `GET /analytics/summary` — Recharts: alerts-by-type bar chart, top sound events bar chart, sessions-over-time line chart

## File structure

```
backend/src/modules/analytics/
├── routes.js
├── controller.js
├── queries.js            # SQL/ORM read queries against sessions/scene_results/alerts
└── analytics.test.js

frontend/src/pages/History/
├── HistoryPage.jsx
└── SessionCard.jsx

frontend/src/pages/SessionDetail/
├── SessionDetailPage.jsx
├── SoundEventTimeline.jsx
└── AlertsList.jsx

frontend/src/pages/Analytics/
├── AnalyticsPage.jsx
├── AlertsByTypeChart.jsx
├── TopSoundEventsChart.jsx
└── SessionsOverTimeChart.jsx
```

## Local setup

```bash
cd backend
npm install
npm run dev        # your routes need Anwin's + Lahari's migrations already applied

cd ../frontend
npm install
npm run dev
```

Test against seed data:
```bash
curl http://localhost:4000/api/v1/sessions?page=1&limit=10
curl http://localhost:4000/api/v1/analytics/summary
```

## Dependencies on others
- **Waits on:** nothing to start UI-wise — build all 3 pages against mocked fixtures (exact JSON shapes in `docs/SRS.md`) from hour 2.
- **Needs from Anwin:** `audio_sessions` table populated (his migration) before your queries return real data.
- **Needs from Lahari:** `scene_results` + `alerts` tables populated before Session Detail / Analytics show real content.
- Since you depend on both their tables, you're naturally the last piece to go fully "real" — plan to swap fixtures for live queries around hour 22–24, and don't block earlier work on it.

## Timeline checkpoints
| Hour | You should have |
|---|---|
| 0–2 | Contract confirmed, especially the `/sessions/:id/result` ownership question with Lahari |
| 2–10 | All 3 pages built against mocked fixtures |
| 10–22 | Analytics/history endpoints written against real schema (tables may still be empty, that's fine) |
| 22–24 | Swap fixtures for live queries once Anwin/Lahari's tables have real data flowing |
| 30–38 | Seed 4-5 realistic demo sessions, verify charts render non-empty, polish empty/loading states |

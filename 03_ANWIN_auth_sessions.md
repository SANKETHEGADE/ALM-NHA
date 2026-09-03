# Anwin — Backend + DB: Auth + Session Lifecycle, and Login Page

## Role
Own the full vertical for authentication and session creation — DB tables, backend endpoints, and the frontend login page. You're the only one who calls the ML service.

## What you own
- `backend/src/modules/auth/` — register, login, me
- `backend/src/modules/sessions/` — create session, upload audio, forward to ML
- `backend/src/db/migrations/001_users_sessions.sql`
- `frontend/src/pages/Login/` — Login/Register page

## Your DB tables

```sql
CREATE TABLE users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,
  name TEXT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE audio_sessions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES users(id),
  source_type TEXT CHECK (source_type IN ('upload','mic')),
  status TEXT CHECK (status IN ('pending','processing','done','failed')) DEFAULT 'pending',
  created_at TIMESTAMPTZ DEFAULT now()
);
```

## Your contract — endpoints you build

```
POST /api/v1/auth/register  → { email, password, name } → 201 { user_id, token }
POST /api/v1/auth/login     → { email, password }        → 200 { user_id, token, name }
GET  /api/v1/auth/me        → header Authorization: Bearer <token> → 200 { user_id, email, name }

POST /api/v1/sessions
  → { source_type: "upload"|"mic" }
  → 201 { session_id, status: "pending", created_at }

POST /api/v1/sessions/:id/audio
  → multipart/form-data { audio_file }
  → internally: forward audio_file + session_id to ML POST /analyze
  → respond 202 { session_id, status: "processing" } IMMEDIATELY, do not block on ML response
  → when ML responds (async), hand result off to Lahari's persistence layer (she listens for this)

GET  /api/v1/sessions/:id → 200 { session_id, status, created_at, result | null }
```

**Critical NFR:** do not block the HTTP thread on the ML call beyond ~12s. Respond `202 processing` right away, let the ML call complete in the background, then trigger Lahari's WebSocket push when it lands.

## What you DO NOT touch
- `backend/src/modules/scene/`, `backend/src/modules/alerts/` (Lahari's)
- `backend/src/modules/analytics/` (Anoop's)
- ML service internals — you only ever call `POST /analyze` as an HTTP client, never look inside `ml-service/`

## File structure

```
backend/src/modules/auth/
├── routes.js
├── controller.js
├── service.js          # password hashing, JWT issuance
└── auth.test.js

backend/src/modules/sessions/
├── routes.js
├── controller.js
├── mlClient.js           # calls ML service /analyze
└── sessions.test.js

frontend/src/pages/Login/
├── LoginPage.jsx
├── RegisterForm.jsx
└── api.js               # calls your auth endpoints
```

## Local setup

```bash
cd backend
npm install
cp .env.example .env        # set DB_URL, JWT_SECRET, ML_SERVICE_URL
npx sequelize-cli db:migrate   # or your ORM's equivalent, runs 001_users_sessions.sql
npm run dev                  # http://localhost:4000

cd ../frontend
npm install
npm run dev                  # http://localhost:5173
```

Test:
```bash
curl -X POST http://localhost:4000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@dot.exe","password":"pass123","name":"Test User"}'
```

## Dependencies on others
- **Waits on:** nothing to start — build against Sanket's mocked `/analyze` response (exact shape in `docs/SRS.md`) from hour 2.
- **You hand off to Lahari:** when ML result comes back, you need an agreed internal hook/event so she can persist it (e.g. emit an internal event, or she polls a queue you write to — decide together at hour 2).
- **Sanket needs from you:** the exact base URL and multipart field names you'll send to `/analyze` — confirm at hour 0–2, this is part of the frozen contract.

## Timeline checkpoints
| Hour | You should have |
|---|---|
| 0–2 | DB schema migrated, contract confirmed with Sanket + Lahari |
| 2–10 | Auth + session endpoints working against mocked ML response |
| 10–16 | Real ML calls wired in, 202-then-async-result pattern working |
| 22–24 | Full integration — real audio upload → real ML → your endpoint → Lahari picks it up |
| 30–38 | Error handling: bad audio, ML timeout, ML 500 retry-once-then-fail logic |

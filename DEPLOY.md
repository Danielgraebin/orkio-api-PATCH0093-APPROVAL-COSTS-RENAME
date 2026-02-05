# Orkio – Railway Deploy Checklist (Final)

## Backend (FastAPI)

### Required variables
- `DATABASE_URL` (from Railway Postgres)
- `JWT_SECRET` (generate a long random string)
- `ADMIN_EMAILS` (comma-separated, e.g. `daniel@patroai.com`)
- `CORS_ORIGINS` (comma-separated, e.g. `https://<your-web>.up.railway.app,http://localhost:5173`)

### Optional variables
- `ADMIN_API_KEY` (optional; enables admin access via `X-Admin-Key`)
- `APP_ENV` (e.g. `production`)
- `JWT_EXPIRES_IN` (seconds, default 3600)

### Railway commands
- **Pre-deploy command**: (optional) `alembic upgrade head`
  - Not required for MVP: backend auto-creates tables on startup when DB is configured.
- **Start command**: leave empty if using Dockerfile CMD, or set:
  - `uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}`

### Smoke tests
- `GET /docs`
- `GET /api/health`  (should return `db_ok: true` when DB connected)
- `POST /api/auth/register`
- `POST /api/auth/login`

## Frontend (Vite/React)

### Required variables
- `VITE_API_BASE_URL` = `https://<your-api>.up.railway.app`
  - **Do not** include `/api` (the app already calls `/api/...`)

### Notes
- If you already set `VITE_API_BASE_URL` with `/api`, this frontend will normalize it.

### Smoke tests
- Open `/` landing
- Go to `/auth`, register, then login
- If your email is in `ADMIN_EMAILS`, `/admin` should open.

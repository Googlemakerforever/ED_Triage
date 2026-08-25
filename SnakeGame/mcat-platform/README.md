# AegisMCAT (Production-Minded Full-Stack MCAT Platform)

A launch-quality MCAT study platform scaffold built with Next.js App Router, TypeScript, Prisma, Zod, and secure backend AI orchestration.

## Stack
- Frontend: Next.js, TypeScript, Tailwind CSS, reusable UI components
- Backend: Next.js route handlers, service/repository layers, Zod validation
- Data: Prisma ORM, PostgreSQL-compatible schema
- Auth: Secure HTTP-only cookie session (JWT)
- AI: Server-only provider abstraction with Claude-ready implementation

## Project Structure
- `app/` routes (public + authenticated + API)
- `components/` reusable UI, layout, and state components
- `features/` page-level feature modules
- `lib/contracts/` typed schema contracts
- `lib/services/` business logic
- `lib/repositories/` data access
- `lib/ai/` provider abstraction + prompt builder + Claude provider
- `lib/auth/` session and guards
- `prisma/` schema + seed script

## Setup
1. `cd mcat-platform`
2. `cp .env.example .env`
3. Fill `.env` values (`DATABASE_URL`, `JWT_SECRET`, `APP_ORIGIN`, optional Claude vars)
4. Install deps: `npm install`
5. Generate Prisma client: `npm run prisma:generate`
6. Run migrations: `npm run prisma:migrate`
7. Seed data: `npm run prisma:seed`
8. Start app: `npm run dev`

Demo login after seed:
- Email: `demo@aegismcat.com`
- Password: `Str0ngPassw0rd!`

## Backend Route Map
Auth:
- `POST /api/auth/signup`
- `POST /api/auth/login`
- `POST /api/auth/logout`
- `GET /api/auth/me`

Profile / Onboarding:
- `GET /api/profile`
- `PATCH /api/profile`
- `POST /api/onboarding`

Dashboard / Study:
- `GET /api/dashboard`
- `GET /api/study-plan`
- `POST /api/study-plan/generate`
- `PATCH /api/study-tasks/:id`

Practice / Review:
- `GET /api/practice/questions`
- `GET /api/practice/passages`
- `POST /api/practice/sessions`
- `POST /api/practice/sessions/:id/submit`
- `GET /api/practice/sessions/:id/review`
- `GET /api/review`

Error Log / SRS:
- `GET /api/error-log`
- `POST /api/error-log`
- `PATCH /api/error-log/:id`
- `GET /api/spaced-repetition/queue`
- `POST /api/spaced-repetition/review`

Analytics:
- `GET /api/analytics/overview`
- `GET /api/analytics/trends`
- `GET /api/analytics/mastery`

Content:
- `GET /api/content/topics`
- `GET /api/content/topics/:slug`

AI Tutor:
- `GET /api/ai/conversations`
- `POST /api/ai/conversations`
- `POST /api/ai/conversations/:id/messages`

## Secure Claude Architecture
Client flow:
1. Frontend sends validated request to backend `/api/ai/*`
2. Backend requires authenticated session
3. Backend fetches profile + weak-topic context from DB
4. Server builds mode-specific prompt (`lib/ai/prompt-builder.ts`)
5. Server calls provider (`lib/ai/claude-provider.ts`) with env key
6. Response is truncated/sanitized and persisted
7. Safe payload returned to frontend

Security controls in place:
- No frontend Claude calls
- No key exposure to client bundles
- No unrestricted prompt passthrough
- Input validation with Zod
- Server-owned system instructions
- Ownership checks at route/repository layers
- Trusted-origin checks on all mutating API routes
- Per-scope rate limiting hooks (auth, AI, write endpoints)
- Revocable DB-backed session validation (JWT + server session record)
- Prompt-injection pattern blocking + message sanitization
- Global secure response headers via `middleware.ts`

## Environment Variables
- `DATABASE_URL`: PostgreSQL connection string
- `JWT_SECRET`: long random secret (min 32 chars)
- `APP_ORIGIN`: app URL (e.g. `http://localhost:3000`)
- `CLAUDE_API_KEY`: Anthropic API key (server only)
- `CLAUDE_MODEL`: model id (default `claude-3-7-sonnet-latest`)

## AI Live/Mock Switching
- If `CLAUDE_API_KEY` is empty: backend returns a safe offline tutor fallback.
- If set: backend uses live Claude via `@anthropic-ai/sdk`.
- Provider swap: implement `AiProvider` in `lib/ai/provider.ts` and replace wiring in `ai-tutor-service.ts`.

## Production Hardening Recommendations
- Add DB-backed session store + token rotation + revocation
- Add CSRF tokens for sensitive write routes
- Add rate limiting per user/IP on auth + AI routes
- Add audit logs for auth, AI usage, and settings changes
- Add background jobs for study-plan generation and analytics rollups
- Add observability (traces/metrics) and structured logs
- Add email provider for real forgot/reset password
- Add S3/object storage for richer content assets

## Security Audit Snapshot
- Fixed: stateless-only auth boundary. Sessions now require both valid JWT and matching server-side session record.
- Fixed: missing mutation origin checks. All `POST`/`PATCH` handlers now enforce trusted-origin policy.
- Fixed: unaudited high-risk endpoints. Auth and AI write paths now have scoped rate limits.
- Fixed: AI mode override risk. Message endpoint now uses persisted conversation mode; client cannot override it.
- Fixed: prompt-injection passthrough risk. AI input is sanitized and obvious jailbreak attempts are blocked before model calls.
- Fixed: unvalidated query params. Practice and analytics query boundaries are now schema-validated.

## Deployment Notes
- Deploy as standard Next.js app (Vercel or containerized Node runtime)
- Use managed PostgreSQL (Neon, Supabase, RDS, etc.)
- Run Prisma migrations in CI/CD release step
- Set all secrets in deployment platform secret manager
- Keep `CLAUDE_API_KEY` server-only and never in public runtime config

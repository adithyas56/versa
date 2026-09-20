# Versa — Deploy to Google Cloud (Cloud Run + Cloud SQL)

A minute, copy-paste guide. Versa is a Starlette app (`probe serve`) that
needs **PostgreSQL + pgvector** (→ Cloud SQL) and three secrets (Gemini,
Moss, and the DB URL). We deploy the container to **Cloud Run** (HTTPS, which
the browser mic needs) and the database to **Cloud SQL**.

> You run every command below on **your machine** in the repo folder. Replace
> the ALL-CAPS placeholders once at the top (Step 0) and the rest just works.
> Lines starting `#` are comments — don't type them.

---

## What you need first (one time)

1. A **Google Cloud account** with **billing enabled** (Cloud Run + Cloud SQL
   need billing; a small demo costs a few dollars — see Cost & cleanup).
2. **Google Cloud CLI** installed → https://cloud.google.com/sdk/docs/install
   Verify: `gcloud version`
3. **Docker Desktop running** (Cloud Build does the build in the cloud, so
   Docker isn't strictly required, but keep it running).
4. Your real keys ready: `GEMINI_API_KEY`, `MOSS_PROJECT_ID`,
   `MOSS_PROJECT_KEY` (they're in your local `.env`).

---

## Step 0 — Set your variables (do this once per terminal)

**PowerShell** (Windows):

```powershell
$PROJECT_ID   = "versa-demo-<yourname>"     # must be globally unique
$REGION       = "us-central1"
$SQL_INSTANCE = "versa-db"
$DB_NAME      = "versa"
$DB_USER      = "versa"
$DB_PASSWORD  = "<pick-a-strong-password-no-@-or-slash>"
$SERVICE      = "versa"
```

(If you use Git Bash instead, use `PROJECT_ID=... ; export PROJECT_ID` etc.)

---

## Step 1 — Create the project & turn on the APIs

```powershell
gcloud auth login
gcloud projects create $PROJECT_ID
gcloud config set project $PROJECT_ID
```

Now **link billing**: open https://console.cloud.google.com/billing , pick
your project, attach a billing account (or run
`gcloud billing projects link $PROJECT_ID --billing-account=YOUR_BILLING_ID`).

Enable the services we use:

```powershell
gcloud services enable run.googleapis.com sqladmin.googleapis.com `
  secretmanager.googleapis.com cloudbuild.googleapis.com `
  artifactregistry.googleapis.com
```

---

## Step 2 — Create the Cloud SQL (PostgreSQL) database

```powershell
# Create the instance (Postgres 16). This takes ~5-10 minutes.
gcloud sql instances create $SQL_INSTANCE `
  --database-version=POSTGRES_16 `
  --tier=db-f1-micro `
  --region=$REGION `
  --storage-size=10GB

# Create the database and the app user
gcloud sql databases create $DB_NAME --instance=$SQL_INSTANCE
gcloud sql users create $DB_USER --instance=$SQL_INSTANCE --password=$DB_PASSWORD

# Get the instance connection name — SAVE THIS, you'll reuse it
gcloud sql instances describe $SQL_INSTANCE --format="value(connectionName)"
```

That last command prints something like `versa-demo-x:us-central1:versa-db`.
Save it:

```powershell
$CONNZ = "PASTE_THE_CONNECTION_NAME_HERE"    # e.g. versa-demo-x:us-central1:versa-db
```

> **pgvector:** Versa's migrations run `CREATE EXTENSION IF NOT EXISTS vector`.
> Cloud SQL for PostgreSQL supports pgvector, and the default `versa` user can
> create it — no extra step needed; the migrate command in Step 4 handles it.

---

## Step 3 — Store the secrets in Secret Manager

The app reads env vars; we keep the sensitive ones in Secret Manager and let
Cloud Run inject them. First build the **DATABASE_URL** for Cloud SQL (it uses
a unix socket, not host/port):

```powershell
$DATABASE_URL = "postgresql://$DB_USER`:$DB_PASSWORD@/$DB_NAME?host=/cloudsql/$CONNZ"
```

Create the three secrets (paste each value when the editor/echo runs):

```powershell
# Gemini
"YOUR_REAL_GEMINI_API_KEY" | gcloud secrets create GEMINI_API_KEY --data-file=-
# Moss
"YOUR_REAL_MOSS_PROJECT_ID"  | gcloud secrets create MOSS_PROJECT_ID  --data-file=-
"YOUR_REAL_MOSS_PROJECT_KEY" | gcloud secrets create MOSS_PROJECT_KEY --data-file=-
# Database URL
$DATABASE_URL | gcloud secrets create DATABASE_URL --data-file=-
```

(Optional ElevenLabs — **not required**, Versa uses the free browser voice:)

```powershell
# "YOUR_ELEVENLABS_KEY" | gcloud secrets create ELEVENLABS_API_KEY --data-file=-
```

---

## Step 4 — Apply the database schema (migrations)

The app does **not** migrate on startup, so we run it once against Cloud SQL
using the **Cloud SQL Auth Proxy** (a secure local tunnel).

1. Download the proxy: https://cloud.google.com/sql/docs/postgres/sql-proxy
   (get `cloud-sql-proxy.exe`), put it in the repo folder.
2. In **one** terminal, start the tunnel (leave it running):

   ```powershell
   ./cloud-sql-proxy.exe $CONNZ --port 5432
   ```
3. In a **second** terminal (repo folder), point Versa at the tunnel and
   migrate:

   ```powershell
   $env:DATABASE_URL = "postgresql://$DB_USER`:$DB_PASSWORD@127.0.0.1:5432/$DB_NAME"
   uv run probe migrate
   uv run probe migrate --status      # should show all applied, none pending
   ```

   You should see `probe migrate: applied N migration(s)`. Stop the proxy
   (Ctrl+C) when done.

> The real **Moss index** (`versa-learner-memory`) is already built in Moss
> Cloud, so nothing to do there — Cloud Run loads it at startup. If you want
> the two demo learners' memories in this new DB, also run (with the proxy
> still up and `DATABASE_URL` pointed at it):
> `uv run python scripts/seed_demo_learners.py` then
> `uv run python scripts/build_moss_index.py` (rebuilds the index once).

---

## Step 5 — Deploy to Cloud Run

One command builds the image (via Cloud Build, using your `Dockerfile`),
pushes it, and deploys — wiring in Cloud SQL and the secrets:

```powershell
gcloud run deploy $SERVICE `
  --source . `
  --region $REGION `
  --platform managed `
  --allow-unauthenticated `
  --add-cloudsql-instances $CONNZ `
  --set-secrets "DATABASE_URL=DATABASE_URL:latest,GEMINI_API_KEY=GEMINI_API_KEY:latest,MOSS_PROJECT_ID=MOSS_PROJECT_ID:latest,MOSS_PROJECT_KEY=MOSS_PROJECT_KEY:latest" `
  --set-env-vars "MOSS_INDEX_NAME=versa-learner-memory,VERSA_RETRIEVAL_ENGINE=moss,MOSS_REQUIRED=false" `
  --memory 2Gi `
  --cpu 2 `
  --cpu-boost `
  --timeout 300 `
  --min-instances 1 `
  --max-instances 3
```

- `--min-instances 1` keeps one instance warm so the demo has no cold-start
  wait (the Moss index stays loaded). Set it back to `0` after the demo to
  save money.
- First deploy takes ~5-8 minutes (it builds the container). When it finishes
  it prints your **Service URL**: `https://versa-xxxxx-uc.a.run.app`.

---

## Step 6 — Verify it works

1. Open the printed **Service URL** in **Chrome**. You should see the Versa
   page (`Probe a mind.`) with the mic orb.
2. Type a question → you get a real Gemini answer.
3. Turn **🔊 speak: on** and tap the **mic** (allow the microphone). Because
   Cloud Run serves **HTTPS**, the browser mic works (it needs a secure
   context — which the `run.app` URL is).
4. Check the logs if anything is off:

   ```powershell
   gcloud run services logs read $SERVICE --region $REGION --limit 50
   ```

That HTTPS Service URL is your **Deployed Link** for submission.

---

## Troubleshooting (most common)

| Symptom | Fix |
|---|---|
| Deploy fails building | Check `Dockerfile` is in repo root and `.dockerignore` exists (it does). Re-run the deploy command. |
| App starts then errors on DB | The `DATABASE_URL` secret must use the socket form `...@/DB?host=/cloudsql/CONNZ` and you must pass `--add-cloudsql-instances $CONNZ`. Re-check both. |
| `relation ... does not exist` | You skipped Step 4 — run the migrations against Cloud SQL. |
| Moss shows `stub-moss` in the UI | Cloud Run couldn't load the real index — check `MOSS_PROJECT_ID/KEY` secrets are correct and `MOSS_INDEX_NAME=versa-learner-memory`. The app still works (stub fallback). |
| Mic does nothing on the deployed site | Use Chrome, click **Allow** on the mic prompt, and make sure your OS default input device is a real mic (not "Stereo Mix"). |
| Slow first response | Cold start + Moss load. `--min-instances 1` avoids it during the demo. |

---

## Cost & cleanup

- `db-f1-micro` + one warm Cloud Run instance is roughly a few dollars for a
  demo period. **After the video**, scale down to avoid charges:

  ```powershell
  gcloud run services update $SERVICE --region $REGION --min-instances 0
  ```
- To remove everything when finished:

  ```powershell
  gcloud run services delete $SERVICE --region $REGION
  gcloud sql instances delete $SQL_INSTANCE
  ```

---

## Redeploying after a code change

Just re-run the **Step 5** command — it rebuilds and rolls out a new
revision. Secrets and the Cloud SQL wiring are remembered, but it's safe to
pass the flags again.

---

## Final stack (for your submission write-up)

```
Tutor / orchestration : Versa (probe)      → Cloud Run (container, HTTPS)
Durable memory        : PostgreSQL+pgvector → Cloud SQL
Semantic retrieval    : Moss                → loaded locally in the Cloud Run instance
Reasoning             : Gemini              → Google Gen AI API
Voice                 : browser Web Speech + speechSynthesis (free, in-page)
Secrets               : Secret Manager
```

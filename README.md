# MedReel — social videos for physicians

Doctors record themselves (or don't), and MedReel produces an agency-quality vertical video:
cuts, captions, cardiology animations, branding, music, in **Arabic (Lebanese), English or French**.
It's a subscription web app: accounts, monthly plans through Stripe, a dashboard, and a
render queue.

## What a user can do
| Mode | Input | What happens |
|---|---|---|
| Upload a recording | phone/webcam video | transcribed, pauses/retakes/fillers cut, captions, animations, branding |
| Recording + my script | video + the script they read | captions use the script's exact words; speech recognition only times them (best for dialect) |
| Write and record here | topic → AI draft (or starter script) → in-browser teleprompter recording | same as above |
| No filming | topic or script | AI narrator (generic voice, labelled "AI voice" on the end card), full-screen animations, kinetic captions |

Graphics (all original, procedural): CT scanner, cardiac CT calcium scoring with the CAC-DRS scale, beating heart with the
coronary tree, plaque/narrowing, plaque rupture and clot, balloon + stent, ECG, blood-pressure card, keyword pop-ups, and the
doctor's own photos (clinic, team…). For every specialty: animated icon cards (692 medical icons from Health Icons, MIT)
triggered by what the doctor says in Arabic/French/English (`engine/concepts.py`), and optional stock footage from Pixabay or Pexels
(`PIXABAY_API_KEY` or `PEXELS_API_KEY`, both free). With `FAL_KEY` (+ `ANTHROPIC_API_KEY` for the briefs): AI medical
illustrations (FLUX.2) in the doctor's style, and 5-second animations of them (Kling 2.6 Pro) — automatic for the key moments,
and on demand in the studio (Generate / Regenerate / Animate). — added automatically and searchable in the studio's "Photos & videos" tab. Hesitations ("ااا", "euh", "um") are cut automatically; a soft music bed sits ~15 dB under the voice.

After the first edit, the doctor reviews it in the **studio**: toggle cuts, fix the transcript, change layouts, add graphics or photos, ask for changes in
plain language ("make the subtitles more dynamic"), check the flagged claims, then export 1080×1920 MP4 + .srt.

## Test it on your own computer (Windows / Mac) — easiest
Install Docker Desktop, download this repository, open a terminal in its folder and run:
```bash
docker compose -f docker-compose.local.yml up --build
```
The first start downloads ~2 GB of models (15–30 min). Then open http://localhost:8080.
Test mode: choosing a plan activates it without payment. Stop with Ctrl+C; your videos are kept for next time.

## Run it locally without Docker (test mode — no Stripe needed)
```bash
pip install -r requirements.txt            # needs ffmpeg on PATH
bash tools/setup_models.sh                 # ~2.5 GB of models into ../models  (WITH_LARGE=1 for Whisper large-v3)
MEDREEL_DEV=1 python wsgi.py               # http://localhost:8080
```
`MEDREEL_DEV=1` makes "Choose plan" activate instantly without payment and shows password-reset links on screen.
The web process runs a render worker in a thread (`WORKER_INLINE=1`, default).

## Deploy (one server, Docker)
Recommended start: a cloud VM with **8 vCPU / 16 GB RAM / 160 GB disk** (Hetzner CCX33, DigitalOcean, OVH…).
Add a GPU worker later for speed (see Scaling).
1. Point a DNS record (e.g. `app.yourdomain.com`) at the server.
2. `git clone` this repo on the server, `cp .env.example .env`, fill it in.
3. `docker compose up -d --build` — starts PostgreSQL, the web app, a render worker and Caddy (automatic HTTPS).

### Stripe setup
1. In Stripe → Products, create **Starter**, **Pro**, **Clinic**, each with a monthly recurring price. Put the price ids
   (`price_…`) in `STRIPE_PRICE_STARTER/PRO/CLINIC`. The prices shown on the site are in `saas/config.py` — keep them equal.
2. Developers → Webhooks → add endpoint `https://app.yourdomain.com/stripe/webhook` with events
   `checkout.session.completed`, `customer.subscription.created`, `customer.subscription.updated`,
   `customer.subscription.deleted`, `invoice.payment_failed`. Put its signing secret in `STRIPE_WEBHOOK_SECRET`.
3. Settings → Billing → Customer portal: enable plan switching between the three prices, cancellation and invoice history.
4. Test with `sk_test_…` keys and card `4242 4242 4242 4242` before switching to live keys.

### Optional services
- `ANTHROPIC_API_KEY` — AI script drafts, automatic clean-up of Arabic transcripts, subtitle translation (EN/FR/AR).
  Without it, starter scripts and manual editing still work.
- `ELEVENLABS_API_KEY` — cloud transcription (ElevenLabs Scribe) for Arabic and French: far more accurate on Lebanese dialect
  than the offline models, and takes seconds. Falls back to the offline models if the key is missing or the service fails.
  The doctor's audio is sent to ElevenLabs — mention it in your privacy policy.
- `SMTP_*` — password-reset emails.

## Architecture
```
browser ── Flask web app (wsgi.py) ── PostgreSQL (users, projects, jobs, usage, stripe events)
                │  enqueue                         ▲
                ▼                                  │ claim / progress
           jobs table  ◄──── render workers (worker.py) ── engine/ (ASR, analysis, planner, renderer)
                                   │
                             STORAGE_DIR (/data/storage/users/<user>/<project>/…)
```
- `saas/` — accounts (`auth.py`), Stripe (`billing.py`), project API (`api.py`), queue (`jobs.py`), models, config.
- `engine/` — the video engine: `asr.py`, `script_align.py`, `synth.py` (AI narration), `scripts.py` (AI + starter scripts),
  `analyze.py`, `planner.py` (EditPlan JSON), `render.py`, `graphics.py` (procedural medical animations), `knowledge.py`
  (EN/FR/AR cardiology vocabulary, graphic triggers, claim checks), `commands.py` (plain-language edits).
- `web/app.html` — landing, sign-in, dashboard, new-video wizard (incl. teleprompter), settings, billing.
  `web/studio.html` — the review/editing studio.

Plans and credits: one **video** = one new project (all edits and re-exports of it are free). Credits reset each calendar
month (UTC) and are kept in a ledger, so deleting a video doesn't return its credit; a failed generation does.

## Security & privacy
Passwords hashed (scrypt), sessions signed and HTTP-only, CSRF token on every state-changing request, login throttling,
per-user ownership checks on every project and file, upload type/duration validation, original recordings read-only,
one-click deletion of a video or the whole account (also cancels the subscription). Before taking real patients'
data, review PHIPA/PIPEDA (Canada), GDPR (France) obligations and host in the right region; recordings should never show patients.

## Scaling
- Rendering a 60 s video takes ~2–4 min on 8 vCPU (Arabic with Whisper large-v3: much longer on CPU).
  Run more `worker.py` processes, or a GPU machine running only `python worker.py` against the same database and storage
  (mount the storage volume over NFS, or move `STORAGE_DIR` to a shared disk).
- Paid plans jump the queue (`priority`).

## Licences
Own code: yours. Dependencies: FFmpeg (LGPL/GPL with x264, used as a separate process), OpenCV (Apache-2.0), YuNet (MIT),
sherpa-onnx (Apache-2.0), Parakeet-TDT weights (CC-BY-4.0), Whisper (MIT), Piper voices: English LibriTTS-R (CC BY 4.0), French SIWIS (CC BY 4.0), Arabic Kareem (dataset licence not yet verified — confirm or replace before charging for AI-voice videos),
Flask (BSD), SQLAlchemy (MIT), Stripe SDK (MIT), Pillow (MIT-CMU), fonts Poppins/Inter/Tajawal (SIL OFL).
Medical animations and music are original procedural code.

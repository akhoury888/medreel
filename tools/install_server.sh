#!/usr/bin/env bash
# MedReel — one-time server installer (Ubuntu 22.04/24.04). Run as root from the project folder:
#   bash tools/install_server.sh
# Installs Docker, opens only the web + SSH ports, asks for your domain and keys, writes .env,
# starts MedReel with automatic HTTPS, and sets up a daily backup. Safe to run again (keeps existing .env values).
set -euo pipefail
cd "$(dirname "$0")/.."

say() { printf "\n\033[1;36m%s\033[0m\n" "$*"; }
ask() {  # ask VAR "Question" [default]  — keeps the current .env value as the default
  local var=$1 q=$2 def=${3:-} cur
  cur=$(grep -E "^$var=" .env 2>/dev/null | head -1 | cut -d= -f2- || true)
  [ -n "$cur" ] && def=$cur
  read -r -p "$q${def:+ [$def]}: " ans || true
  printf -v "$var" '%s' "${ans:-$def}"
}

say "1/6  Installing Docker (if needed)…"
if ! command -v docker >/dev/null 2>&1; then
  apt-get update -y && apt-get install -y ca-certificates curl
  curl -fsSL https://get.docker.com | sh
fi
apt-get install -y ufw >/dev/null 2>&1 || true

say "2/6  Firewall: only SSH (22) and the website (80, 443) are open"
ufw allow 22/tcp >/dev/null; ufw allow 80/tcp >/dev/null; ufw allow 443/tcp >/dev/null
ufw --force enable >/dev/null

say "3/6  Your settings (press Enter to keep a value shown in [brackets])"
ask DOMAIN "Website address without https (e.g. app.medreel.com)"
ask ANTHROPIC_API_KEY "Claude key (sk-ant-...)"
ask ELEVENLABS_API_KEY "ElevenLabs key"
ask FAL_KEY "fal.ai key"
ask PIXABAY_API_KEY "Pixabay key (optional, Enter to skip)"
ask STRIPE_SECRET_KEY "Stripe secret key (sk_live_... or sk_test_...; Enter to add later)"
ask STRIPE_WEBHOOK_SECRET "Stripe webhook secret (whsec_...; Enter to add later)"
ask STRIPE_PRICE_STARTER "Stripe price id for Starter (price_...; Enter to add later)"
ask STRIPE_PRICE_PRO "Stripe price id for Pro (price_...)"
ask STRIPE_PRICE_CLINIC "Stripe price id for Clinic (price_...)"
ask SMTP_HOST "Email server for password resets (e.g. smtp.postmarkapp.com; Enter to add later)"
ask SMTP_USER "Email username / API token"
ask SMTP_PASSWORD "Email password / API token"
ask MAIL_FROM "Sender address" "MedReel <no-reply@${DOMAIN#app.}>"

SECRET_KEY=$(grep -E "^SECRET_KEY=" .env 2>/dev/null | cut -d= -f2- || true)
case "$SECRET_KEY" in ""|change-me*) SECRET_KEY=$(openssl rand -base64 48 | tr -d '\n/+=' | cut -c1-60);; esac
POSTGRES_PASSWORD=$(grep -E "^POSTGRES_PASSWORD=" .env 2>/dev/null | cut -d= -f2- || true)
case "$POSTGRES_PASSWORD" in ""|change-me*) POSTGRES_PASSWORD=$(openssl rand -hex 24);; esac

say "4/6  Writing .env (kept private: only root can read it)"
cat > .env <<EOF
DOMAIN=$DOMAIN
PUBLIC_URL=https://$DOMAIN
SECRET_KEY=$SECRET_KEY
POSTGRES_PASSWORD=$POSTGRES_PASSWORD
STRIPE_SECRET_KEY=$STRIPE_SECRET_KEY
STRIPE_WEBHOOK_SECRET=$STRIPE_WEBHOOK_SECRET
STRIPE_PRICE_STARTER=$STRIPE_PRICE_STARTER
STRIPE_PRICE_PRO=$STRIPE_PRICE_PRO
STRIPE_PRICE_CLINIC=$STRIPE_PRICE_CLINIC
STRIPE_AUTOMATIC_TAX=0
PRICE_CURRENCY=USD
ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY
ELEVENLABS_API_KEY=$ELEVENLABS_API_KEY
FAL_KEY=$FAL_KEY
PIXABAY_API_KEY=$PIXABAY_API_KEY
MEDREEL_AI_ANIMATE=2
MEDREEL_GEN_ANIMS=2
MEDREEL_LLM=claude-sonnet-5-5
SMTP_HOST=$SMTP_HOST
SMTP_PORT=587
SMTP_USER=$SMTP_USER
SMTP_PASSWORD=$SMTP_PASSWORD
MAIL_FROM=$MAIL_FROM
MAX_UPLOAD_MB=1500
MAX_VIDEO_SECONDS=600
WITH_LARGE=0
MEDREEL_DEV=0
EOF
chmod 600 .env

say "5/6  Starting MedReel (first time: ~10-20 minutes to build and download the speech models)…"
docker compose up -d --build

say "6/6  Daily backup at 03:30 (database + videos, last 7 days kept in /root/medreel-backups)"
cat > /usr/local/bin/medreel-backup <<EOF
#!/usr/bin/env bash
set -e
D=/root/medreel-backups; mkdir -p \$D; S=\$(date +%F)
cd $(pwd)
docker compose exec -T db pg_dump -U medreel medreel | gzip > \$D/db-\$S.sql.gz
docker run --rm -v \$(basename \$(pwd))_storage:/data -v \$D:/out alpine tar czf /out/videos-\$S.tgz -C /data . 2>/dev/null || true
find \$D -type f -mtime +7 -delete
EOF
chmod +x /usr/local/bin/medreel-backup
( crontab -l 2>/dev/null | grep -v medreel-backup; echo "30 3 * * * /usr/local/bin/medreel-backup" ) | crontab -

say "Done. In a few minutes open:  https://$DOMAIN"
echo "(HTTPS works as soon as the domain points at this server's IP: $(curl -s -4 ifconfig.me || echo 'see your Hetzner page'))"
echo "Logs:   docker compose logs -f web worker"
echo "Update: git pull && docker compose up -d --build"

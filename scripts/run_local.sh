#!/bin/bash
# Minimal local launcher for the OpenAI Realtime + VRM avatar path.
# Unlike start.sh, this does NOT do AWS SSO / Secrets Manager — it just sources
# .env for OPENAI_API_KEY and runs the app with the local venv.
set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$SCRIPT_DIR"

if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# Fix macOS SSL cert path using certifi's CA bundle, but ONLY if it resolves to
# a real path. Exporting an empty SSL_CERT_FILE breaks TLS verification, so guard.
_CERT="$(venv/bin/python -c 'import certifi; print(certifi.where())' 2>/dev/null || true)"
if [ -n "$_CERT" ] && [ -f "$_CERT" ]; then
  export SSL_CERT_FILE="$_CERT"
  export SSL_CERT_DIR=""
fi

exec venv/bin/python run.py

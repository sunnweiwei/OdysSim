#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
  cat <<'EOF'
Usage:
  WANDB_API_KEY=... bash scripts/submit_sotopia_reward_hacking_h100.sh [--dry-run] [extra amlt run args...]

Set AMLT_CONFIG to choose a different AMLT YAML. Defaults to
amlt_sotopia_reward_hacking_h100.yaml.

HF_TOKEN is optional for public data/model access. If set or found in the local
Hugging Face CLI cache, it is passed to the AMLT job. Secrets are injected only
into a temporary AMLT config and are not written into the repo.

TRAPI_ACCESS_TOKEN is optional. If unset, this script obtains a short-lived
token from the current local `az login` for scope api://trapi/.default and
passes it as OPENAI_API_KEY to the AMLT job.

AMLT_VULN_SCAN_MODE defaults to none because the mgalleycr2 GRPO image is not
discoverable for Defender scanning from this Azure identity.
EOF
}

require_real_env() {
  local name="$1"
  local value="${!name:-}"
  if [ -z "$value" ] || [[ "$value" == \<* ]]; then
    echo "Set $name to a real value before submitting the Sotopia reward-hacking run." >&2
    exit 2
  fi
}

load_hf_token_from_login() {
  if [ -n "${HF_TOKEN:-}" ]; then
    return
  fi
  local path
  for path in "$HOME/.cache/huggingface/token" "$HOME/.huggingface/token"; do
    if [ -s "$path" ]; then
      HF_TOKEN="$(tr -d '\r\n' < "$path")"
      export HF_TOKEN
      return
    fi
  done
}

load_wandb_key_from_login() {
  if [ -n "${WANDB_API_KEY:-}" ]; then
    return
  fi
  local key
  key="$(python3 - <<'PY'
import netrc
from pathlib import Path

path = Path.home() / ".netrc"
if not path.exists():
    raise SystemExit(0)
try:
    auth = netrc.netrc(path).authenticators("api.wandb.ai")
except Exception:
    auth = None
if auth and auth[2]:
    print(auth[2])
PY
)"
  if [ -n "$key" ]; then
    WANDB_API_KEY="$key"
    export WANDB_API_KEY
  fi
}

is_probable_jwt() {
  [[ "$1" == eyJ*.*.* ]]
}

first_jwt_from_text() {
  python3 -c 'import re, sys
text = sys.stdin.read()
match = re.search(r"\beyJ[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+){2}\b", text)
if match:
    print(match.group(0))'
}

load_trapi_token_from_az_login() {
  if [ -n "${TRAPI_ACCESS_TOKEN:-}" ]; then
    if ! is_probable_jwt "$TRAPI_ACCESS_TOKEN"; then
      echo "TRAPI_ACCESS_TOKEN is set but does not look like an AAD bearer JWT." >&2
      exit 2
    fi
    OPENAI_API_KEY="$TRAPI_ACCESS_TOKEN"
    export OPENAI_API_KEY
    return
  fi
  if [ -n "${OPENAI_API_KEY:-}" ]; then
    if ! is_probable_jwt "$OPENAI_API_KEY"; then
      echo "OPENAI_API_KEY is set but does not look like an AAD bearer JWT for TRAPI local-token mode." >&2
      exit 2
    fi
    TRAPI_ACCESS_TOKEN="$OPENAI_API_KEY"
    export TRAPI_ACCESS_TOKEN
    return
  fi
  local token
  if command -v az >/dev/null 2>&1; then
    token="$(az account get-access-token --scope api://trapi/.default --query accessToken -o tsv 2>/dev/null || true)"
  elif command -v az.exe >/dev/null 2>&1; then
    token="$(az.exe account get-access-token --scope api://trapi/.default --query accessToken -o tsv 2>/dev/null | tr -d '\r' || true)"
  elif command -v cmd.exe >/dev/null 2>&1; then
    token="$(cmd.exe /C "az account get-access-token --scope api://trapi/.default --query accessToken -o tsv" 2>/dev/null | tr -d '\r' || true)"
  else
    token=""
  fi
  token="$(printf "%s" "$token" | tr -d '\r' | first_jwt_from_text)"
  if [ -z "$token" ] || ! is_probable_jwt "$token"; then
    echo "Could not obtain a valid TRAPI JWT from local az login." >&2
    echo "Run az login with the SLC account in the same WSL environment, or set TRAPI_ACCESS_TOKEN to a fresh token for scope api://trapi/.default." >&2
    exit 2
  fi
  TRAPI_ACCESS_TOKEN="$token"
  OPENAI_API_KEY="$token"
  export TRAPI_ACCESS_TOKEN OPENAI_API_KEY
}

redact() {
  python3 -c 'import os, sys
text = sys.stdin.read()
for name in ("HF_TOKEN", "WANDB_API_KEY", "TRAPI_ACCESS_TOKEN", "OPENAI_API_KEY"):
    value = os.environ.get(name)
    if value:
        text = text.replace(value, f"<{name}_REDACTED>")
sys.stdout.write(text)'
}

dry_run=0
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=1
  shift
fi

load_hf_token_from_login
load_wandb_key_from_login
load_trapi_token_from_az_login
require_real_env WANDB_API_KEY
require_real_env OPENAI_API_KEY

export AMLT_VULN_SCAN_MODE="${AMLT_VULN_SCAN_MODE:-none}"
export AMLT_PROJECT_DIR="${AMLT_PROJECT_DIR:-$(cd .. && pwd)/amlt-projects/odysim_tau}"
if [ ! -d "$AMLT_PROJECT_DIR" ]; then
  echo "AMLT_PROJECT_DIR does not exist: $AMLT_PROJECT_DIR" >&2
  echo "Create/check out the AMLT project first, e.g. project odysim-tau-usi on yingxinwustorage." >&2
  exit 2
fi

tmp_config="$(mktemp "${TMPDIR:-/tmp}/amlt_sotopia_reward_hacking.XXXXXX.yaml")"
chmod 600 "$tmp_config"
trap 'rm -f "$tmp_config"' EXIT

python3 - "$tmp_config" <<'PY'
import json
import os
from pathlib import Path
import sys

repo_dir = Path.cwd()
output = Path(sys.argv[1])
config_path = Path(os.environ.get("AMLT_CONFIG", "amlt_sotopia_reward_hacking_h100.yaml"))
config = config_path.read_text(encoding="utf-8")
config = config.replace("local_dir: $CONFIG_DIR", f"local_dir: {repo_dir}")
config = config.replace(
    'WANDB_API_KEY: "<wandb-api-key>"',
    f"WANDB_API_KEY: {json.dumps(os.environ['WANDB_API_KEY'])}",
)
config = config.replace(
    'OPENAI_API_KEY: "<trapi-access-token>"',
    f"OPENAI_API_KEY: {json.dumps(os.environ['OPENAI_API_KEY'])}",
)
hf_token = os.environ.get("HF_TOKEN", "")
config = config.replace('HF_TOKEN: ""', f"HF_TOKEN: {json.dumps(hf_token)}")
output.write_text(config, encoding="utf-8")
PY

amlt=( "$HOME/.local/bin/uvx" --from amlt --index-url https://msrpypi.azurewebsites.net/stable/leloojoo amlt )
workspace="${AMLT_WORKSPACE:-mgalleyws2}"
experiment="${AMLT_EXPERIMENT:-sotopia-reward-hacking-osim8b-exposure}"
description="${AMLT_DESCRIPTION:-sotopia_reward_hacking_osim8b_exposure}"

if [[ "$dry_run" == "1" ]]; then
  "${amlt[@]}" run "$tmp_config" "$experiment" --ws "$workspace" --description "$description" --dump "$@" | redact
  exit "${PIPESTATUS[0]}"
fi

set +e
"${amlt[@]}" run --yes "$tmp_config" "$experiment" --ws "$workspace" --description "$description" "$@" 2>&1 | redact
status="${PIPESTATUS[0]}"
set -e
exit "$status"

#!/usr/bin/env bash
# Install dependencies and download the models (~1.2 GB).
#
#   ./setup.sh [--samples] [--strict-samples] [--no-app]
#
#   --samples         also fetch the public-domain test clips (best effort: a failure warns, setup goes on)
#   --strict-samples  fetch the clips and exit non-zero if any could not be downloaded
#   --no-app          don't add app launchers (macOS: /Applications/Lipflow.app and QuietBridge.command
#                     in this folder; Linux: quietbridge.desktop in ${XDG_DATA_HOME:-~/.local/share}/applications)
#
# Downloads come from scripts/models.txt and scripts/samples.txt. Each is retried with backoff
# (honouring Retry-After) and resumed from its .part file. Tunables for CI and tests:
#   LIPFLOW_DOWNLOAD_RETRIES (default 6), LIPFLOW_DOWNLOAD_BACKOFF (first wait in seconds, default 5),
#   LIPFLOW_DOWNLOAD_MAX_WAIT (longest single wait in seconds, default 60, also caps Retry-After).
set -euo pipefail

# A generic, descriptive client name: Wikimedia throttles default library user agents.
UA="lipflow-setup/0.1 (model and sample downloader; curl)"
RETRIES="${LIPFLOW_DOWNLOAD_RETRIES:-6}"
BACKOFF="${LIPFLOW_DOWNLOAD_BACKOFF:-5}"
MAX_WAIT="${LIPFLOW_DOWNLOAD_MAX_WAIT:-60}"

warn() {
  echo "WARNING: $*" >&2
  if [ -n "${GITHUB_ACTIONS:-}" ]; then echo "::warning::$*"; fi
}

_retry_after() {  # headers-file -> seconds from the last Retry-After header (numeric form), or nothing
  [ -f "$1" ] || return 0
  tr -d '\r' <"$1" | awk 'tolower($1) == "retry-after:" { v = $2 } END { if (v ~ /^[0-9]+$/) print v }'
}

get() {  # url dest [min-bytes] -> 0 on success; on failure prints the URL and returns 1
  local url=$1 dest=$2 min=${3:-1} part="$2.part" headers="$2.headers"
  local attempt=1 wait=$BACKOFF rc code size after
  if [ -s "$dest" ]; then echo "✓ $dest"; return 0; fi
  mkdir -p "$(dirname "$dest")"
  while :; do
    echo "↓ $dest (attempt $attempt/$RETRIES)"
    rc=0
    # -f: an HTTP error writes nothing into the .part file; -C -: resume what an earlier try saved.
    code=$(curl -fL -sS -A "$UA" -C - --connect-timeout 30 --speed-limit 1024 --speed-time 60 \
                -D "$headers" -o "$part" -w '%{http_code}' "$url") || rc=$?
    if [ "$rc" -eq 0 ]; then
      size=$(wc -c <"$part" | tr -d ' ')
      if [ "$size" -ge "$min" ]; then
        mv "$part" "$dest"
        rm -f "$headers"
        return 0
      fi
      echo "  got only $size bytes (expected at least $min); starting over" >&2
      rm -f "$part"
    elif [ "$code" = 416 ] || [ "$rc" -eq 33 ]; then
      echo "  the server can't resume this file; starting over" >&2  # stale or unsupported range
      rm -f "$part"
    elif [ "$rc" -eq 22 ] && ! [[ "$code" =~ ^(408|425|429|500|502|503|504)$ ]]; then
      echo "  HTTP $code is not temporary; giving up on $url" >&2
      rm -f "$headers"
      return 1
    else
      echo "  failed (curl exit $rc, HTTP $code)" >&2
    fi
    if [ "$attempt" -ge "$RETRIES" ]; then
      echo "  giving up after $attempt attempts: $url" >&2
      rm -f "$headers"
      return 1
    fi
    after=$(_retry_after "$headers")
    local sleep_for=${after:-$wait}
    [ "$sleep_for" -gt "$MAX_WAIT" ] && sleep_for=$MAX_WAIT
    echo "  retrying in ${sleep_for}s${after:+ (server asked: Retry-After $after)}" >&2
    sleep "$sleep_for"
    attempt=$((attempt + 1))
    wait=$((wait * 2))
    [ "$wait" -gt "$MAX_WAIT" ] && wait=$MAX_WAIT
  done
}

_desktop_quote() {  # a path as one quoted .desktop Exec argument (freedesktop Desktop Entry spec)
  # Inside quotes \ " ` $ need a backslash, and the string-level escape rule doubles every
  # backslash again (so a literal \ becomes four); % becomes %% (field codes).
  local value=$1 bs='\'
  value=${value//"$bs"/"$bs$bs$bs$bs"}
  value=${value//'"'/"$bs$bs"'"'}
  value=${value//'`'/"$bs$bs"'`'}
  value=${value//'$'/"$bs$bs"'$'}
  value=${value//'%'/'%%'}
  printf '"%s"' "$value"
}

_desktop_string() {  # a plain .desktop string value: a literal backslash is written as \\
  local bs='\'
  printf '%s' "${1//"$bs"/"$bs$bs"}"
}

write_linux_launcher() {  # checkout-dir -> writes quietbridge.desktop (idempotent); skipped without a data dir
  local checkout=$1 base=${XDG_DATA_HOME:-}
  case "$base" in /*) ;; *) base="$HOME/.local/share" ;; esac  # the spec ignores a relative XDG_DATA_HOME
  if [ ! -d "$base" ]; then
    echo "(no $base; skipping the QuietBridge menu launcher)"
    return 0
  fi
  case "$checkout" in *$'\n'*|*$'\r'*)
    echo "(this folder's path contains a line break; skipping the QuietBridge menu launcher)"
    return 0 ;;
  esac
  local dir="$base/applications" file tmp
  file="$dir/quietbridge.desktop"
  mkdir -p "$dir"
  tmp="$file.tmp.$$"
  {
    echo "[Desktop Entry]"
    echo "Type=Application"
    echo "Name=QuietBridge"
    echo "Comment=Type, lip movements and gestures become text you review, then speech"
    echo "Exec=$(_desktop_quote "$checkout/.venv/bin/quietbridge")"
    echo "Path=$(_desktop_string "$checkout")"
    echo "Icon=$(_desktop_string "$checkout/quietbridge/ui/favicon.svg")"
    echo "Terminal=false"
    echo "Categories=Accessibility;Utility;"
  } >"$tmp"
  mv -f "$tmp" "$file"
  echo "✓ menu launcher: $file"
}

write_macos_command() {  # checkout-dir -> a double-clickable QuietBridge.command in the checkout (idempotent)
  local file="$1/QuietBridge.command"
  cat >"$file" <<'COMMAND'
#!/bin/bash
# Double-click to start QuietBridge; the page opens in your browser. Quit from the page's Settings.
cd "$(dirname "$0")" && exec ./.venv/bin/quietbridge "$@"
COMMAND
  chmod +x "$file"
  echo "✓ launcher: $file (double-click it in Finder)"
}

fetch_list() {  # manifest -> prints each destination that failed, returns 1 if any did
  local manifest=$1 url dest min failed=0
  while read -r url dest min; do
    case "$url" in ''|'#'*) continue ;; esac
    if ! get "$url" "$dest" "${min:-1}" </dev/null; then
      echo "FAILED: $dest from $url" >&2
      failed=1
    fi
  done <"$manifest"
  return "$failed"
}

main() {
  local samples=0 strict=0 app=1 arg
  for arg in "$@"; do
    case "$arg" in
      --samples) samples=1 ;;
      --strict-samples) samples=1; strict=1 ;;
      --no-app) app=0 ;;
      -h|--help) sed -n '2,13p' "$0"; return 0 ;;
      *) echo "unknown option: $arg (see ./setup.sh --help)" >&2; return 2 ;;
    esac
  done
  cd "$(dirname "$0")"

  if ! command -v uv >/dev/null; then
    echo "Installing uv (Python package manager)…"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
  uv sync

  # Required: without these nothing can be lip-read, so a failure stops setup.
  if ! fetch_list scripts/models.txt; then
    echo "Model download failed (see FAILED lines above). Re-run ./setup.sh to resume." >&2
    return 1
  fi

  if [ "$samples" = 1 ] && ! fetch_list scripts/samples.txt; then
    # Only tests use the clips; a rate-limited clip must not stop a working install.
    warn "Sample clips could not be downloaded (see FAILED lines above); tests that need them will be skipped. Re-run ./setup.sh --samples later."
    if [ "$strict" = 1 ]; then return 1; fi
  fi

  if [ "$(uname -s)" = Darwin ]; then
    # The small on-device cleanup model (~350 MB), so the first launch doesn't stall on it
    if uv run python -c "from mlx_lm import load; load('mlx-community/Qwen3-0.6B-4bit')" >/dev/null 2>&1; then
      echo "✓ cleanup model"
    else
      warn "the on-device cleanup model did not load; Lipflow falls back to basic cleanup"
    fi
    # The app bundle: its own permissions, Spotlight/Launchpad, Login Items
    if [ "$app" = 1 ]; then
      rm -rf ~/Applications/Lipflow.app  # older installs went here
      uv run python scripts/make_app.py --dest /Applications
      write_macos_command "$PWD"
    fi
    echo
    echo "Done. Open Lipflow from Spotlight (or: open /Applications/Lipflow.app)."
    echo "The first launch walks you through permissions, your Wispr Flow words, and ~24 practice sentences."
    echo "QuietBridge: double-click QuietBridge.command here, or run: uv run quietbridge (the page opens by itself)"
  else
    if [ "$app" = 1 ]; then
      write_linux_launcher "$PWD"
    fi
    echo
    echo "Done. Start QuietBridge from your app menu, or with: uv run quietbridge (the page opens by itself)"
    echo "(The Lipflow menu-bar dictation app is macOS/Windows only; \`uv run lipflow file VIDEO\` works here.)"
  fi
  echo "Optional stock gesture model: download it from Setup in the QuietBridge page, or: uv run python -m quietbridge.signs.assets --download"
  echo "Check everything with: uv run lipflow doctor"
}

# Sourcing the file (tests) defines the functions without running setup.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  main "$@"
fi

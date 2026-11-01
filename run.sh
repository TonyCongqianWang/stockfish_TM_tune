#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# Kaggle and common cluster images already provide these; install only if missing.
missing=()
for spec in 'python3:python3' 'git:git' 'make:make' 'g++:g++' 'curl:curl'; do
  command -v "${spec%%:*}" >/dev/null 2>&1 || missing+=("${spec#*:}")
done
if ((${#missing[@]})); then
  if [[ $(id -u) == 0 ]] && command -v apt-get >/dev/null; then
    apt-get update
    apt-get install -y "${missing[@]}" ca-certificates
  elif command -v sudo >/dev/null && sudo -n true 2>/dev/null; then
    sudo -n apt-get update
    sudo -n apt-get install -y "${missing[@]}" ca-certificates
  else
    echo "Install these packages, then rerun: ${missing[*]}" >&2
    exit 1
  fi
fi
exec python3 tools/tune.py "$@"

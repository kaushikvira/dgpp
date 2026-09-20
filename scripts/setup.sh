#!/usr/bin/env bash
# One entry point from a fresh clone; Python dependencies are not required.
# setup.py checks the interpreter's version itself, before its project imports.
set -eu
if ! command -v python3 >/dev/null 2>&1; then
  echo "DGPP setup needs Python 3.10+. On DGX OS: sudo apt-get install python3" >&2
  exit 2
fi
exec python3 "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/setup.py" "$@"

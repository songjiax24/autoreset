#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/third_party/cubiomes"

if [[ -d "$DEST/.git" ]]; then
  echo "cubiomes already cloned at $DEST"
  exit 0
fi

git clone --depth 1 https://github.com/Cubitect/cubiomes.git "$DEST"
echo "Cloned cubiomes to $DEST"
echo "Build the library separately before using cubiomes_bindings."

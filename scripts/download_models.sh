#!/usr/bin/env bash
# Download the trained model weights into ./checkpoints/.
#
# The weights (~43 MB each) are NOT stored in git. They are published as assets
# on a GitHub Release. This script fetches them into the checkpoints/ folder the
# app and API expect.
#
# Usage:
#   ./scripts/download_models.sh            # uses defaults below
#   REPO=owner/name TAG=models-v1 ./scripts/download_models.sh
set -euo pipefail

REPO="${REPO:-nurkal022/Beewings}"
TAG="${TAG:-models-v1}"
DEST="${BEEWINGS_CHECKPOINTS:-checkpoints}"

# asset filename -> local filename
FILES=("alpatov12.pt" "tofilski19.pt")

mkdir -p "$DEST"
cd "$(dirname "$0")/.."

have() { command -v "$1" >/dev/null 2>&1; }

download_one() {
  local name="$1" out="$DEST/$1"
  if [[ -f "$out" && -s "$out" ]]; then
    echo "✓ $name already present — skipping"
    return 0
  fi
  echo "↓ downloading $name ..."
  if have gh; then
    gh release download "$TAG" --repo "$REPO" --pattern "$name" --dir "$DEST" --clobber
  else
    local url="https://github.com/$REPO/releases/download/$TAG/$name"
    curl -fL --retry 5 --retry-delay 3 -o "$out" "$url"
  fi
  echo "✓ $name -> $out"
}

for f in "${FILES[@]}"; do
  download_one "$f"
done

echo
echo "Done. Models in: $DEST"
ls -lh "$DEST"/*.pt 2>/dev/null || true

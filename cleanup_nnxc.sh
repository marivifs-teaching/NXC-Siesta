#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   DRYRUN=1 bash cleanup_nnxc.sh   # show actions only
#   bash cleanup_nnxc.sh            # perform moves
#
# Optional:
#   KEEP_SYMLINKS=1 bash cleanup_nnxc.sh   # leave symlinks at old locations

DRYRUN="${DRYRUN:-0}"
KEEP_SYMLINKS="${KEEP_SYMLINKS:-0}"
DATE_TAG="$(date +%Y-%m-%d)"
RUN_DIR="runs/${DATE_TAG}_spin_validation"

# Use git mv if repo; otherwise mv
MOVE_CMD="mv"

run() {
  if [ "$DRYRUN" = "1" ]; then
    echo "[DRYRUN] $*"
  else
    eval "$@"
  fi
}

mkdirp() { run "mkdir -p '$1'"; }

move_if_exists() {
  local src="$1"
  local dst="$2"
  if [ -e "$src" ]; then
    mkdirp "$(dirname "$dst")"
    run "$MOVE_CMD '$src' '$dst'"
  fi
}

symlink_if_requested() {
  local target="$1"
  local linkname="$2"
  if [ "$KEEP_SYMLINKS" = "1" ] && [ "$DRYRUN" != "1" ]; then
    if [ ! -e "$linkname" ]; then
      ln -s "$target" "$linkname"
      echo "[SYMLINK] $linkname -> $target"
    fi
  fi
}

echo "== Creating directory structure =="
mkdirp validation/fortran
mkdirp validation/python
mkdirp validation/docs
mkdirp models/polarized
mkdirp models/unpolarized
mkdirp weights/polarized
mkdirp weights/unpolarized
mkdirp weights/grid
mkdirp weights/slim
mkdirp grids
mkdirp "$RUN_DIR"
mkdirp legacy/docs
mkdirp legacy/old-tools

echo "== Move validation assets from debug/ =="
# From debug/ (based on your listing)
for f in debug/*.f90; do
  [ -e "$f" ] || continue
  move_if_exists "$f" "validation/fortran/$(basename "$f")"
done

for f in debug/*.py; do
  [ -e "$f" ] || continue
  move_if_exists "$f" "validation/python/$(basename "$f")"
done

for f in debug/*.md; do
  [ -e "$f" ] || continue
  move_if_exists "$f" "validation/docs/$(basename "$f")"
done

# Grid file used for validation
move_if_exists "debug/nn_rho_grad_w.dat" "grids/nn_rho_grad_w.dat"

# JAX models used for validation
move_if_exists "debug/model1" "models/polarized/model1"

# Parameter directory used by validation
# (keeps the tested weights with the validation snapshot)
move_if_exists "debug/nn_params" "weights/polarized/nn_params"

echo "== Move generated CSV outputs into runs/ =="
for f in debug/*.csv; do
  [ -e "$f" ] || continue
  move_if_exists "$f" "${RUN_DIR}/$(basename "$f")"
done

echo "== Move top-level parameter directories into weights/ =="
move_if_exists "nn_params_spin"      "weights/polarized/nn_params_spin"
move_if_exists "nn_params"           "weights/unpolarized/nn_params"
move_if_exists "nn_params_grid"      "weights/grid/nn_params_grid"
move_if_exists "nn_params_pyscf_slim"   "weights/slim/nn_params_pyscf_slim"
move_if_exists "nn_params_siesta_slim"  "weights/slim/nn_params_siesta_slim"

echo "== Move Sample-Models into models/ =="
move_if_exists "Sample-Models/Polarized"   "models/polarized/Sample-Models"
move_if_exists "Sample-Models/Unpolarized" "models/unpolarized/Sample-Models"
# If Sample-Models is now empty, remove it
if [ "$DRYRUN" != "1" ] && [ -d Sample-Models ] && [ -z "$(ls -A Sample-Models)" ]; then
  rmdir Sample-Models || true
fi

echo "== Move legacy docs / duplicates =="
move_if_exists "README.txt"     "legacy/docs/README.txt"
move_if_exists "README_Unpol"   "legacy/docs/README_Unpol"
# Keep README.md at top-level (canonical). If you currently have README.MD, rename it:
if [ -e "README.MD" ]; then
  # Prefer README.md for GitHub rendering
  move_if_exists "README.MD" "README.md"
fi

echo "== Move older tool package if you decide it's legacy =="
# If you want nnxc_tools_pkg to remain primary library code, comment this out.
# For now, move it under legacy to reduce clutter.
move_if_exists "nnxc_tools_pkg" "legacy/old-tools/nnxc_tools_pkg"

echo "== Optional: leave symlinks at old locations (KEEP_SYMLINKS=1) =="
# Example symlinks for convenience (only if requested)
if [ "$KEEP_SYMLINKS" = "1" ] && [ "$DRYRUN" != "1" ]; then
  symlink_if_requested "validation" "debug"
fi

echo "== Done =="
echo "Next: review tree with:  find . -maxdepth 2 -type d | sort"

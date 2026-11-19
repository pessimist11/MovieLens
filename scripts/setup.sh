#!/usr/bin/env bash
# Create a standard venv and install the project. Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")/.."

PY_VERSION=3.12

if [ ! -x .venv/bin/python ]; then
  if command -v uv >/dev/null 2>&1; then
    echo "[setup] creating venv with Python ${PY_VERSION} via uv"
    uv python install "$PY_VERSION"
    uv venv --python "$PY_VERSION" .venv
  else
    echo "[setup] uv not found; creating venv with the system python3"
    python3 -m venv .venv
  fi
fi

if command -v uv >/dev/null 2>&1; then
  uv pip install --python .venv/bin/python -r requirements-dev.txt
  # uv-created venvs do not ship pip; install it so the editable step can run.
  uv pip install --python .venv/bin/python pip setuptools wheel
else
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements-dev.txt
fi

# Editable install; some sandboxed environments cannot build the wheel,
# in which case a plain .pth pointing at src/ gives the same import behaviour.
if ! .venv/bin/python -m pip install -e . --no-deps --no-build-isolation -q >/dev/null 2>&1; then
  SITE=$(.venv/bin/python -c "import site; print(site.getsitepackages()[0])")
  printf "%s\n" "$PWD/src" > "$SITE/movielens_dev.pth"
  echo "[setup] wheel build unavailable; used .pth instead (equivalent for imports)"
fi

.venv/bin/python -c "import movielens; print('[setup] movielens', movielens.__version__, 'ready')"

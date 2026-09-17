#!/usr/bin/env bash
# Launch HoloSmart Music Explorer.
cd "$(dirname "$0")"
exec .venv/bin/python -m app.main "$@"

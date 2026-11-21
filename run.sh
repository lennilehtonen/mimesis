#!/usr/bin/env bash
# One command: venv, print the Sensor Logger URL, start the receiver + dashboard, open the browser.
set -e
cd "$(dirname "$0")"
PORT="${PORT:-8000}"
if [ ! -d .venv ]; then
  python3.12 -m venv .venv || python3 -m venv .venv
  .venv/bin/pip install -q -r requirements.txt
fi
source .venv/bin/activate
python scripts/build_ui.py >/dev/null
scripts/find_mac_ip.sh "$PORT" || true
( sleep 2; open "http://localhost:$PORT/" ) &
exec python receiver.py --host 0.0.0.0 --port "$PORT"

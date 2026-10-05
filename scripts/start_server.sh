#!/usr/bin/env bash
# Ensure the local systemone server is running; start it detached if not, and wait until ready.
#   start_server.sh [--lan]      --lan binds 0.0.0.0 so other machines can reach it
#   start_server.sh --stop
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$DIR/../logs/server.log"
PY="$HOME/.venvs/clef/Scripts/python.exe"
[ -x "$PY" ] || PY="$HOME/.venvs/clef/bin/python"
PORT="${S1_PORT:-11436}"
mkdir -p "$(dirname "$LOG")"

pids() { netstat -ano 2>/dev/null | grep -E "[:.]$PORT .*LISTENING" | awk '{print $NF}' | sort -u; }

if [ "${1:-}" = "--stop" ]; then
  for p in $(pids); do taskkill //PID "$p" //F >/dev/null 2>&1 || kill "$p"; done
  echo "stopped"; exit 0
fi

if curl -sf "http://127.0.0.1:$PORT/s1/health" >/dev/null; then
  echo "already running: $(curl -s "http://127.0.0.1:$PORT/s1/health")"; exit 0
fi
[ -x "$PY" ] || { echo "no env at ~/.venvs/clef — run setup.sh first"; exit 1; }

HOST=127.0.0.1; [ "${1:-}" = "--lan" ] && HOST=0.0.0.0
cd "$DIR" && PYTHONIOENCODING=utf-8 nohup "$PY" s1_server.py --host "$HOST" --port "$PORT" > "$LOG" 2>&1 &
echo "starting on $HOST:$PORT (load + compile takes ~1-3 min) ..."
for _ in $(seq 1 120); do
  if grep -q "READY" "$LOG"; then grep READY "$LOG"; exit 0; fi
  if grep -qE "Traceback|Error" "$LOG" && ! pids >/dev/null; then tail -20 "$LOG"; exit 1; fi
  sleep 3
done
echo "timed out; see $LOG"; exit 1

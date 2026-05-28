#!/bin/bash
# Cleanup vLLM server and ensure port 8000 and TPU resources are released.
#
# Note: vLLM V1 uses a multi-process architecture where the API Server (CPU)
# and Engine Core (TPU) run in separate processes. Killing the API Server
# might leave the Engine Core running as an orphan, holding the TPU devices
# (/dev/vfio/*). This script ensures both are cleaned up.

echo "Cleaning up vLLM server..."

# 1. Try to kill processes gracefully
pkill -TERM -f "vllm serve" 2>/dev/null || true
sleep 2

# 2. Force kill process occupying port 8000
echo "Checking for processes occupying port 8000..."
if command -v lsof >/dev/null; then
  lsof -ti:8000 | xargs kill -9 2>/dev/null || true
else
  pkill -9 -f "vllm" 2>/dev/null || true
fi

# 3. Force kill processes occupying TPU devices (/dev/vfio/*)
# This targets the EngineCore processes in vLLM V1.
echo "Checking for processes occupying TPU devices (/dev/vfio/*)..."
if command -v lsof >/dev/null; then
  lsof -t /dev/vfio/* | xargs kill -9 2>/dev/null || true
elif command -v fuser >/dev/null; then
  fuser -k -9 /dev/vfio/* 2>/dev/null || true
else
  echo "lsof/fuser not found. Scanning /proc to find TPU users..."
  for pid_dir in /proc/[0-9]*; do
    if [ -d "$pid_dir/fd" ]; then
      for fd in "$pid_dir/fd"/*; do
        if [ -L "$fd" ]; then
          link=$(readlink "$fd" 2>/dev/null)
          if [[ "$link" == /dev/vfio/* ]]; then
            pid=${pid_dir##*/}
            echo "Killing process $pid holding TPU device $link"
            kill -9 "$pid" 2>/dev/null || true
            break
          fi
        fi
      done
    fi
  done
fi

# 4. Wait for port to be released
echo "Waiting for port 8000 to be released..."
if timeout 30s bash -c '
  while python3 -c "import socket; s=socket.socket(); s.settimeout(1); exit(0 if s.connect_ex((\"localhost\", 8000)) == 0 else 1)" 2>/dev/null; do
    sleep 1
  done
'; then
  echo "vLLM server cleanup complete and port 8000 is free."
else
  echo "Port 8000 is still occupied after 30 seconds!"
fi

# 5. Remove lock files
rm -f /tmp/libtpu_lockfile*

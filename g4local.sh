#!/bin/bash
# Self-hosted Gemma 4 edge judges (CPU only). usage: ./g4local.sh [LIMIT] ["e4b e2b"]
# Needs Docker. Caps the server at 4 CPUs / 6 GB; downloads the official Q4_0 QAT GGUF weights from Hugging Face.
set -u; LIM=${1:-}; WHICH=${2:-"e4b e2b"}
cd "$(dirname "$0")"; W=$PWD/weights; mkdir -p $W run/licenses
declare -A R=([e4b]="google/gemma-4-E4B-it-qat-q4_0-gguf|gemma-4-E4B_q4_0-it.gguf|8081" [e2b]="google/gemma-4-E2B-it-qat-q4_0-gguf|gemma-4-E2B_q4_0-it.gguf|8082")
for k in $WHICH; do
  IFS='|' read REPO F P <<< "${R[$k]}"; J="gemma-4-$k"
  curl -sfL -o run/licenses/$J-README.md "https://huggingface.co/$REPO/resolve/main/README.md"
  echo "$(date -u +%FT%TZ) $J $REPO $F apache-2.0" >> run/licenses/LICENSE_LOG.txt
  [ -s $W/$F ] || curl -sfL -o $W/$F "https://huggingface.co/$REPO/resolve/main/$F" || { echo "download failed $F"; exit 1; }
  ls -la $W/$F
  docker rm -f g4-$J >/dev/null 2>&1
  docker run -d --restart on-failure:20 --name g4-$J --cpus=4 --memory=6g -v $W:/models:ro -p 127.0.0.1:$P:8080 ghcr.io/ggml-org/llama.cpp:server -m /models/$F -t 3 -c 4096 --jinja --host 0.0.0.0 --port 8080
  for i in $(seq 90); do curl -sf 127.0.0.1:$P/health >/dev/null && break; sleep 5; done
  curl -s 127.0.0.1:$P/health; echo
  avail=$(free -m | awk '/Mem:/{print $7}'); echo "$J loaded, available RAM ${avail} MB"
  if [ "$avail" -lt 2048 ]; then echo "RAM guard: stop"; docker stop g4-$J; exit 1; fi
  START=$(date +%s)
  G4_JUDGE=$J G4_PORT=$P nice -n 19 python3 run_judges.py g4local $LIM >> run/g4local.log 2>&1
  echo "$J judged in $(( $(date +%s) - START )) s"
  docker stop g4-$J; docker rm g4-$J
done
echo G4LOCAL-DONE

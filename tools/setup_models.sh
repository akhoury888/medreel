#!/usr/bin/env bash
# Downloads model weights into $MEDREEL_MODELS (default: ../models next to the repo). ~2.5 GB, or ~4.2 GB with WITH_LARGE=1.
set -euo pipefail
DEST="${MEDREEL_MODELS:-$(cd "$(dirname "$0")/../.." && pwd)/models}"
mkdir -p "$DEST" && cd "$DEST"
B=https://github.com/k2-fsa/sherpa-onnx/releases/download
get() { [ -d "$2" ] || { echo "downloading $2"; curl -fL --retry 3 -o x.tar.bz2 "$B/$1/$2.tar.bz2"; tar xjf x.tar.bz2; rm x.tar.bz2; }; }
get asr-models sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8   # English ASR, word timing (CC-BY-4.0)
get asr-models sherpa-onnx-whisper-turbo                     # Arabic / French ASR (MIT)
[ "${WITH_LARGE:-0}" = "1" ] && get asr-models sherpa-onnx-whisper-large-v3   # best for Lebanese dialect; needs a GPU to be fast
get tts-models vits-piper-en_US-libritts_r-medium            # AI narrator voices (CC BY 4.0 data)
get tts-models vits-piper-ar_JO-kareem-medium
get tts-models vits-piper-fr_FR-siwis-medium
[ -f yunet.onnx ] || curl -fL -o yunet.onnx https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
echo "Models ready in $DEST"

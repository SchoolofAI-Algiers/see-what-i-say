#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
MODEL_SIZE="${SAM2_MODEL:-small}"
CHECKPOINT_DIR="${SAM2_CHECKPOINT_DIR:-${PROJECT_ROOT}/checkpoints}"
BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824"

case "${MODEL_SIZE}" in
  tiny)
    CHECKPOINT="sam2.1_hiera_tiny.pt"
    ;;
  small)
    CHECKPOINT="sam2.1_hiera_small.pt"
    ;;
  base_plus)
    CHECKPOINT="sam2.1_hiera_base_plus.pt"
    ;;
  large)
    CHECKPOINT="sam2.1_hiera_large.pt"
    ;;
  *)
    printf 'Unsupported SAM2_MODEL=%s. Choose tiny, small, base_plus, or large.\n' "${MODEL_SIZE}" >&2
    exit 2
    ;;
esac

if ! command -v curl >/dev/null 2>&1; then
  printf 'curl is required to download model checkpoints.\n' >&2
  exit 1
fi

mkdir -p -- "${CHECKPOINT_DIR}"
DESTINATION="${CHECKPOINT_DIR%/}/${CHECKPOINT}"
if [[ -s "${DESTINATION}" ]]; then
  printf 'Checkpoint already exists: %s\n' "${DESTINATION}"
  exit 0
fi

TEMP_FILE="$(mktemp "${CHECKPOINT_DIR%/}/.${CHECKPOINT}.download.XXXXXX")"
cleanup() {
  rm -f -- "${TEMP_FILE}"
}
trap cleanup EXIT

printf 'Downloading SAM 2.1 Hiera %s checkpoint to %s\n' "${MODEL_SIZE}" "${DESTINATION}"
curl --fail --location --retry 3 --retry-delay 2 \
  "${BASE_URL}/${CHECKPOINT}" \
  --output "${TEMP_FILE}"

if [[ ! -s "${TEMP_FILE}" ]]; then
  printf 'Download completed but the checkpoint file is empty.\n' >&2
  exit 1
fi

mv -- "${TEMP_FILE}" "${DESTINATION}"
trap - EXIT
printf 'Checkpoint ready: %s\n' "${DESTINATION}"

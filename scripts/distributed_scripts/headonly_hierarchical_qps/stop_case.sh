#!/usr/bin/env bash
set -euo pipefail

CASE_DIR="${1:-${CASE_DIR:-}}"
if [[ -z "${CASE_DIR}" ]]; then
  echo "usage: $0 <case_dir>" >&2
  exit 2
fi

for pid_file in "${CASE_DIR}"/system/*.pid; do
  [[ -f "${pid_file}" ]] || continue
  name="$(basename "${pid_file}" .pid)"
  pid="$(cat "${pid_file}")"
  if kill -0 "${pid}" >/dev/null 2>&1; then
    kill "${pid}" >/dev/null 2>&1 || true
    echo "stopped ${name} pid=${pid}"
  else
    echo "already stopped ${name} pid=${pid}"
  fi
done

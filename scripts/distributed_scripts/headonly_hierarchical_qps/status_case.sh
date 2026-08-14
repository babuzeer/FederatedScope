#!/usr/bin/env bash
set -euo pipefail

CASE_DIR="${1:-${CASE_DIR:-}}"
if [[ -z "${CASE_DIR}" ]]; then
  echo "usage: $0 <case_dir>" >&2
  exit 2
fi

if [[ ! -d "${CASE_DIR}/system" ]]; then
  echo "missing system dir: ${CASE_DIR}/system"
  exit 0
fi

for pid_file in "${CASE_DIR}"/system/*.pid; do
  [[ -f "${pid_file}" ]] || continue
  name="$(basename "${pid_file}" .pid)"
  pid="$(cat "${pid_file}")"
  if kill -0 "${pid}" >/dev/null 2>&1; then
    echo "${name}: running pid=${pid}"
  else
    echo "${name}: stopped pid=${pid}"
  fi
done

for log_file in "${CASE_DIR}"/system/*.stdout.log; do
  [[ -f "${log_file}" ]] || continue
  echo "==> ${log_file}"
  tail -n 20 "${log_file}"
done

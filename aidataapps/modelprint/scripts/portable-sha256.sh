#!/usr/bin/env bash

sha256_file() {
  local file="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$file" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$file" | awk '{print $1}'
  else
    echo "Neither sha256sum nor shasum is available." >&2
    return 1
  fi
}

write_sha256_sidecar() {
  local file="$1"
  printf '%s  %s\n' "$(sha256_file "$file")" "$(basename "$file")" >"$file.sha256"
}

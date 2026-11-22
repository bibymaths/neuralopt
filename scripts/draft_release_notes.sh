#!/usr/bin/env bash
# Draft the notes for an upcoming release from conventional commits since the last tag.
# Prints to stdout only: curate the draft into docs/release-notes/vX.Y.Z.md and a
# CHANGELOG.md section by hand (see CONTRIBUTING.md). Never writes tracked files.
set -euo pipefail
if [[ $# -ne 1 || ! $1 =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "usage: $0 vX.Y.Z" >&2
  exit 2
fi
cd "$(git rev-parse --show-toplevel)"
exec git cliff --config cliff.toml --unreleased --tag "$1" --strip header

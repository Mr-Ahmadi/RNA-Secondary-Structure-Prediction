#!/bin/sh
# Download the Rfam 15.1 files the benchmark is built from (about 6 MB).
set -e
cd "$(dirname "$0")/../data/rfam" 2>/dev/null || { mkdir -p "$(dirname "$0")/../data/rfam"; cd "$(dirname "$0")/../data/rfam"; }
base=https://ftp.ebi.ac.uk/pub/databases/Rfam/15.1
for f in Rfam.seed.gz Rfam.clanin; do curl -fsSL -o "$f" "$base/$f"; done
shasum -a 256 Rfam.seed.gz Rfam.clanin

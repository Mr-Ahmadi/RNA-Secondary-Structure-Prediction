#!/bin/sh
# Build the external baselines (RNAalifold 2.7.2 and IPknot 1.1.0) in an isolated
# conda environment. They are used only by scripts/run_baselines.py, never by ekh.
#
#   scripts/setup_baselines.sh [ENV_DIR]      (default: .bench-env in the repository)
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
ENV=${1:-$ROOT/.bench-env}
IPKNOT_COMMIT=dcaa369

conda create -y -p "$ENV" -c conda-forge -c bioconda python=3.12 viennarna=2.7.2 glpk pkg-config cmake make

rm -rf "$ENV/src/ipknot"
git clone -q https://github.com/satoken/ipknot.git "$ENV/src/ipknot"
git -C "$ENV/src/ipknot" checkout -q "$IPKNOT_COMMIT"

mkdir -p "$ENV/src/ipknot/build"
cd "$ENV/src/ipknot/build"
PKG_CONFIG_PATH="$ENV/lib/pkgconfig" "$ENV/bin/cmake" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$ENV" -DGLPK_ROOT_DIR="$ENV" ..
# The RNAlib2 pkg-config file passes the compiler flag -fno-lto to the macOS linker, which rejects it.
sed -i.bak 's/-fno-lto//g' CMakeFiles/ipknot.dir/link.txt
"$ENV/bin/make" -j8

"$ENV/bin/RNAalifold" --version
./ipknot --version

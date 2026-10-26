#!/usr/bin/env bash
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# Qualification helper for the dedicated workspace-error branch, not the PR.
# Use an isolated Git checkout and an existing, read-only CUDA bootstrap.
set -euo pipefail
: "${SLURM_JOB_ID:?Run through Slurm}"
: "${TE_WS_EXPECT_REVISION:?Set the reviewed Git revision}"
: "${TE_WS_BOOTSTRAP:?Set the existing environment bootstrap}"
: "${TE_WS_OUTPUT:?Set an output directory outside the checkout}"
: "${TE_WS_BASE_TREE:?Set the retained pre-fix TE build}"
: "${TE_WS_CUTLASS_SOURCE:?Set the existing pinned CUTLASS checkout}"
root=$(git rev-parse --show-toplevel)
test "$(git rev-parse HEAD)" = "$TE_WS_EXPECT_REVISION"
test -z "$(git status --porcelain)"
mkdir -p "$TE_WS_OUTPUT/tmp" "$TE_WS_OUTPUT/runtime"
export SLURM_TMPDIR="$TE_WS_OUTPUT/tmp" TMPDIR="$TE_WS_OUTPUT/tmp"
export SAIMAP_LOCAL_SCRATCH="$TE_WS_OUTPUT/runtime"
export ALLOW_GPU_LAUNCH=1 GPU_LAUNCH_AUTHORIZED=1 PYTHONDONTWRITEBYTECODE=1
source "$TE_WS_BOOTSTRAP"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=32 CUDA_LAUNCH_BLOCKING=0
export NVTE_FLASH_ATTN_V2=1 NVTE_FLASH_ATTN_V3=0 NVTE_FLASH_ATTN_V4=0
export NVTE_USE_CUTLASS_GROUPED_GEMM=0 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export NVTE_FRAMEWORK=pytorch NVTE_CUDA_ARCHS=90
export NVTE_WITH_NCCL_EP=0 NVTE_WITH_CUBLASMP=0 NVTE_WITH_CUSOLVERMP=0 NVTE_WITH_CUTEDSL=0
# Dependencies are supplied from the pinned offline cache below.
export NVTE_SKIP_SUBMODULE_CHECKS_DURING_BUILD=1
unset DX_PATCHES DX_INFER_DEP_OVERLAY DX_INFER_SOURCE_JIT_ONLY NVTE_DEBUG
PY=$(command -v python)
git rev-parse HEAD > "$TE_WS_OUTPUT/revision.txt"
sha256sum transformer_engine/common/gemm/cublaslt_gemm.cu tests/pytorch/test_gemm_workspace.py > "$TE_WS_OUTPUT/source_sha256.txt"

# A separate process makes a failed capture in the old core harmless to the candidate.
set +e
(
cd "$TE_WS_OUTPUT"
PYTHONPATH="$TE_WS_BASE_TREE" "$PY" -m pytest -q --tb=short \
    "$root/tests/pytorch/test_gemm_workspace.py::test_grouped_gemm_exception_joins_helper_streams" \
    --junitxml="$TE_WS_OUTPUT/baseline.xml" > "$TE_WS_OUTPUT/baseline.out" 2>&1
)
baseline_status=$?
set -e
printf '%s\n' "$baseline_status" > "$TE_WS_OUTPUT/baseline.exit"

# Reuse the previously fetched dependency at the same submodule revision.
# No package is installed into the environment.
test -d "$TE_WS_CUTLASS_SOURCE/include"
if [[ ! -d "$root/3rdparty/cutlass/include" ]]; then
    cp -a "$TE_WS_CUTLASS_SOURCE/." "$root/3rdparty/cutlass/"
fi
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-8}" NVTE_BUILD_MAX_JOBS="${SLURM_CPUS_PER_TASK:-8}"
export NVTE_BUILD_THREADS_PER_JOB=1
export CUDNN_PATH="${VIRTUAL_ENV:?}/lib/python3.12/site-packages/nvidia/cudnn"
export CPATH="$CUDNN_PATH/include:${CPATH:-}" CPLUS_INCLUDE_PATH="$CUDNN_PATH/include:${CPLUS_INCLUDE_PATH:-}"
export PYTHONPATH="$root"
"$PY" setup.py build_ext --inplace > "$TE_WS_OUTPUT/build.out" 2>&1
"$PY" -c 'import torch, transformer_engine, transformer_engine_torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name()); print(transformer_engine.__file__); print(transformer_engine_torch.__file__)' > "$TE_WS_OUTPUT/runtime.txt"
sha256sum "$root"/libtransformer_engine.so "$root"/transformer_engine_torch*.so > "$TE_WS_OUTPUT/native_sha256.txt"
"$PY" -m pytest -q --tb=short tests/pytorch/test_gemm_workspace.py \
    --junitxml="$TE_WS_OUTPUT/workspace.xml" > "$TE_WS_OUTPUT/workspace.out" 2>&1
date -u +%FT%TZ > "$TE_WS_OUTPUT/COMPLETE"

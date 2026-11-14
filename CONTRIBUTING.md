# Contributing to ComputeEdge

Thank you for your interest in contributing to ComputeEdge.

ComputeEdge is both a systems research instrument and a production-quality C++ tensor engine. Measurement reproducibility and research integrity take precedence over speculative code additions or abstractions.

---

## 1. Development Principles

1. **Research Data Immutability:** Historical raw experimental evidence under `data/sessions/orig/` (137 files) is permanently frozen and verified by `docs/verification/baseline/BASELINE.sha256`. Never modify or delete historical evidence.
2. **Measurement Parity:** Pre-registered claims C1–C17 and benchmark measurement semantics must remain reproducible. Any change affecting benchmark loops, warm-up iterations, or synchronization barriers must pass the automated equivalence gate.
3. **Targeted, Tested Changes:** Every code change must be accompanied by dedicated unit tests. Bug fixes must include regression tests.

---

## 2. Environment Setup & Prerequisites

```sh
# Required tools (macOS / Apple Silicon)
brew install cmake ninja clang-format shellcheck

# Python environment for analysis & tooling
python3 -m venv .venv
source .venv/bin/activate
pip install -e . -r benchmarks/adapters/mlx/requirements.txt pytest hypothesis ruff==0.8.4
```

---

## 3. Build & Test Workflow

ComputeEdge uses standard CMake presets defined in `CMakePresets.json`:

```sh
# Release build & full test suite (207 tests)
cmake --preset release && cmake --build --preset release && ctest --preset release --output-on-failure

# Debug build with AddressSanitizer and UndefinedBehaviorSanitizer
cmake --preset debug && cmake --build --preset debug && ctest --preset debug --output-on-failure

# Python analysis test suite (22 pytest tests)
pytest

# Linters (clang-format, ruff, shellcheck)
./tools/lint.sh

# Automated refactoring equivalence gate
./tools/verification/check_equivalence.sh --skip-build
```

---

## 4. Coding Standards

- **C++:** C++20 standard. All C++ code must format cleanly with `clang-format` according to the repository's `.clang-format`.
- **Public Headers:** Public headers under `include/computeedge/` must remain pure C++ and contain no Objective-C syntax. They must compile independently without transitive header dependencies.
- **Metal Runtime:** Objective-C++ code resides exclusively under `src/metal/` or `src/bench/` (`*.mm`). Automatic ARC is enabled.
- **Python:** Python 3.10+ with type annotations. Code must pass `ruff check .` with zero findings.
- **Shell Scripts:** All shell scripts must adhere to POSIX / Bash standards and pass `shellcheck` with zero warnings.

---

## 5. Submitting Changes

1. Create a feature branch from `main`.
2. Ensure all presets build cleanly with `-Werror`, all 207 C++ tests pass, all 21 Python tests pass, and `./tools/lint.sh` reports 0 findings.
3. Verify that `./tools/verification/check_equivalence.sh` passes.
4. Submit a Pull Request following the PR template.

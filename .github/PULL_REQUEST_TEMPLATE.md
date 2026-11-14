## Description

Provide a clear and concise summary of the changes made and the motivation behind them.

---

## Type of Change

- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New benchmark or kernel capability
- [ ] Refactoring / structural improvement (no behavioral change)
- [ ] Documentation / governance update
- [ ] Analysis / plotting / tooling enhancement

---

## Verification & Checkpoint Checklist

- [ ] `cmake --preset release && cmake --build --preset release` passes with `-Werror` (0 warnings).
- [ ] `cmake --preset debug && cmake --build --preset debug` passes with ASan + UBSan (0 warnings).
- [ ] `ctest --preset release` passes (all unit and contract tests).
- [ ] `ctest --preset debug` passes.
- [ ] `pytest` passes (all Python analysis tests).
- [ ] `./tools/lint.sh` passes (0 format, 0 ruff, 0 shellcheck findings).
- [ ] `python3 tools/verification/verify_baseline_hashes.py docs/verification/baseline/BASELINE.sha256 .` passes (137/137 files identical).
- [ ] `./tools/verification/check_equivalence.sh --skip-build` passes (all equivalence gates pass).

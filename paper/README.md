# Paper draft

Short characterization paper (target: IISWC short-paper track or IEEE CAL, 4 pages).

```sh
paper/build.sh                 # claims check -> numbers.tex -> figures -> main.pdf (draft, with annotations)
paper/build.sh --final         # main-final.pdf without TODO / VERIFY / claim-ID annotations
paper/build.sh --session s1    # draw figures from confirmation session s1 instead of the original runs
```

- **No hand-copied numbers.** Every value in `main.tex` is a macro from `generated/numbers.tex`, written by
  `benchmarks/analysis/confirm_claims.py`: cross-session medians, plus `...Min`/`...Max` ranges and
  `\verdictof{Cn}`, the confirmation status of each claim.
- **Claim IDs.** Grey `[Cn]` tags in the draft map each sentence to a claim and its pass criterion in
  `confirm_claims.py`.
- **Open items.** Red `[TODO]` and orange `[VERIFY]` markers. `refs.bib` comments carry each citation's
  verification status from `research/related_work.md`; every UNVERIFIED or SNIPPET entry must be checked against
  the primary source before submission.
- **Build requirements.** `tectonic` (Homebrew) and the project `.venv`. Outputs (`*.pdf`, `*.log`) are not
  committed.

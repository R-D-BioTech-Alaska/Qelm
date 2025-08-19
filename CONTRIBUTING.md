# CONTRIBUTING

**Project stance:** Single-file monolith by design. Do not propose refactors or modularization. If you do not understand it, don't mess it up.

## Allowed contributions
- **Bugs** with minimal diffs + failing→passing repro.
- **Performance** with measured speed/accuracy deltas.
- **UI usability** that improves end-user workflow, not code layout.
- **Docs** only when tied to a real feature or bugfix.

## Not accepted
- Spelling/grammar-only edits
- Style-only changes
- Architecture refactors or file splits
- “Drive-by” PRs without a linked issue + reproducer

## For backend changes (required)
1. Open an issue with: problem, minimal repro, proposed fix (≤200 words), risks.
2. Include a **design note** (`docs/design/<issue-id>.md`) in the PR.
3. Run `python QELM.py --selftest` and attach logs (first-100-steps loss ↓).
4. Keep diff small and localized.

## PR checklist (must pass)
- [ ] Linked issue
- [ ] Selftest logs attached
- [ ] No unrelated changes
- [ ] CI green

# Fresh pre-run nonlinear-oracle code review

Date: 2026-10-11. Skill: experiment-bridge. Fresh reviewer: /root/nonlinear_oracle_review, gpt-6-astra, xhigh. Review independence: same-family. Acceptance status: provisional.

Verdict: "Same-family provisional clearance to run the scoped CPU gate." No blocking computation, statistics, information-access or protocol issue found. Reviewer independently passed all 11 tests existing at review time and verified prepared artifacts contained 96 development, six stress and 96 confirmation vectors, with no new outcome files and no exact old/fresh vector duplicates.

Two nonblocking items were addressed before the main run:

- Numerical argmin may choose a symmetry-equivalent count vector over a mathematically lexicographically earlier one because conditional risks differ by 1.11e-16. Protocol now says first computed numerical minimum in lexicographic allocation order; this clarifies rounding without changing the distribution or optimization method.
- The derivations reference was temporarily absent while documentation was being written. experiments/nonlinear_oracle/DERIVATIONS.md was added before evaluation.

An additional independent saved-table audit and corruption regression brought the full suite to 12 passing tests before the run. Review supports this exact CPU headroom diagnostic only. It does not certify a deployable allocator, LLM gain or online-training GO.

# CPU target-score geometry audit

Target: measure conditional, token-mean thought-score Gram geometry from a frozen checkpoint and independently selected replay. Exporter accepts a supplied CPU model and processed replay batch; analyzer accepts exported exact score vectors. No training or synthetic replacement of missing target data.

Inputs: checkpoint with intended trainable parameter scope; processed full sequences, attention masks and thought boundaries, M continuations/thought; IDs of independent normalization groups and prompts. Outputs: vectors.npz and metadata.json, geometry.csv and summary.json. Real-data run pending required artifacts. CPU unit-model runs are tooling smoke only.

Success checks: preflight CLI writes availability.json; unit suite verifies trainer loss gradient against weighted thought-score vectors, masks, parameter scope and geometry identities. Actual target audit requires checkpoint/replay and provenance before producing measured results.

Base commit: recorded in BASE_COMMIT.txt. Review uses the new file list rather than unrelated existing workspace diff. Cuts: no checkpoint download, model training, outcome-conditioned clipping replay, target replacement, allocator search or benchmark redesign.

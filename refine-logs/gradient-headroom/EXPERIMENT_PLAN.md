# Experiment Plan: gradient-aware headroom

**Problem**: Credit MSE có thể không phản ánh tác động của thought lên gradient update.
**Method thesis**: Exact update-risk objective có thể có allocation headroom khác credit-risk objective; phải kiểm tra trước khi phát triển allocator.
**Date**: 2026-10-11. Đây là CPU feasibility probe đã được người dùng yêu cầu; protocol được viết trước kết quả.

## Claim Map

| Claim cần kiểm tra | Evidence tối thiểu | Block |
|---|---|---|
| C1: fixed synthetic score geometry tạo headroom cho gradient-aware allocation | Exact update oracle vượt uniform và credit oracle theo frozen development screen | B2 |
| C2: metric tính đúng covariance và cancellation | Direct-outcome checks, PSD, oracle bounds, null geometry và uniform reconstruction | B1 |

Chưa có claim về novelty, pilot deployability, real-model distribution hoặc training improvement. Anti-claim: không biến oracle biết p thành phương pháp học được từ ít pilot.

## Paper Storyline

Diagnostic này chưa phải main paper result. Giữ full tables và failure analysis; chưa chạy LLM/training, không cần frontier-model primitive.

## Experiment Blocks

### B1: Metric và exact oracle sanity — MUST-RUN

So sánh exact Binomial moments với direct squared update errors trên small budget; audit null control, covariance decomposition, PSD, global integer search, exact budget và minimum coverage. Jacobian finite-difference check ở cả hai floor regions. Tests fail thì không chạy screen. Artifact: TESTS.txt và AUDIT.json.

### B2: Frozen headroom study — MUST-RUN

K4/B16/m0=1/2; 96 consumed heterogeneous vectors và sáu stresses không đổi. Uniform, exact credit oracle, exact update oracle. Bốn fixed Gram geometries và toàn bộ24 coordinate orders khai báo trong [protocol](../../experiments/gradient_headroom/protocol.json). Primary: ratio of mean update risks, equal weights m0/three informative geometries/components; aligned-null chỉ dùng sanity, ratio NA. Paired stratified2000-bootstrap development intervals. Screen đòi lower endpoints>5% cho cả headroom so uniform và advantage so credit oracle, và ít nhất2/3 informative geometries có cả point gains>5%. Không đạt thì dừng route trong mô hình hiện tại. Đạt thì chỉ mở giai đoạn thiết kế deployable estimator; chưa confirmation.

### B3: Mechanism và failure analysis — MUST-RUN

Báo cáo toàn bộ sáu stresses, m0 fractions1/4 và1/2, fixed dispersion bins [0,.1),[.1,.25),[.25,infinity), contrast leverage và average selected counts. Không lọc geometry/vector theo kết quả. Headroom không-regression là invariant của oracle, không phải robustness guarantee cho future method. Artifact: raw risks, second moments, allocations, per-vector/per-stress tables, plot.

## Run Order and Milestones

| Stage | Task | Gate | Cost / risk |
|---|---|---|---|
| M0 | Freeze derivation/protocol, metric tests | All meaningful checks pass | CPU seconds; nonlinear cross terms |
| M1 | Enumerate455 m0=1 fixed counts, subset165 for m0=2; exact all-outcome risks | All planned conditions evaluated | CPU ceiling600s; tensor batching |
| M2 | Paired development intervals, invariants, figure | Frozen screen; no confirmation GO | CPU seconds; consumed-data bias |
| M3 | Future pilot-based method and independent confirmation | Separate design/protocol required | Deferred; no GPU budget authorized by this test |

## Compute and Data Budget

GPU-hours0. Single BLAS thread; reuse original vectors without regeneration. Exact reward sums remove reward-MC uncertainty. Bootstrap reflects consumed-vector variation only; synthetic geometry uncertainty is not estimated. Preserve incomplete outputs if CPU ceiling is exceeded.

## Risks and Mitigations

Fixed allocations do not test adaptive pilot learning: label m0 as minimum coverage. Oracle knows p: use only headroom wording. Synthetic score geometry may be unrealistic: require real score replay later under separate protocol. Nonlinear covariance is large at small budgets: use exact second moments for primary endpoint. Aligned-null relative endpoint is undefined: report NA and preexclude.

## Final Checklist

- [x] Comparators, endpoint, geometries, benchmark and screen declared before results.
- [x] Three compact required blocks; real-model evaluation deferred.
- [x] Oracle information and consumed-data limitations explicit.
- [x] Metric checks complete; active suite19PASS, frozen lattice/moment/bootstrap reconstruction PASS.
- [x] Exact outcomes, intervals and full failure tables complete in [run02](20261011_run02/DECISION.md). Frozen screen failed; no method-development or training GO under this protocol.

# Gradient update error: derivation package

## Target

Đo headroom của allocation theo gradient geometry trong mô hình Bernoulli với score vectors cố định. Mục tiêu trước mắt là exact identity và CPU diagnostic, chưa đề xuất allocator triển khai từ pilot.

## Status

COHERENT AS STATED — cho conditional fixed-score update MSE và fixed allocations. Local surrogate là approximation có điều kiện, không phải objective chính của test.

## Invariant Object

Với K thoughts, probabilities p, credit a=h_tau(p), estimator ahat=h_tau(phat), và score vectors g_i cố định, đặt u=sum_i a_i g_i, uhat=sum_i ahat_i g_i. Đại lượng chính là R_G=E||uhat-u||², conditional trên p và các g_i. Không có optimizer step, clipping hoặc gradient noise từ score estimation. Nếu trainer lấy trung bình K terms, absolute risks nhân thêm 1/K²; relative comparisons không đổi.

## Assumptions

- K=4, B=16; n_i>=m0, m0=1 hoặc2, sum_i n_i=B. Allocation được cam kết trước mọi reward; m0 chỉ là minimum coverage, không có pilot-dependent decision.
- S_i độc lập Binomial(n_i,p_i). phat_i=S_i/n_i. Dùng tất cả observations trong estimator, không có adaptive reuse trong diagnostic này.
- h_tau(x)=(x-mean(x)1)/max(s(x),tau), s là sample standard deviation (ddof=1), tau=1e-4.
- Score vectors là synthetic và cố định, không phụ thuộc outcomes, n hoặc probabilities. Gram Q_ij=<g_i,g_j> là PSD. Tổng squared norms của score vectors bằng K trong mỗi geometry.
- Credit và update oracles biết true p, chỉ chọn deterministic fixed allocation. Exact minimization xét mọi feasible integer allocation, không phải Bellman adaptive optimum.

## Notation

e=ahat-a, mu=E[e], C=Cov(e), M=E[ee^T]=C+mu mu^T. P=I-11^T/K là centering projector. J là Jacobian của h_tau tại p, nếu tồn tại. Q là score Gram matrix. Contrast leverage ell_i=(PQP)_ii, không phải chỉ ||g_i||².

## Derivation Strategy

Từ update error sang Gram contraction của exact credit-error second moment; sau đó xây exact finite outcome sum. Cuối cùng phân biệt local delta-method allocator với exact nonlinear objective.

## Derivation Map

1. Fixed g_i cho phép uhat-u=sum_i e_i g_i.
2. Bình phương norm tạo e^T Q e; expectation cho tr(QM).
3. M phân rã thành covariance và squared bias; không bỏ cross terms.
4. Fixed independent Binomial sampling cho exact outcome probabilities.
5. Linearization e≈J(phat-p) cho local surrogate. Chỉ bước5 là approximation.

## Main Derivation

**Identity 1.** R_G=E[e^T Q e]=tr(QM)=tr(QC)+mu^T Q mu. Credit risk R_I=tr(M). Off-diagonal Q và M có thể tạo cancellation hoặc reinforcement. Dùng sum_i ||g_i||² E[e_i²] sẽ bỏ các cross terms và thường đổi objective.

**Identity 2.** Vì 1^T h_tau(x)=0 với mọi x, 1^T e=0. Do đó R_G=tr(PQP M). Nếu g_i cùng bằng một vector, Q=11^T times ||g||², R_G=0 với mọi outcome, cả khi credit error khác0. Đây là structural null control; relative reduction so với uniform không xác định và phải báo NA, không gán thành0% hay100%.

**Exact outcome sum.** Với fixed n, enumerate s_i=0,...,n_i. w_n(s;p)=product_i binom(n_i,s_i) p_i^s_i(1-p_i)^(n_i-s_i). M(n,p)=sum_s w_n(s;p)[h_tau(s/n)-h_tau(p)][...]^T. Khi p_i=0 hoặc1, dùng Binomial endpoint convention. Đây là exact finite distribution calculation, chỉ có floating-point error; không có posterior quadrature hoặc Monte Carlo reward error.

**Oracle definitions.** n_credit(p)=argmin_n tr M(n,p), n_update(p,Q)=argmin_n tr(QM(n,p)). Ties dùng lexicographic first allocation. Feasible sets chứa uniform n=(4,4,4,4), nên exact update oracle risk <= uniform và <= credit oracle update risk pointwise. Các inequalities này là sanity invariants của comparator, không phải evidence một deployable algorithm thành công.

**Jacobian proposition.** Đặt z=Pp, s²=||z||²/(K-1). Nếu s>tau, J=P/s-zz^T/((K-1)s³). Nếu s<tau, J=P/tau. Tại s=tau, map có kink; không dùng Jacobian. Không dùng surrogate để chấm exact primary risks.

**Local approximation.** Khi sampling errors nhỏ, p ở vùng differentiable và không thường xuyên đi qua floor/kink, e≈J(phat-p). Fixed sampling có E[phat-p]=0, Cov(phat)=diag(p_i(1-p_i)/n_i), nên R_G≈sum_i c_i/n_i với c_i=p_i(1-p_i)(J^T Q J)_ii>=0. Leading surrogate không chứa nonlinear bias hoặc higher-order terms. Tại p all-equal, J=P/tau tồn tại nhưng delta method thường rất kém ở budget nhỏ vì noise vượt floor.

**Continuous allocator proposition.** Với active lower bounds, n_i=max(m0,sqrt(c_i/lambda)); lambda chọn để sum n_i=B. Nếu mọi c_i=0, objective local không phân biệt allocation. Integer optimization và exact nonlinear risk cần giải riêng. Với adaptive allocation, independence/diagonal covariance của empirical means không được tự động giữ; phải quay lại exact M và bao gồm bias/covariance reuse.

## Remarks and Interpretation

Synthetic score geometry là phép kiểm tra mechanism: cùng credit error có thể tạo update error khác nhau. Không suy ra true score geometry có phân phối này. Centered leverage giúp giải thích khi thought nào có trọng số; cancellation cho thấy norm-only allocation không đủ.

## Boundaries and Non-Claims

Không có training benefit, novelty claim, unbiased adaptive plug-in claim hoặc optimality giữa tất cả sequential policies. Đây là oracle diagnostic trên consumed development data; headroom dương chỉ cho phép xây phương pháp mới và protocol confirmation riêng. Các STOP verdict cũ giữ nguyên.

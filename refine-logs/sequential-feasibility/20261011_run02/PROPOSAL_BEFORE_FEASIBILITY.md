# Research proposal: Uncertainty-Aware Sequential Sampling for Nonlinear Thought Credit

**Phiên bản:** 0.1 — 11/10/2026  
**Trạng thái:** phương pháp đề xuất, chưa được thực nghiệm xác nhận.  
**Phạm vi trước mắt:** CPU; chi phí GPU bằng 0.  
**Research gap chính:** pilot information insufficiency.  
**Objective:** sai số nonlinear thought credit. Gradient error là bước kiểm chứng sau.

## 1. Tóm tắt

Trong thought-level reinforcement learning, mỗi thought được đánh giá bằng một số hữu hạn continuation rollouts. Phân bổ đều dễ triển khai nhưng có thể bỏ qua khác biệt về độ bất định giữa các thought. Phân bổ thích nghi từ một pilot nhỏ có nguy cơ khuếch đại sai số ước lượng: chỉ một hoặc hai quan sát thường chưa đủ để xác định thought nào thực sự cần thêm sampling. Ngoài ra, advantage normalization ghép toàn bộ nhóm thought thành một đại lượng nonlinear, khiến giảm phương sai từng reward estimate chưa chắc giảm sai số credit cuối cùng.

Proposal này nghiên cứu **phân bổ sampling tuần tự dựa trên giá trị thông tin đối với nonlinear credit risk**. Thay vì sử dụng pilot một lần để cam kết toàn bộ ngân sách còn lại, phương pháp cập nhật thông tin sau mỗi quan sát và chỉ ưu tiên một thought khi lợi ích dự kiến có đủ độ tin cậy theo mô hình đã khai báo. Khi chưa đạt điều kiện này, phương pháp tiếp tục lấy mẫu theo lịch cân bằng.

Đây là một hướng mới cần được bác bỏ hoặc xác nhận bằng thực nghiệm, không phải kết luận rằng JADA đã được sửa thành công. Phương pháp phải vượt uniform allocation dưới cùng tổng ngân sách, cùng terminal estimator và benchmark được khai báo trước; đồng thời tránh hồi quy lớn trên các stress regimes. Chỉ sau khi qua gate CPU mới xem xét dữ liệu score thật hoặc training.

## 2. Vấn đề và căn cứ thực nghiệm

Các kết quả hiện có chỉ ra hai vấn đề khác nhau: objective xấp xỉ có thể chọn allocation sai; và pilot có thể không cung cấp đủ thông tin để khai thác headroom của một objective tốt hơn.

| Kết quả đã có | Bằng chứng | Hàm ý cho proposal |
|---|---|---|
| JADA không qua estimator gate | Giảm MSE so với variance-only: −3.2385%, CI 95% [−3.5860%, −2.8910%] | Không tiếp tục đặt JADA làm phương pháp chính |
| Exact nonlinear oracle có headroom | Adaptive true-p oracle cải thiện 10.2214%, CI [8.8747%, 11.5681%] so với uniform trong gate K=4 | Còn tiềm năng allocation, nhưng oracle biết thông tin không có lúc chạy |
| Posterior allocator với pilot nhỏ gần uniform | Cải thiện 0.0125%, CI [0.0085%, 0.0165%] | Thay objective bằng posterior risk chưa đủ |
| Tăng pilot không cứu allocator hiện tại | Gate giữ total budget: m0=4 cải thiện 0.0832%, CI [0.0304%, 0.1361%] | Không lấy “thêm pilot” làm đóng góp chính |
| Đổi credit estimator vẫn có stress failure | Posterior credit sau convergence cải thiện trung bình 16.4346%, nhưng sparse-extreme/paired-correlated xấu hơn 17.5374% | Không dùng mean improvement để bỏ qua robustness |

Nguồn: [CPU estimator gate](archive/jada_proposal/refine-logs/cpu-estimator-gate/20261011_run01/RESULTS.md), [nonlinear oracle](archive/jada_proposal/refine-logs/NONLINEAR_ORACLE_RESULTS.md), [posterior allocator](archive/jada_proposal/refine-logs/POSTERIOR_ALLOCATOR_RESULTS.md), [pilot information gate](archive/jada_proposal/refine-logs/PILOT_INFORMATION_RESULTS.md), [posterior-credit convergence](archive/jada_proposal/refine-logs/POSTERIOR_CREDIT_CONVERGENCE_RESULTS.md). Các hàng dùng protocol và target khác nhau; không gộp chúng thành một effect size chung. Dữ liệu đã dùng là development evidence, không phải confirmation mới.

Khoảng trống cần nghiên cứu là: **với ít quan sát, làm sao quyết định nên lấy thêm thông tin ở đâu trước khi phân bổ sampling mạnh hơn?** Nonlinear-aware allocation là cách định nghĩa lợi ích của thông tin. Gradient-aware estimation là cách kiểm tra xem credit tốt hơn có tạo ra update chính xác hơn hay không.

## 3. Câu hỏi và hai giả thuyết

**Câu hỏi chính:** Uncertainty-aware sequential sampling có thể cải thiện nonlinear thought-credit estimation so với uniform allocation khi ngân sách quan sát nhỏ hay không?

- **H1 — acquisition:** cập nhật quyết định sau từng quan sát có thể khai thác thông tin tốt hơn quyết định allocation một lần sau pilot, dưới cùng budget và terminal estimator.
- **H2 — reliability:** điều kiện tin cậy và fallback cân bằng có thể giảm stress regression so với acquisition không có điều kiện tin cậy, mà vẫn giữ lợi ích trung bình có ý nghĩa.

Đây là giả thuyết thực nghiệm. Không có theorem bảo đảm phương pháp vượt uniform trên mọi probability vector. Nếu pilot và các quan sát tiếp theo vẫn không đủ để tạo tín hiệu đáng tin, kết quả hợp lệ là tiếp tục dùng uniform và dừng hướng này.

## 4. Định nghĩa bài toán và target

Một prompt có K thought cố định. Thought j có success probability p_j; mỗi lần lấy mẫu tạo reward X_{j,t}∼Bernoulli(p_j). Có B lần lấy mẫu tổng cộng, bao gồm mọi quan sát ban đầu và quan sát dùng để cập nhật acquisition. Giả định các reward độc lập có điều kiện theo p; thought và mô hình không thay đổi trong một episode đánh giá.

Đặt P=I−11ᵀ/K và τ=10⁻⁴. Target chính theo normalization contract của trainer là

\[
r(p)=\sqrt{\|Pp\|^2/(K-1)},\qquad
a(p)=h_\tau(p)=\frac{Pp}{\max(r(p),\tau)}.
\]

Sau B quan sát, state D_B chứa counts n_j và successes s_j. **Terminal estimator chính được cố định**:

\[
\widehat a(D_B)=h_\tau(\widehat p),\qquad \widehat p_j=s_j/n_j.
\]

Uniform, fixed-pilot và sequential allocation đều dùng estimator này trong comparison chính. Điều đó cho phép quy thay đổi risk cho allocation. Posterior mean credit không phải terminal estimator chính của proposal này.

Credit risk tại p là

\[
R(\pi,p)=\mathbb E_{D_B\sim\pi,p}
\left[\|\widehat a(D_B)-a(p)\|_2^2\right],
\]

với π là sampling policy. Mục tiêu là giảm risk trên một distribution khai báo trước, dưới cùng B. Estimator sau adaptive sampling có thể biased; proposal đánh giá trực tiếp MSE và bias–variance decomposition, không giả định raw unbiasedness.

[Target contract](archive/jada_proposal/idea-stage/docs/ADVANTAGE_TARGET_CONTRACT.md) phân biệt h_tau với target lịch sử f_epsilon của CPU gate. Bảng replication lịch sử giữ target cũ; comparison mới dùng h_tau nhất quán. Không lấy khác biệt giữa hai target làm gain của phương pháp.

## 5. Phương pháp đề xuất

Tên làm việc: **uncertainty-aware sequential thought sampling**. Tên này không hàm ý novelty đã được xác nhận.

### 5.1. Thông tin được phép dùng

Runtime chỉ dùng (n_j,s_j), ngân sách còn lại và randomness khai báo trước. Acquisition dùng prior độc lập Beta(1/2,1/2) duy nhất trong prototype đầu. Không dùng true p, tên regime, xác suất benchmark, outcomes tương lai hay score geometry chưa đo được. Benchmark heterogeneous không đồng nhất với prior của policy; mismatch là một phần của đánh giá.

Mỗi thought nhận m0 quan sát ban đầu. Sau đó policy chọn một thought cho từng quan sát đến khi dùng hết B. Mọi pilot đều được tính vào B; không có ngân sách thông tin miễn phí.

### 5.2. So sánh hành động bằng terminal nonlinear risk

Gọi U là lịch completion cân bằng: chọn thought có n_j nhỏ nhất; xử lý ties bằng một permutation ngẫu nhiên độc lập, được cố định cho mỗi episode. Với state D và q=B−Σn_j quan sát còn lại, định nghĩa

\[
Q_U(p,D,q)=\mathbb E[\|\widehat a(D_B)-a(p)\|^2\mid p,D,\text{completion }U].
\]

Với hành động j, lấy một reward ở thought j rồi dùng U cho q−1 quan sát còn lại:

\[
Q_j(p,D,q)=p_jQ_U(p,D^{j,1},q-1)
 +(1-p_j)Q_U(p,D^{j,0},q-1).
\]

Đặt Δ_j(p,D,q)=Q_U(p,D,q)−Q_j(p,D,q). Đây là lợi ích của **một deviation rồi completion cân bằng**, chưa phải lợi ích của toàn bộ policy sequential sẽ tiếp tục replan.

Posterior cung cấp hai đại lượng:

\[
\bar\Delta_j=\mathbb E_{p\mid D}[\Delta_j],\qquad
c_j=\Pr_{p\mid D}(\Delta_j>0).
\]

Policy chỉ chọn trong tập A(D)={j: barΔ_j>0 và c_j≥0.90}, rồi lấy j có barΔ_j lớn nhất. Nếu tập rỗng, thực hiện một bước U. Sau reward mới, cập nhật D và tính lại. Ngưỡng 0.90 là lựa chọn thiết kế predeclared, không phải mức bảo đảm frequentist 90%; không tune ngưỡng sau khi nhìn winners.

Baseline acquisition không gate dùng cùng Q_j và posterior nhưng chọn barΔ_j lớn nhất, gồm cả bước U làm lựa chọn với gain bằng 0. Comparison này kiểm tra riêng giá trị của điều kiện tin cậy.

### 5.3. Pseudocode

```text
sample m0 rewards from each of the K thoughts
while total count < B:
    update the declared posterior from all observed counts
    evaluate one-action nonlinear terminal-risk improvements
    retain actions with positive mean gain and posterior reliability >= 0.90
    if retained actions exist:
        sample the action with largest mean gain
    else:
        sample a least-observed thought using the balanced tie schedule
return h_tau(success counts / observation counts)
```

### 5.4. Giới hạn tính toán

K nhỏ dùng enumeration của outcome states và quadrature posterior; phần reward-state sum là exact, integral trên prior liên tục là approximate. K lớn dùng paired Monte Carlo và posterior integration có kiểm tra convergence. Không thay Q bằng Jacobian surrogate trong phương pháp chính.

Acquisition có thể đắt hơn rollout đã tiết kiệm. Vì vậy phải báo cả số reward samples và CPU wall time, memory, số risk evaluations. Nếu one-action lookahead không khả thi ngay ở K=4/B=16, đây là implementation gate thất bại; không tuyên bố compute efficiency từ sample efficiency.

## 6. Theory package và proof obligations

**Target:** terminal nonlinear credit MSE. **Status:** COHERENT AS STATED dưới mô hình Bernoulli và fixed total budget; cải thiện so với uniform chưa được chứng minh. **Invariant object:** R(π,p), không phải phương sai riêng từng thought. Ký hiệu và assumptions ở Sections 4–5.

**Derivation map:** adaptive likelihood → posterior → conditional terminal risk → one-action improvement → sequential policy → đánh giá risk ngoài prior. Approximation chỉ xuất hiện ở integration và simulation; từ local action gain sang global policy gain là giả thuyết, không phải identity.

### 6.1. Posterior dưới predictable adaptive sampling — proposition

Nếu hành động tại thời điểm t chỉ phụ thuộc history và random seed độc lập với p, likelihood của history factorizes thành action probabilities và Bernoulli observations. Action factors không phụ thuộc p khi history đã biết, nên triệt tiêu khi chuẩn hóa posterior. Với prior đã chọn,

\[
p_j\mid D\sim\operatorname{Beta}(1/2+s_j,\,1/2+n_j-s_j).
\]

Điều này làm posterior cập nhật hợp lệ trong mô hình; không làm empirical mean unbiased dưới adaptive counts, và không bảo đảm calibration khi prior không mô tả benchmark.

### 6.2. Conditional risk và Bellman diagnostic — identities

Terminal cost C(D)=E_{p|D}[||a_hat(D)−a(p)||²]. Với horizon r, Bayes-optimal diagnostic dưới prior có recursion

\[
V_0(D)=C(D),\quad
V_r(D)=\min_j\{\rho_j(D)V_{r-1}(D^{j,1})
 +(1-\rho_j(D))V_{r-1}(D^{j,0})\},
\]

trong đó ρ_j=(1/2+s_j)/(1+n_j). Recursion exact đối với posterior model và terminal estimator đã cố định. Nó là small-case information-constrained control: không biết true p. Enumeration có thể quá lớn; chỉ dùng cho K=2 và các horizon thực sự khả thi sau complexity audit. Đây không phải thuật toán triển khai mặc định, cũng không phải oracle trên benchmark thực tế.

Nếu cả diagnostic này không cho headroom đáng kể trong prior-predictive small case, cần xem lại khả năng khai thác thông tin của thiết kế trước khi mở rộng. Headroom theo prior không chứng minh headroom theo heterogeneous benchmark.

### 6.3. Jacobian — identity để audit, không làm acquisition chính

Với z=Pp và r>τ,

\[
J_h(p)=P/r-zz^\top/((K-1)r^3).
\]

Với r<τ, J_h=P/τ. Tại r=τ, map không có một Jacobian duy nhất theo mọi hướng. Kiểm tra finite differences phải tách hai miền và không dùng threshold kink để kết luận Jacobian sai. Linearization J_hδ là local surrogate; không mặc định đủ chính xác ở small budgets hoặc near-equal probabilities.

### 6.4. Reuse và bias — boundary

Raw fixed-stage reuse covariance đã có chỉ áp dụng khi allocation main phụ thuộc pilot và main sample mean conditional-unbiased theo allocation đó. Khi allocation liên tục nhìn main outcomes, điều kiện này không còn tự động đúng. Proposal phải tính risk của full adaptive path hoặc mô phỏng policy đầy đủ. Không chuyển proof cũ sang policy sequential bằng cách thay counts.

Posterior reliability c_j là độ tin cậy theo prior cho deviation được định nghĩa trong Section 5; không phải confidence interval của gain thực nghiệm, cũng không phải proof toàn policy cải thiện.

## 7. Experiment roadmap và benchmark khai báo trước

### 7.1. Development CPU nhỏ trước

Implement K=2/4, B=16, m0=1/2, cùng h_tau và terminal plug-in. Dùng dữ liệu đã consumed chỉ để debug và phát hiện lỗi, không tạo confirmation CI. Các checks bắt buộc: total budget đúng B; coverage n_j≥m0; không truy cập true p; permutation equivariance theo phân phối ties; posterior update đúng; path probabilities sum to one; risk enumeration khớp simulation; candidate fallback khớp U; Jacobian audit ở hai miền.

Đối chiếu Bayes DP ở small cases khả thi, uniform, ungated acquisition và gated acquisition. Trước khi chạy benchmark mới, đóng băng code, integration settings, ties, random seeds và protocol hash. Nếu complexity hoặc numerical stability không đạt, dừng ở feasibility report.

### 7.2. Grid chính

Sweep K∈{2,4,8,16}, B∈{16,32,64}, m0∈{1,2,4}. Để đối chiếu với fixed-stage gate, chỉ giữ B≥K(m0+1), tức 26 configurations. Sequential chỉ bắt buộc Km0 initial observations; điều kiện mạnh hơn này giữ cùng grid so sánh, không phải tính chất lý thuyết của sequential sampling.

Distribution benchmark được khóa **trước khi xem kết quả mới**: sáu components có trọng số bằng nhau:

1. p_j iid Beta(0.5,0.5).
2. p_j iid Beta(2,2).
3. p_j iid Beta(8,8).
4. p_j iid Beta(0.5,3).
5. p_j iid Beta(3,0.5).
6. Logistic-normal: location ℓ∼N(0,1), scale σ với log σ∼Uniform(log 0.15,log 3), p_j=sigmoid(ℓ+σz_j), z_j iid N(0,1).

Một vector chia sẻ ℓ và σ trong component cuối. Không thay distribution để thuận lợi cho candidate. Planned fresh set: 384 vectors, 64/component, dùng seed 2449482027 sau khi kiểm tra seed chưa dùng trong repository; nếu trùng, thay và ghi lại **trước execution**, không sau outcomes.

Giữ sáu K=4 stress regimes: symmetric-wide [0.95,0.55,0.45,0.05]; symmetric-mid [0.90,0.52,0.48,0.10]; all-equal [0.5,0.5,0.5,0.5]; near-equal [0.55,0.52,0.48,0.45]; asymmetric [0.90,0.80,0.70,0.10]; sparse-extreme [0.99,0.99,0.50,0.01]. Không tự suy rộng sáu vector này sang K khác; distribution sweep bao phủ K khác, stress table này ghi rõ K=4.

### 7.3. Baselines và attribution

Comparison mới gồm uniform/U, variance-only fixed-pilot, JADA fixed-pilot, posterior nonlinear fixed-pilot allocator đã có, ungated sequential và gated sequential. True-p exact nonlinear oracle là upper-reference riêng, không phải deployable baseline. Small-case Bayes DP được ghi rõ dùng cùng thông tin counts nhưng tối ưu theo prior.

Fixed-pilot methods dùng cùng terminal plug-in trong allocation comparison chính. Đồng thời giữ historical factorial allocation {uniform, variance-only, JADA, oracle} × reuse {main-only, naive, fixed 0.5, budget-precommitted} làm bảng audit riêng. Mỗi reuse rule phải giữ định nghĩa và target của protocol tương ứng. Không gán fixed-stage reuse labels cho sequential mà chưa định nghĩa một estimator và kiểm tra adaptive bias.

Primary sequential estimator sử dụng mọi observations qua sample means. So với uniform cùng estimator là comparison bắt buộc. Uniform posterior credit và frozen scale control là sensitivity comparisons bổ sung; chúng không được dùng để chọn terminal estimator thắng nhất sau sweep.

### 7.4. Monte Carlo và uncertainty

Exact enumeration khi khả thi; trường hợp lớn dùng 10,000 paired episodes/vector/configuration. Tạo independent reward streams indexed bởi (vector, episode, thought, sample index); mỗi policy đọc stream tương ứng theo action của nó. Điều này ghép outcomes hợp lệ ngay khi policies có counts khác nhau. Randomness của posterior integration tách khỏi evaluation streams.

Primary estimand là ratio of mean risks: 1−mean(R_candidate)/mean(R_uniform), equal component weights, equal weights across 26 configurations. Không lấy mean của per-vector percentage gains khi baseline risk gần zero. Báo 95% stratified cluster-bootstrap CI với 2,000 replicates, resample vectors trong components và giữ toàn bộ paired configurations của một vector. Báo MC standard error; bootstrap này phản ánh cả finite-episode noise nhưng không thay thế convergence check. Nếu MC precision không đủ so với ngưỡng gate, báo inconclusive thay vì tăng trials sau khi nhìn winners.

### 7.5. Pilot fraction, dispersion và leverage

Báo gain theo pilot fraction f=Km0/B và true group dispersion r(p). True p chỉ dùng evaluator để stratify. Bins dispersion cố định [0,0.025), [0.025,0.1), [0.1,∞); không chia bins theo winners. Báo số vectors, baseline risk và CI cùng mỗi bin.

Trong validation gradient, leverage của thought j là G_jj/tr(G), với G là score Gram. Báo gain theo bảy geometries đã khai báo, số lần policy lệch khỏi uniform, phân bố terminal counts và tần suất fallback. Phân tích tương tác pilot fraction–dispersion–leverage là diagnostic; subgroup CI không thay gate aggregate và stress gate.

## 8. Decision gates và tiêu chí dừng

**CPU credit GO** yêu cầu đồng thời: lower bound CI 95% của primary reduction >5% so với uniform cùng terminal estimator; không có stress/configuration nào có credit risk regression >10%; numerical convergence đạt tolerances khai báo trước; và overhead được báo đầy đủ. Stress risks dùng exact sums khi khả thi; nếu dùng MC, chỉ nhận GO khi upper confidence bound của regression ≤10%. Chưa đủ precision là inconclusive.

Integration checks dùng ít nhất hai nested resolution levels với random scrambles độc lập. Trước confirmation phải khóa concrete counts cho từng K và memory limit từ feasibility run. Tolerances đề xuất: posterior risk changes ≤1% của uniform reference risk, action-score numerical intervals không vượt zero tại quyết định ưu tiên; nếu không đủ numerical evidence, fallback U. Sai số số học của integration phải được phân biệt với posterior uncertainty c_j. Đây là **proposal**, chưa là execution-ready preregistration khi các counts này chưa được ghi vào protocol.

**Dừng** nếu candidate không có hiệu ứng thực dụng, stress gate thất bại, acquisition quá đắt hoặc acquisition gần như luôn quay về uniform. Không rescue bằng đổi prior, ngưỡng, distribution, target hay lựa chọn subset K/B sau khi thấy kết quả. Một method revision cần proposal/protocol và fresh evaluation mới; không sửa verdict cũ.

## 9. Gradient-aware validation sau credit gate

Giữ bảy synthetic trace-normalized score Grams đã có; bốn primary geometries là contrast-rank1, heterogeneous-diagonal, paired-correlated và dense-correlated. Với S chứa thought score vectors, G=SᵀS, tại old-policy ratio bằng 1 và score cố định,

\[
\mathbb E\|S(\widehat a-a)\|^2
=\mathbb E[(\widehat a-a)^\top G(\widehat a-a)].
\]

Đây là exact conditional update-error identity trong slice đã nêu. Policy chính vẫn không nhìn G; bước này hỏi nonlinear credit gains có chuyển sang update gains không. Yêu cầu lower CI >5% so với uniform trong aggregate bốn primary Grams và không stress regression >10%, với protocol mới frozen trước evaluation. Dữ liệu dùng để qua credit gate không tự trở thành fresh gradient confirmation.

Synthetic G không chứng minh lợi ích training. Bước tiếp theo, nếu cả hai gate đạt, là đo frozen-model thought scores và replay update errors với token weighting, answer loss và clipping contract thực tế. Away from ratio=1, sign-dependent clipping có thể làm fixed-Gram representation không đủ. GPU chỉ được cân nhắc cho bước real-model này theo kế hoạch riêng; proposal hiện tại không đặt job GPU.

## 10. Related work và ranh giới novelty

[EPIG-Tree: Compute-Optimal Branching for Gradient-Efficient Reinforcement Learning](https://arxiv.org/html/2609.20004v1) đã nghiên cứu branching/suffix allocation dựa trên gradient information, score norms và continuation uncertainty; đồng thời thảo luận hạn chế khi pilot nhỏ. Vì vậy “dùng gradient để phân bổ rollout” không phải novelty claim của proposal này.

[Adaptive Rollout Allocation for Online Reinforcement Learning with Verifiable Rewards](https://arxiv.org/html/2602.01601v1) nghiên cứu phân bổ rollout giữa prompts bằng dự đoán success probability từ history và prompt similarity. Proposal này tập trung vào sampling giữa các thought cố định trong một prompt, với nonlinear group-normalized target.

Novelty candidate hẹp hơn là phối hợp **sequential information acquisition, exact nonlinear group-credit risk và posterior reliability fallback** dưới finite budgets. Đây mới là positioning dự kiến, chưa phải kết luận novelty audit toàn diện. Trước khi viết claim “first” hoặc triển khai nghiên cứu dài hạn, cần đối chiếu thêm Bayesian experimental design, adaptive stratified sampling và best-arm identification. Không tự coi việc ghép các thành phần quen thuộc là đóng góp đủ mạnh.

## 11. Deliverables và đóng góp kỳ vọng

1. Một policy cụ thể và implementation CPU có kiểm tra information access, budget invariants và numerical convergence.
2. Theory package phân biệt adaptive posterior validity, biased terminal estimation, local lookahead và global risk; exact small-case diagnostic nơi khả thi.
3. Frozen benchmark với full-grid tables, stress results, paired CIs và gain theo pilot fraction, dispersion, leverage.
4. Quyết định GO/STOP có thể kiểm tra từ artifacts, bao gồm overhead; không dùng một biểu đồ trung bình để che regression.

Đóng góp kỳ vọng là xác định được khi nào acquisition có thể vượt việc cam kết allocation từ pilot nhỏ, hoặc chỉ ra rằng headroom oracle chưa khai thác được ở ngân sách nghiên cứu. Cả hai đều cần bằng chứng; proposal không hứa hẹn một kết quả dương.

## 12. Lộ trình thực hiện

| Giai đoạn | Công việc | Điều kiện chuyển bước |
|---|---|---|
| A — formulation và feasibility | Implement risk/action calculation K nhỏ, invariants, posterior và Jacobian audits, CPU profiling | Công thức/code nhất quán; integration và cost khả thi |
| B — protocol freeze | Khóa settings, seeds, benchmark, baselines, terminal estimator, CI và stress rules | Protocol/hash đầy đủ, không còn implementation choice phụ thuộc winners |
| C — CPU confirmation | Full 26-config sweep, paired MC/exact risks, uncertainty và stress tables | Credit GO theo Section 8 |
| D — gradient validation | Synthetic fixed-score validation với data separation và protocol riêng | Update-error GO theo Section 9 |
| E — real-model evaluation | Frozen-score replay rồi mới quyết định training | Proposal riêng, measurements thật và budget rõ ràng |

**Bước tiếp theo cụ thể:** implement feasibility spine ở K=2/4 để kiểm tra acquisition và overhead; chưa mở full sweep hoặc training khi feasibility/protocol chưa hoàn tất. Proposal này không thay đổi các verdict STOP của các phương pháp trước.

# Kết luận gradient-aware CPU headroom

**Đã có headroom so với uniform, nhưng chưa qua screen cho lợi ích riêng của gradient-aware objective.** Không mở training, không đổi geometry/benchmark/threshold để chọn trường hợp thắng.

Test dùng K=4, B=16, minimum coverage m0=1/2; toàn bộ96 consumed heterogeneous vectors, sáu stress cases và24 coordinate orders. Enumerate455 fixed allocations ở m0=1, subset165 ở m0=2. Ba comparator là uniform, true-p exact credit oracle và true-p exact update oracle. Oracles chọn count vector trước mọi reward, không dùng observed outcomes. Đây chưa phải allocator pilot-based.

Objective đã được dẫn xuất: R_Q=tr(Q E[ee^T])=tr(Q Cov(e))+E[e]^T Q E[e], với e=ahat-a và Q_ij=<g_i,g_j>. Exact nonlinear outcome sums giữ toàn bộ covariance/cancellation; local Jacobian surrogate chỉ xuất hiện trong derivation, không dùng cho primary risks.

| Endpoint pooled | Reduction | Development interval hai phía95% | Frozen requirement |
|---|---:|---:|---|
| Update oracle so với uniform | 16,12% | [14,15%;18,38%] | Lower endpoint>5%: đạt |
| Update oracle so với credit oracle | 4,39% | [3,91%;4,84%] | Lower endpoint>5%: chưa đạt |

Pooled metric dùng ratio of mean risks, trộn đều m0 và ba informative geometries; equal component weights trên consumed vectors. Aligned-null là structural zero: ratios NA, loại khỏi pooled endpoint trước khi chạy. Bootstrap phản ánh vector variation trên development data đã dùng, không phải confirmation.

| Geometry đã khai báo | Headroom so uniform | Lợi ích thêm so credit oracle |
|---|---:|---:|
| Orthogonal | 12,28% | 0% |
| Opposed | 15,76% | 3,98% |
| Unequal norm | 20,45% | 9,32% |

Chỉ1/3 informative geometries có cả hai point gains>5%, trong khi protocol đòi ít nhất2/3. Vì vậy verdict giữ **NO_PREDECLARED_GRADIENT_HEADROOM_SIGNAL**. Tên verdict chỉ nói screen chưa đạt; không nói headroom bằng0 hoặc gap gradient-aware bất khả thi. Unequal-norm signal là diagnostic có điều kiện, không đủ để chọn lại distribution hay tuyên bố phương pháp tổng quát thắng.

Minimum coverage chiếm25% budget: headroom20,13%, advantage5,36%, advantage interval[4,67%;6,01%]. Với50%: headroom12,12%, advantage3,48%, interval[3,14%;3,82%]. Đây là fixed-count lower-bound effect, không phải lợi ích của việc dùng pilot tốt hơn. Dispersion bins và contrast leverage được báo cáo đầy đủ trong [figure](gradient_headroom.png), [dispersion](dispersion.csv) và [leverage](leverage.csv); không có subgroup nào được dùng để thay primary screen.

**Bước nghiên cứu tiếp theo, nếu tiếp tục:** giữ route synthetic này ở trạng thái chưa qua screen. Trước một proposal gradient-aware mới, cần audit score geometry của target task/model cố định bằng dữ liệu độc lập: norm heterogeneity và covariance/cancellation có thực sự đủ lớn không? Chỉ khi có problem anchor này mới khai báo protocol riêng và xây estimator dùng p-unknown pilot observations. Test này không tự động cho phép chạy full sweep hay training.

CPU execution hoàn tất trong **17,27 giây**, GPU cost0. 19 tests của active suite PASS; direct-error discrepancy tối đa1,25e-14, uniform reconstruction1,07e-14. Independent saved-lattice/moment/hash/bootstrap reconstruction PASS. Đây là deterministic local audit; chưa có independent-family semantic review. Run01 lỗi JSON serialization được giữ cùng snapshots; run02 chỉ sửa audit writer, không thay numeric objective hoặc protocol.

[Derivation](DERIVATION_PACKAGE.md) · [Protocol](protocol.json) · [Results](RESULTS.md) · [Figure](gradient_headroom.png) · [Numerical audit](AUDIT.json) · [Saved audit](SAVED_AUDIT.json).

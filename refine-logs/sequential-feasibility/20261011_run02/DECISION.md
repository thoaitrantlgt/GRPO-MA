# Kết luận CPU feasibility

**Thiết kế hiện tại chưa đạt: dừng prototype one-action lookahead rồi completion cân bằng trước full sweep.**

Đã chạy K=2/4, B=16, m0=1/2 trên 96 probability vectors development cũ và sáu stress vectors, tổng cộng 2.040 method/vector evaluations. Risk theo reward outcomes được tính exact cho policy đã cố định; tích phân posterior liên tục vẫn là approximation. Run đã sửa mất khoảng 70 giây CPU; không dùng GPU.

| Method | K=4, m0=1 | K=4, m0=2 |
|---|---:|---:|
| Gated sequential | 0% | 0% |
| Ungated sequential | +0,0218% | +0,0218% |
| JADA với trainer-target Jacobian | −1,2857% | −0,1931% |

Các số là reduction của mean credit MSE so với uniform, trên consumed development. Ungated có descriptive interval khoảng [0,0085%, 0,0370%], rất nhỏ so với ngưỡng 5%; không phải fresh confirmation. Gated trùng uniform trên toàn bộ outcome histories đã enumerate, nên interval [0,0] mô tả sự đồng nhất policy trong trial này, không chứng minh mọi policy sequential đều vô ích.

**Nguyên nhân cấu trúc:** khi một hành động khác rồi completion cân bằng vẫn tạo cùng terminal counts, phân phối terminal successes và credit risk không đổi. Với B=16, one-action score chỉ có thể bắt đầu phân biệt allocation trong một quan sát cuối ở K=2, hoặc ba quan sát cuối ở K=4. Nó không định giá thông tin sớm để hỗ trợ các quyết định allocation thích nghi sau đó.

Quadrature nhỏ cũng chưa ổn định: numerical fallback chiếm khoảng 89–90% các quan sát sau pilot ở K=2 và 98–99% ở K=4. Vì vậy trial chưa kiểm chứng được tác dụng riêng của reliability gating. Tăng quadrature không loại bỏ identity về terminal counts.

Kiểm chứng: 10 tests PASS; exact completion risk khớp enumeration độc lập; budget, posterior, Jacobian và permutation-averaged bias–variance decomposition đã kiểm tra. Paired MC giữ 512 episodes/vector/configuration: các deviation hiếm vẫn chưa đủ precision; zero-SE discrepancies được ghi inconclusive. Primary risks dùng exact reward-state propagation.

Small-case Bayes Bellman control ở K=2 cải thiện khoảng 0,1204% trên consumed data, với projected stress regression lớn. Nó chỉ tối ưu theo finite prior và không có reliability gate; không cứu verdict của phương pháp chính và không phải một replacement đã qua gate.

**Điều chỉnh cần thiết:** acquisition phải xét tác động của quan sát lên các quyết định allocation tương lai. Cần một continuation policy có khả năng thay đổi terminal allocation, hoặc một information-constrained planning objective. Chưa chạy full 26 configurations, gradient validation hay training.

Nguồn: [báo cáo đầy đủ](RESULTS.md), [mechanism derivation](mechanism-diagnostic/RESULTS.md), [metrics](metrics.csv), [protocol](protocol.json), [audit](AUDIT.json), [biểu đồ](feasibility.png). Independent assumption review là same-family/provisional, không phải chứng nhận method thành công.

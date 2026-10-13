# Quyết định sau K4 information control

**Dừng mở rộng control hiện tại trước full sweep hoặc training.** Control đã chạy thành công; research screen chưa đạt. Không dùng GPU.

K=4, B=16, m0=1/2; giữ nguyên iid Jeffreys prior, terminal credit estimator, 96 consumed heterogeneous vectors và sáu stress cases. Cả ba mức quadrature đã khai báo (16/64/256 base points mỗi scramble, support 768/3072/12288) đều được chạy, không chọn thêm mức sau khi xem kết quả. Bellman planning cho phép các quan sát mới thay đổi terminal allocation, khắc phục hạn chế cấu trúc của one-action/balanced completion.

Ở mức cao nhất, reduction theo ratio of mean risks là **1,4866%**, interval mô tả development **[1,0203%; 1,9956%]**. Primary metric trộn đều m0=1/2, dùng trọng số đều giữa sáu thành phần heterogeneous, loại stress khỏi trung bình. Interval là percentile bootstrap hai phía 95% trên dữ liệu đã dùng; không phải xác nhận generalization hay sai số quadrature.

| Kiểm tra | Kết quả | Ngưỡng khai báo | Đánh giá |
|---|---:|---:|---|
| Lower endpoint của reduction | 1,0203% | >5% | Chưa đạt |
| Stress regression lớn nhất | 0,7028% | <=10% | Đạt |
| Aggregate risk change / uniform | 0,0922% | <=1% | Đạt |
| Stress risk change / uniform | 5,5695% | <=1% | Chưa đạt |
| Prior value change / prior uniform | 0,6213% | <=1% | Đạt |
| Lower-policy regret / prior uniform | 0,7777% | <=1% | Đạt |

Risk benchmark được tính bằng exact reward-state propagation. Bellman tối ưu chính xác trên finite support; approximation của continuous prior chưa qua kiểm tra ổn định. Control này không phải oracle biết true p và không chứng minh pilot information gap là bất khả thi. Kết quả chưa xác nhận lợi ích lên gradient hoặc training.

Ba mức chạy mất **22,51 giây CPU**, process peak working set **206,22 MiB**. 15 implementation tests PASS. [Kiểm tra độc lập policy đã lưu](SAVED_AUDIT.json) PASS: direct-error risk khớp trong 2,23e-15, Bellman checks với likelihood độc lập và budget/coverage đúng. Review assumptions cùng họ mô hình có trạng thái provisional; các làm rõ metric sau review được ghi riêng, không đổi frozen code, policy hay kết quả.

Không mở K8/K16, full 26-config confirmation hoặc GPU cho control hiện tại. Nếu tiếp tục nghiên cứu, cần một thiết kế và protocol riêng cho model/objective khác, cùng lý do cơ chế rõ ràng; không đổi benchmark để ưu tiên các stress đang thắng. Dữ liệu hiện tại là evidence phát triển, không dùng làm confirmation mới.

[Bảng kết quả đầy đủ](RESULTS.md) · [Biểu đồ](control_stability.png) · [Protocol](protocol.json) · [Làm rõ metric](PROTOCOL_CLARIFICATION.md) · [Review assumptions](../../../implement-stage/information_control_k4/SILENT_ASSUMPTION_SWEEP.json).

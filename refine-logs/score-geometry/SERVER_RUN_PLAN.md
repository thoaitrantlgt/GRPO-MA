# Plan chạy score geometry trên server

**Mục tiêu:** xác minh norm heterogeneity và cancellation của thought-score gradients trên Qwen2.5-VL-3B / ShareRobot trước một proposal gradient-aware mới. Đây là frozen-checkpoint audit, không phải training hoặc thử allocator. Các STOP trước giữ nguyên.

**Trạng thái thực tế:** preflight và prompt-selection CLI đã có. Replay capture, gradient exporter và analyzer target chưa được implement. Vì vậy bạn chạy được bước0–1 ngay; bước2–4 dưới đây là đặc tả cần hoàn thiện trước main job, không phải lệnh đã chạy được. Không gọi `train.py` hoặc `scripts/run_grpo_lora.sh` để thay thế exporter: chúng gọi training và tạo adapter mới theo cấu hình.

## 1. Inputs cần đặt trên server

| Input | Yêu cầu |
|---|---|
| Base model | Local Qwen2.5-VL-3B-Instruct, đủ weight shards, tokenizer và multimodal processor |
| Checkpoint/adapter | Checkpoint tại step cố định; nếu training dùng LoRA, dùng đúng adapter đã train và lưu parameter names/shapes; không khởi tạo adapter ngẫu nhiên |
| Holdout manifest |128 prompts có ID duy nhất, độc lập với dữ liệu đã dùng để chọn checkpoint/method; JSON list theo ShareRobot schema |
| Images | Image root đúng với relative `image` paths trong manifest |
| Replay nếu có | Full sequences, attention masks, thought boundaries, M, thứ tự thought/answer và normalization group IDs; multimodal tensors/processor settings |
| Training settings | training_args.json hoặc bản khai báo checkpoint, task, generation temperature, token/image limits, dtype, adapter config và world size |

Manifest trong repo chưa có; metadata/README.md ghi URL download là placeholder. Không sử dụng URL đó như nguồn dữ liệu đã được xác minh. Có128 prompts chưa đủ chứng minh independence: ghi rõ split provenance và checkpoint selection history. Nếu chỉ có train data, kết quả phải mang nhãn consumed/exploratory, không báo independent audit.

## 2. Quy mô và tài nguyên

- Main:128 independent prompts, một frozen group4 thoughts/prompt, seed1732052027. Smoke: first4 prompts trong chính selection này; không chọn lại theo norm/reward.
- Ưu tiên replay hiện có. Nếu phải generate: freeze settings từ checkpoint trước; planned fallback4 thoughts ×4 answers, max_completion_length1024, max_pixels1003520. Temperature còn phải lấy từ training_args trước generation; không dùng default ngầm. Không truncate prompt để cứu OOM: trainer hiện bỏ max_prompt_length setting.
- Dùng một GPU server sẵn có để thử smoke, sequential thought forwards. Chưa cam kết mức VRAM hoặc số giờ vì phụ thuộc images, sequence lengths và trainable parameter count. Không cần dựng4-GPU training job chỉ để đo một K4 group.
- Sau smoke, dự toán main cost≈128×thời gian/group, cộng load/I/O. Báo cả peak VRAM/RAM và số parameters. Không chuyển ước tính thành lời hứa wall time.
- Gradient coordinates giữ nguyên LoRA training scope, base/vision frozen như checkpoint. Frozen base checkpoint không có trained adapter chưa đại diện cho LoRA checkpoint geometry; phải khai báo một audit scope khác trước khi chạy.
- Dùng dtype được ghi trong checkpoint; accumulate Gram ở float64 từ exact stored per-parameter gradients, ghi source gradient dtype. Model.eval() tắt dropout: đây là deterministic geometry, chưa bao gồm training dropout variability.

## 3. Bước0 — preflight chạy được ngay

Chạy từ repository root trên Linux server đã có môi trường GRPO-MA. Sửa bốn paths và adapter path theo máy bạn. Không tự cài toàn bộ requirements đè môi trường server đang hoạt động; trước tiên thu inventory.

```bash
export SCG_MODEL=/path/to/Qwen2.5-VL-3B-Instruct
export SCG_ADAPTER=/path/to/frozen-trained-adapter
export SCG_MANIFEST=/path/to/independent_share_robot_manifest.json
export SCG_IMAGES=/path/to/ShareRobot/trajectory/images
export SCG_RUN=refine-logs/score-geometry/server_run01

python -m experiments.score_geometry.preflight \
  --checkpoint "$SCG_MODEL" \
  --adapter "$SCG_ADAPTER" \
  --manifest "$SCG_MANIFEST" \
  --images "$SCG_IMAGES" \
  --output "$SCG_RUN/preflight"
```

Nếu checkpoint thật là full-model thay vì LoRA adapter, bỏ `--adapter` và khai báo parameter scope riêng; không áp dụng kết luận LoRA. Preflight chỉ kiểm tra file/package presence và GPU inventory; exporter còn phải load đầy đủ checkpoint để xác minh shard completeness. Artifact: availability.json, environment versions; exit2 nghĩa thiếu capabilities. Thư mục output phải mới.

Dependencies có trong requirements của repo: torch/transformers/peft/trl/accelerate và qwen-vl-utils. Ghi version đang chạy vào artifact. Việc source-check ở máy local phát hiện thiếu peft/trl không nói môi trường server cũng thiếu. CLI không load weights hoặc start GPU job.

## 4. Bước1 — khóa prompt selection, chạy được ngay

```bash
python -m experiments.score_geometry.prepare_manifest \
  --manifest "$SCG_MANIFEST" \
  --images "$SCG_IMAGES" \
  --split-description "Describe the independent split and checkpoint selection history here" \
  --output "$SCG_RUN/selection"
```

Salt score-geometry-v1, sort SHA256 của prompt ID, chọn128, smoke first4. Selection độc lập score/reward. CLI kiểm tra tất cả selected image paths và từ chối thiếu prompts, duplicate IDs hoặc missing images; không thay prompt thiếu ảnh bằng prompt khác. `split-description` là operator declaration, không phải chứng nhận independence. Artifact: selected_manifest.json và selection.json.

Nếu IDs đã có trong development/replay được dùng để chọn phương pháp, ghi data status exploratory và dừng independent claim; cần nguồn holdout mới cho mục tiêu audit đã nêu.

## 5. Bước2 — capture replay và implement exporter trước khi chạy

**Chưa có CLI cho bước này.** Cần implement trong experiments/score_geometry/, không sửa training dynamics. Các acceptance requirements:

1. Load đúng base + trained adapter; lưu checkpoint path/revision/step và ordered trainable parameter names/shapes. Chỉ bật gradients đúng parameter scope. Không optimizer, backward step, save updated checkpoint hoặc RNG-dependent adapter initialization.
2. Load/reuse frozen complete replay. Nếu một prompt có nhiều replay groups, chọn group có salted SHA256(group_id) nhỏ nhất trước khi đọc scores; salt score-geometry-v1. Nếu chưa có, generate sau khi khóa generation settings và seed, giữ image processing như target. Lưu generated sequences trước khi xem geometry. Reward/format chỉ dùng làm diagnostics, không chọn subgroup tốt.
3. Mỗi thought lấy continuation đầu tiên (rows0,M,2M,3M) như trainer. Recompute per-token logps từ full sequence; mask positions `[thinking_start-1, thinking_end-1)` theo code, bỏ prompt/answer/padding tokens. g_i=gradient(mean thought-token logprob). Empty mask cho zero score và flag; invalid/truncated thoughts giữ selected IDs và flags, không thay thế bằng resampling.
4. Tiếp tục dùng nguyên image-conditioned sequence và đúng processor; không bỏ images để giảm VRAM. Microbatch forwards theo thought; parameter coordinates chung. Multimodal tensor slicing phải dùng đúng per-sequence patch/image boundaries, không slice image tensors theo batch index một cách ngầm định.
5. Score matrix S[K,D], Q=SS^T. Có thể accumulate Q chính xác theo parameter blocks và chỉ lưu Q + parameter manifest để giảm disk; không dùng random projection hoặc normalize từng g_i. Nếu không lưu full S, phải lưu raw per-thought norms và kiểm tra direct weighted update norm từ gradients trên smoke.
6. Group4 thoughts/prompt là standalone K4 diagnostic. Trainer normalization thật gather global values; nếu replay đến từ nhiều ranks/prompts, lưu actual full normalization group, tính full-group H hoặc báo K4 local block chỉ là proxy. Không center mỗi local rank rồi gọi đó là full training geometry.

Acceptance smoke: weighted sum of g_i/K phải khớp negative gradient của thought PPO loss tại ratio1 trên cùng sequences/parameters. Answer/KL và clipping ở policy đã đổi không thuộc identity này. Validate relative gradient discrepancy<=1e-4 và absolute Gram contraction discrepancy<=1e-6×max(reference squared norm,1), có dtype ghi rõ. Không đạt thì dừng; không tăng tolerance sau khi xem main results. Đây là numerical gate, không thay gate5% cũ.

Artifact contract: checkpoint.json; parameter_manifest.json; replay.pt hoặc equivalent tensor chunks; group_metadata.jsonl; gram.npz(Q[G,K,K], token_counts[G,K], valid_flags[G,K]); smoke_audit.json. Với ragged actual global groups dùng per-group files, không padding thêm thoughts rồi coi là samples.

## 6. Bước3 — main geometry audit, sau smoke PASS

**Chưa có analyzer CLI.** Implement analyzer theo [server protocol](server_protocol.json), trước main run. Primary descriptive metrics trên128 frozen prompts:

| Metric | Definition / interpretation |
|---|---|
| Raw norm CV | std_population(sqrt(diag Q))/mean(norm); norm heterogeneity |
| Centered norm CV | Tương tự với H=PQP; leverage trên normalized-credit contrast |
| Contrast trace fraction | tr(H)/tr(Q) |
| Raw cancellation ratio | ||sum g_i||²/(K tr Q)=1−contrast fraction |
| Centered effective rank | tr(H)²/tr(H²), rank<=K−1 |
| Pairwise cosines | Q_ij/sqrt(Q_ii Q_jj), báo NA với zero norm |

Zero denominators NA; báo số zero/invalid/incomplete groups, không lọc theo score. Incomplete normalization groups không tính vào primary summary; planned128 coverage chưa đủ thì ghi INCOMPLETE, không tự thay missing group. Report64/128 completion progress chỉ để monitor, không dùng early stop theo winners.

Equal prompt weights, 2000 paired prompt-bootstrap replicates, seed1732052027, two-sided95 percentile intervals. Một frozen checkpoint; chưa cần3 training seeds vì không có training. Có thể đo stochastic replay replication sau bằng protocol riêng; không bổ sung theo kết quả.

Artifacts: geometry.csv(all selected IDs), summary.json, intervals.json, geometry.png/.pdf, runtime.json, AUDIT.json, DECISION.md. Biểu đồ raw/centered norm CV, cosine distribution, contrast effective rank và histogram normalized leverage. Raw checkpoint-scale norms giữ lại, không chỉ trace-normalized matrices.

## 7. Quy tắc quyết định

Audit này chỉ xác minh problem anchor. Norm heterogeneity hoặc cancellation lớn không chứng minh allocation hiệu quả. Không có automatic training/method GO và không sửa benchmark synthetic vì unequal-norm case đang thắng. Nếu geometry gần isotropic hoặc chủ yếu common mode, ghi rõ bằng evidence; nếu heterogeneous, bước sau là **protocol mới** cho unknown-p, pilot-based method và update-error headroom trên data phù hợp, không mở training ngay.

OOM: giảm concurrent forwards/CPU-offload score buffers, giữ sequences/images/parameters và IDs. Không giảm token limits, bỏ long thoughts, đổi model hoặc parameter scope để cứu một run. Vẫn không vừa thì báo incomplete và cấu hình cần thiết; không rental/launch GPU tự động trong plan này.

## 8. Thứ tự job và readiness

| Job | Work | Dependencies | Status hiện tại |
|---|---|---|---|
| J0 | Server preflight | Supplied checkpoint/manifest/images | CLI READY, chưa chạy trên server |
| J1 | Freeze128 prompt IDs | Independent holdout and image paths | CLI READY, chưa chạy trên server |
| J2 | Exporter implementation + 4-prompt smoke | Frozen replay/checkpoint, gradient tests | NOT IMPLEMENTED |
| J3 |128-prompt exact Gram extraction | J2 PASS, cost measured | BLOCKED on J2 and inputs |
| J4 | Geometry/bootstrap report | Gram dataset and analyzer | NOT IMPLEMENTED |
| J5 | Decide whether to propose pilot-based method | J4 evidence | DEFERRED, no training GO |

Bạn có thể bắt đầu J0/J1 bằng các lệnh trên. Để mở J2 cần paths/settings thực có và hoàn thiện exporter; plan này không chứa một lệnh đo gradients giả định là đã tồn tại.

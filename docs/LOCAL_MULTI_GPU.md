# 本地模型 + 多卡 (如 8×H100) 使用说明

## 当前行为（单进程 serve-api）

- **模型加载**：`device_map="auto"`，32B 模型通常占 1–2 张卡（如 8×H100 80GB 下只用 1–2 张，其余空闲）。
- **推理**：`/v1/chat/completions` 用信号量串行，同一时刻只跑 1 个请求，避免单进程内多路推理抢显存。
- **客户端**：本地模型默认 **8 并发**（`--workers 8`），用于保持请求队列，避免服务端等请求。

因此：**单进程 + 当前参数对 8 卡机是合理的**——用 1–2 卡做推理，客户端 8 并发保持队列即可。

## 参数建议（8×H100 80GB）

| 参数 | 建议 | 说明 |
|------|------|------|
| 生成 `--workers` | **8**（默认） | 保持队列，无需再调高（服务端仍串行）。 |
| 评估 `--parallel` | **20**（默认） | 评估走 InceptBench，与本地卡数无关。 |
| 单条超时 | **900s**（客户端默认） | 32B 长 prompt 单条可能 10–15 分钟，可设 `LOCAL_API_TIMEOUT=1200`。 |
| `max_tokens` | **4096**（本地） | 足够 MCQ JSON 输出。 |

## 每张卡利用率都不到 10%，正常吗？怎么改进？

- **是否正常**：**正常**。单进程只跑 1 个 32B 模型，通常只占 1–2 张卡；且 **单条序列生成（batch=1）** 时，显存带宽、解码步长等会限制算力利用，单卡利用率 5–15% 很常见。其余 6–7 张卡空闲，整体看起来「每张都不到 10%」。
- **怎么改进**：要 **用满 8 卡、提高利用率**，需要 **8 个 serve-api 实例**（每卡 1 个），客户端对多地址 **轮询** 发请求：
  1. 起 8 个 serve-api（每实例绑定 1 张卡、不同端口），见下方「多实例启动」。
  2. 设置 **逗号分隔的多地址**，客户端会自动轮询：
     ```bash
     export LOCAL_API_BASE="http://127.0.0.1:8000,http://127.0.0.1:8001,http://127.0.0.1:8002,http://127.0.0.1:8003,http://127.0.0.1:8004,http://127.0.0.1:8005,http://127.0.0.1:8006,http://127.0.0.1:8007"
     python main.py closed-loop --model local --pass-rate-target 0 --max-rounds 10
     ```
  3. 这样 8 个 worker 会分散到 8 个实例，8 张卡同时推理，**总吞吐约 8 倍、各卡利用率会明显上升**。

## 8 卡利用率都很低（如均 &lt;5%）是否正常？如何验证？

- **是否正常**：单进程 serve-api 用 `device_map="auto"` 时，32B 通常只占 **1 张或 2 张** GPU，其余 6–7 张空闲（0% 或极低）是正常的。但 **承载模型的那 1–2 张卡**在真正做 `model.generate()` 时，利用率通常会到 **几十到接近 100%**。若 **所有 8 张卡都长期 &lt;5%**，可能表示：
  - 请求还没进到 `model.generate()`（卡在 tokenizer/拼 prompt 等 CPU 步骤），或
  - 当前没有请求在执行推理（请求未到达或已超时）。
- **如何验证**：
  1. **看 serve-api 终端**：是否有 `[推理] 开始  prompt 约 X token, ..., cuda:N X.XGB`。有则说明已进入推理且模型在 `cuda:N`；随后应出现 `[推理] 完成  耗时 Xs  ...`。
  2. **看 GPU 占用**：`watch -n 1 nvidia-smi`，看哪张卡有 **显存占用**（约 60–70GB 为 32B bf16）。有显存的那张在推理时 **利用率** 会升高；若一直 &lt;5%，可能是单步算力小或卡在非 GPU 步骤。
  3. **看进程**：`nvidia-smi` 里应能看到 **python/main 进程** 挂在某 1–2 张卡上；若没有，说明模型可能未用 GPU 或进程不是当前 serve-api。
  4. **查 API**：`curl -s http://127.0.0.1:8000/config` 可看 `device`（如 `cuda:0`），确认模型所在卡。

## 若 10+ 分钟无一条生成输出

- **原因**：服务端串行处理，**首条请求**若 prompt 很长（多示例），32B 推理可能需 **10–15 分钟**，客户端会一直等该条返回后才打印。
- **确认**：看 **serve-api 所在终端**是否出现 `[推理] 开始  prompt 约 X token` 和 `[推理] 完成  耗时 Xs`。若能看到「开始」且长时间无「完成」，说明单条推理很慢或卡住（可查 GPU 占用）。
- **处理**：耐心等首条完成，或减少示例数（如 fewer-shot）、降低 `max_tokens`，或延长 `LOCAL_API_TIMEOUT`。

## 多实例启动（用满 8 卡）

1. **启动 8 个 serve-api**（每卡 1 个，端口 8000–8007；`serve-api` 已支持 `--port`）：
   ```bash
   for i in 0 1 2 3 4 5 6 7; do
     CUDA_VISIBLE_DEVICES=$i python main.py serve-api --model models/qwen3-32B/final_model --port $((8000+i)) &
   done
   # 8 个实例在后台运行，再开终端执行闭环
   ```
   或逐条起（便于看日志）：
   ```bash
   CUDA_VISIBLE_DEVICES=0 python main.py serve-api --model models/qwen3-32B/final_model --port 8000 &
   CUDA_VISIBLE_DEVICES=1 python main.py serve-api --model models/qwen3-32B/final_model --port 8001 &
   # ... 8002–8007
   ```

2. **客户端设置多地址轮询**（脚本已支持逗号分隔，自动轮询）：
   ```bash
   export LOCAL_API_BASE="http://127.0.0.1:8000,http://127.0.0.1:8001,http://127.0.0.1:8002,http://127.0.0.1:8003,http://127.0.0.1:8004,http://127.0.0.1:8005,http://127.0.0.1:8006,http://127.0.0.1:8007"
   python main.py closed-loop --model local --pass-rate-target 0 --max-rounds 10
   ```

3. **workers**：默认 8 即可（每个实例 1 个请求）；可设 `--workers 16` 让部分实例排队，进一步提高吞吐。

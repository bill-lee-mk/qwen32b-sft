# Token 用量与费用说明

## 缓存

DeepSeek API 的**上下文硬盘缓存默认开启**，无需修改代码。当请求的前缀（system prompt、examples 等）与之前的请求相同时，重复部分会计为「缓存命中」，价格更低：

- 缓存命中：1 元/百万 token（vs 未命中 4 元/百万）
- 多轮闭环时，同 (std,diff) 的 system+examples 可能命中上一轮缓存

## 为什么费用随轮次增加？

**每轮是 237 次新的 API 调用**，每次调用都有 input + output：

- 1 轮 = 237 次调用 ≈ ¥5
- 10 轮 = 2370 次调用 ≈ ¥50

每轮「单题成本」大体相同，但**轮次越多，总调用次数越多，总费用自然增加**。

## 实际用量记录

`--all-combinations` 生成完成后，用量会写入与 mcqs 同名的 token 文件：

- `mcqs_237_deepseek-reasoner_round1.json` → `token_237_deepseek-reasoner_round1.json`
- `mcqs_237_deepseek-reasoner_best_96_2.json` → `token_237_deepseek-reasoner_best_96_2.json`

```json
{
  "model": "deepseek-reasoner",
  "mcqs_path": "evaluation_output/mcqs_237_deepseek-reasoner_round1.json",
  "n_calls": 237,
  "prompt_tokens": 725000,
  "completion_tokens": 142200,
  "prompt_cache_hit_tokens": 0,
  "prompt_cache_miss_tokens": 725000,
  "estimated_cost_cny": 5.18
}
```

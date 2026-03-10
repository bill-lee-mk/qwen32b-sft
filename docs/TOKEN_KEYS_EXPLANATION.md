# Token 消耗字段说明

`token_237_xxx_roundN.json` 中各键含义（现已不再单独生成，用量信息写入综合日志）：

| 键 | 含义 |
|----|------|
| **prompt_tokens** | 输入 token 总数：发送给模型的 prompt 总长度（含系统提示、示例、用户问题等） |
| **completion_tokens** | 输出 token 总数：模型生成的回答长度 |
| **prompt_cache_hit_tokens** | 缓存命中 token 数：与之前请求相同的前缀部分，API 直接复用缓存，不计入计费或按更低单价计费 |
| **prompt_cache_miss_tokens** | 缓存未命中 token 数：本次请求中需要重新计算的 prompt 部分（prompt_tokens = cache_hit + cache_miss） |
| **estimated_cost_cny** | 估算费用（人民币元）：按 DeepSeek/Kimi 等定价（缓存命中、未命中、输出不同单价）计算 |

**说明**：闭环多轮时，首轮 cache 较少，后续轮次因 prompt 规则相似，cache 命中会增多，费用会下降。

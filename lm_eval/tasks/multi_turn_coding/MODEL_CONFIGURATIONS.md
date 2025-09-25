# Model Configuration Guide

## 配置文件对比

| 模型 | 配置文件 | max_gen_toks | 温度 | 推荐用途 |
|-----|----------|-------------|------|---------|
| **Claude Code SDK** | `multi_turn_coding.yaml` | 2000 | 0.0 | 文件操作密集型任务 |
| **DeepSeek** | `multi_turn_coding_deepseek.yaml` | 4000 | 0.0 | 代码生成和性价比 |
| **OpenAI** | `multi_turn_coding_openai.yaml` | 4000 | 0.0 | 通用性和稳定性 |
| **通用配置** | `multi_turn_coding_universal.yaml` | 2500 | 0.0 | 任何模型的安全配置 |

## 使用示例

### Claude Code SDK (推荐)
```bash
lm_eval --model claude-code-local \
        --model_args model=claude-3-haiku-20240307,multi_turn=true \
        --tasks multi_turn_coding \
        --limit 1
```

### DeepSeek (高性价比)
```bash
lm_eval --model deepseek \
        --model_args model=deepseek-v3.1 \
        --tasks multi_turn_coding_eval_deepseek \
        --limit 1
```

### OpenAI (通用性强)
```bash
lm_eval --model openai-completions \
        --model_args model=gpt-4-turbo \
        --tasks multi_turn_coding_eval_openai \
        --limit 1
```

### 通用配置 (兜底选择)
```bash
lm_eval --model anthropic_llms \
        --model_args model=claude-3-sonnet-20240229 \
        --tasks multi_turn_coding_eval_universal \
        --limit 1
```

## Token限制说明

### 为什么不同模型设置不同的token限制？

1. **Claude Code SDK**: 
   - 限制: 2000 tokens
   - 原因: SDK会将token数乘以倍数，2000实际可能变成8192，需要控制在Claude Haiku的4096限制内

2. **DeepSeek**: 
   - 限制: 4000 tokens
   - 原因: DeepSeek模型支持更长的输出，且成本低廉，可以设置更高限制

3. **OpenAI**: 
   - 限制: 4000 tokens
   - 原因: GPT-4支持8K+输出，4000tokens为安全边界，确保复杂任务完整输出

4. **通用配置**: 
   - 限制: 2500 tokens
   - 原因: 保守设置，兼容大多数模型的限制

## 性能优化建议

### 对于成本敏感的用户
```bash
# 使用DeepSeek，成本最低
lm_eval --model deepseek --tasks multi_turn_coding_eval_deepseek --limit 5
```

### 对于质量要求高的用户
```bash  
# 使用Claude Code，功能最强
lm_eval --model claude-code-local --tasks multi_turn_coding --limit 3
```

### 对于需要并发测试的用户
```bash
# 使用OpenAI，API最稳定
lm_eval --model openai-completions --tasks multi_turn_coding_eval_openai --limit 10
```

## 常见问题

### Q: token超限怎么办？
**A**: 使用更保守的配置:
```bash
lm_eval --model your-model \
        --model_args max_tokens=1500 \
        --tasks multi_turn_coding_eval_universal
```

### Q: 如何提高输出质量？
**A**: 调整温度和token数:
```bash
lm_eval --model your-model \
        --model_args temperature=0.1,max_tokens=3500 \
        --tasks your-task
```

### Q: 如何加速评估？
**A**: 使用更高的batch size:
```bash
lm_eval --model your-model \
        --tasks your-task \
        --batch_size 5 \
        --limit 10
```

## 配置文件自定义

如果需要创建自己的配置，复制通用配置并修改:

```yaml
generation_kwargs:
  temperature: 0.0
  max_gen_toks: 自定义数值  # 根据你的模型调整
  until: []
  do_sample: false
```

推荐token限制:
- **小型模型** (如GPT-3.5): 2000-2500
- **大型模型** (如GPT-4): 3000-4000  
- **专业模型** (如Claude Code): 1500-2000
- **开源模型**: 根据具体模型文档设置
# LoCoMo 评测脚本（来自 EverOS）

来源：[EverMind-AI/EverOS](https://github.com/EverMind-AI/EverOS) 的 `methods/EverCore/evaluation/`

已排除 EverMemOS 自有方法（`evermemos` adapter 及相关配置），保留 LoCoMo 评测流水线与第三方 memory 系统 adapter。

## 目录结构

```
methods/EverCore/
├── evaluation/          # 评测主代码
│   ├── cli.py           # 命令行入口
│   ├── config/
│   │   ├── datasets/locomo.yaml
│   │   └── systems/     # mem0, memos, memu, zep
│   ├── data/locomo/locomo10.json
│   └── src/
│       ├── core/        # add → search → answer → evaluate 流水线
│       ├── evaluators/  # LLM Judge / Exact Match
│       └── adapters/    # 各 memory 系统 adapter（不含 evermemos）
├── src/common_utils/    # 最小依赖工具
├── src/memory_layer/llm/  # 轻量 LLM 封装
├── env.template
└── requirements-eval.txt
```

## 安装

```bash
cd methods/EverCore
pip install -r requirements-eval.txt
cp env.template .env
# 编辑 .env，填入 LLM_API_KEY
```

## 运行 LoCoMo 评测

在 `methods/EverCore` 目录下执行：

```bash
# 快速冒烟测试（mem0 示例）
python -m evaluation.cli --dataset locomo --system mem0 --smoke

# 完整评测
python -m evaluation.cli --dataset locomo --system mem0

# 只跑 search / answer / evaluate
python -m evaluation.cli --dataset locomo --system mem0 --stages search answer evaluate
```

结果保存在 `evaluation/results/locomo-{system}/`。

## 接入自己的 RAG 方法

1. 在 `evaluation/src/adapters/` 新建 adapter，继承 `BaseAdapter`，实现 `add` / `search` / `answer`。
2. 在 `evaluation/src/adapters/registry.py` 注册你的 adapter。
3. 在 `evaluation/config/systems/` 添加对应 yaml 配置。
4. 运行：`python -m evaluation.cli --dataset locomo --system your_system`

## 评测流程

1. **Add** — 写入对话并建索引  
2. **Search** — 为每个问题检索记忆  
3. **Answer** — 用检索结果生成答案  
4. **Evaluate** — LLM Judge 打分（LoCoMo 默认）

## 数据

LoCoMo 数据已包含在 `evaluation/data/locomo/locomo10.json`。  
原始数据来源：https://github.com/snap-research/locomo

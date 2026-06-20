# Seed Preview CV

通过 Minecraft World Preview 第一视角初始画面，预测 seed 是否适合速通。

- **游戏版本**: Minecraft Java Edition 1.16.1
- **截图来源**: 自制 mod（训练与推理保持一致）
- **程序化标签**: [cubiomes](https://github.com/Cubitect/cubiomes)

## 项目结构

```
src/seed_preview_cv/
  common/           # 配置、路径等共享工具
  seed_selection/   # 环节 1：筛选进入数据集的种子
  collection/       # 环节 2：mod 采集截图与精确出生点
  labeling/         # 环节 3：基于出生点构建标签
  cubiomes_bindings/ # cubiomes C 库 Python 绑定（待实现）
```

数据集构建与最终 CV 任务的关系：

| 环节 | 目的 |
|------|------|
| seed_selection | 从大量 seed 中平衡采样，选出进入数据集的子集 |
| collection | 对选定 seed 采集截图 + 精确出生点 |
| labeling | 用 cubiomes + 出生点生成训练标签 |

## 环境

使用 [uv](https://docs.astral.sh/uv/) 管理依赖：

```bash
uv sync
uv run pytest
```

## cubiomes

cubiomes 作为外部依赖，不混入业务代码。克隆到 `third_party/cubiomes`：

```bash
./scripts/setup_cubiomes.sh
```

Python 侧通过 `cubiomes_bindings` 调用（绑定待实现）。

## 数据目录

| 路径 | 用途 |
|------|------|
| `data/raw/` | 原始输入 |
| `data/interim/` | 环节 1 中间结果、环节 2 出生点与索引 |
| `data/screenshots/` | 环节 2 原始截图（`{seed}.png`） |
| `data/screenshots_masked/` | 环节 2 去 HUD 截图（训练用） |
| `data/labels/` | 环节 3 标签（`pilot_labels.csv`） |
| `outputs/` | 各环节运行输出与统计 |

## 训练与推理

训练（示例）：

```bash
PYTHONPATH=src .venv/bin/python -m seed_preview_cv.training.train --config configs/training.yaml
```

推理（示例）：

```bash
PYTHONPATH=src .venv/bin/python -m seed_preview_cv.inference.predict \
  --checkpoint outputs/models/scratch_cnn_v1/best.pt \
  --input-csv data/labels/splits/test_natural.csv \
  --output outputs/predictions/predictions.csv \
  --preprocess-mode prepared
```

**Preprocess mode 说明：**

| Mode | 适用输入 |
|------|----------|
| `prepared`（默认） | 已完成 mask + resize 的数据集图片（如 `data/dataset_images/`） |
| `raw` | 原始 World Preview 截图（如 2560×1600）；会走与数据集构建相同的 mask + resize 流程 |

对原始截图部署时**必须**使用 `--preprocess-mode raw`；默认 `prepared` 不会 mask HUD。

中断后恢复训练（从 `last.pt` 继续）：

```bash
PYTHONPATH=src .venv/bin/python -m seed_preview_cv.training.train \
  --config configs/training.yaml \
  --batch-size 32 \
  --run-dir outputs/models/scratch_cnn_v1 \
  --resume outputs/models/scratch_cnn_v1/last.pt
```

Resume 会恢复 model、optimizer、best_val_loss、epoch，并 append `train_history.json`。

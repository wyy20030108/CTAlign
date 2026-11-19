# 胸部 CT 前期实验源码

本目录归档从全卷基线、同卷文本交换和共享候选框到指定所见独立评价的历史程序。[`source-index.json`](source-index.json) 列出 12 个本地复现包及包内源文件对应的仓库路径和 SHA-256。每个包都可以按原有相对目录恢复源码。相同字节的程序共用已有路径。最早试跑的 `audit_pairs.py` 保留历史版本，文档字符串中的示例文件名采用通用格式，索引同时记录归档和仓库哈希。

| 用途 | 索引中的试验名 |
|---|---|
| 初始试跑、积液候选和局部残差探索 | `smoke`、`effusion_initial`、`effusion_residual` |
| 同卷文本交换、给定共享候选范围内的对照 | `text_swap`、`shared_boxes` |
| 早期指定所见半监督和定位误差诊断 | `finding_ssl`、`finding_localization` |
| 肺不张全卷基线、滑窗与初代定位精分割 | `atelectasis_full_volume`、`atelectasis_sliding`、`atelectasis_refine` |
| 原两掩膜预算的软肺叶定位改版及独立评价 | `atelectasis_bridge`、`atelectasis_holdout` |

这些程序对应不同开发阶段和队列，包括阴性对照与停止扩展的探索性路线。肺结节主要结果及补证分别见根目录的 [`experiments`](../../experiments) 和 [`nodule_followup`](../nodule_followup)。PRCV、DASFAA 各自的原任务源码见 [DiffUnder](https://github.com/wyy20030108/DiffUnder) 和 [SliceBudget](https://github.com/wyy20030108/SliceBudget)。

## 校验和恢复

从仓库根目录用 Python 3.12 运行以下命令。第二条比对本地原始复现包及其源文件成员。第三条在指定的新目录恢复所选试验在复现包中的源码结构。

```bash
python studies/earlier_ct/verify_sources.py
python studies/earlier_ct/verify_sources.py --archives /path/to/local-archives
python studies/earlier_ct/verify_sources.py --study atelectasis_bridge --destination /path/to/new-bridge-source
```

历史训练、既有权重的逐体素回读及源数据筛选按照各脚本的相对路径和参数运行。研究者从授权环境恢复各阶段清单、输入数组、外部模型和缓存。历史包中的运行脚本反映当时的工作目录和软件环境。获准下载的外部数据与模型来源见[复现说明](../../docs/reproduction.md)。

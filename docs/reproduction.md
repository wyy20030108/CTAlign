# 复现说明

## 运行环境

原训练和体素级复核入口面向 Linux 和 CUDA。以下环境使用 Python 3.12，以及初期 GPU 记录中的 PyTorch 2.8.0 和 CUDA 12.8 作为安装起点。CUDA 驱动须与所装 PyTorch wheel 兼容。只核对模型评分表时，安装 NumPy 即可。[PyTorch 官方安装说明](https://pytorch.org/get-started/previous-versions/)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip check
```

运行下列原始脚本前进入 `experiments/`。路径相对当前目录，权重和数据完整性检查直接使用文件字节。涉及候选下载进程的历史编排还需要 Linux 的 `/proc`。早期实验中调用 `nnunet.inference` 的冻结轮廓先验需要另装原版 `nnunet`，主肺结节流程使用 MONAI 定位器和 `lungmask`。

## 数据、外部权重与清单

1. 用已获相应数据集访问许可的 Hugging Face 账号获得 CT-RATE 和 ReXGroundingCT 资源。脚本固定了作为清单和报告输入的发布版本。原始影像及掩膜获取入口读取本机 Hugging Face 登录状态。 [CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE) · [ReXGroundingCT](https://huggingface.co/datasets/rajpurkarlab/ReXGroundingCT)
2. 准备外部 MONAI LUNA16 定位器 `lung_nodule_ct_detection` 0.6.10 的 `models/model.pt`，在当前 Linux 用户可访问的 `/tmp/joint_external_nodule/model.pt` 就位。`joint_external_nodule.py` 会核验其 SHA-256。使用 `lungmask` 作者发布的 LTRCLobes 权重 `unet_ltrclobes-3a07043d.pth`，放入 `/tmp/thesis_lobes_weights/`。共同的候选和解剖模型属于外部监督来源。 [MONAI 模型卡](https://huggingface.co/MONAI/lung_nodule_ct_detection/blob/0.6.10/docs/README.md) · [lungmask 原实现](https://github.com/JoHof/lungmask)
3. 精确重放使用原研究保留的私有患者清单、相应数组、检测和肺叶缓存，并保持锁文件引用的数据路径可解析。相对路径以 `experiments/` 为起点。授权取得的数据可按脚本重新生成或从私有实验备份恢复。用不同清单重训属于新一次运行，结果应另行记录。

```bash
cd experiments
hf auth login
python fetch_manifest.py --destination metadata --download
```

核心输入位置与结构如下。

| 输入 | 原代码期望 |
|---|---|
| `phase9_joint/nodule_v2/cohort.private.json` | `train`、`dev`、`test`、`unlabelled` 四个患者组以及 `labels["20"]` 里的 7 份可见掩膜。匹配条目含 `code`、`patient`、`query`、`path` 和 `sha`，有标条目包含 `coords` 等质控字段。 |
| 清单 `path` 指向的 `.npz` | 三维 `ct`、对应 RAS `affine`，重采样网格为 1.5 mm。有标条目带同网格 `mask`；无掩膜条目只含 `ct` 与 `affine`。 |
| `phase9_joint/nodule_replication/` | 固定模型复验的 `replication_manifest.private.json`、候选及几何收据、`design.json`、缓存收据和模型锁。重建所用的暂存数组在 `/tmp/joint_nodule_replication_data/arrays/`。 |
| 外部检测和解剖缓存 | `phase9_joint/nodule_v2/external_luna16/` 存冻结定位候选，`phase8_thesis/lobe_anatomy/` 存肺叶缓存。各病例缓存校验源数组或模型的哈希。 |

`joint_nodule_cohort.py` 支持 `screen` 和四个队列角色的 `fetch`，基于发布清单、同次检查报告以及前期实验的患者排除记录生成候选，`joint_nodule_run.py build` 生成病例角色和 7 份可见掩膜。按相同原则 `joint_replication_cohort.py` 提供复验的 `screen`、`fetch`、`seal`。这些入口使用历史 `phase8_thesis/thesis_private.json` 等私有排除清单实现研究内部患者隔离。复验归档的纳入计数为 112 名候选，几何质控后 99 名。 [ReXGroundingCT 任务和发布分组](https://rexrank.ai/ReXGroundingCT/)

## 重放冻结模型的两次评价

本地联合增益复现包中的 `nodule_v2/` 与 `nodule_replication/` 对应 `experiments/phase9_joint/` 的同名子目录。使用其中的 27 份权重、开发收据和冻结锁，与受保护的原始清单、数组以及对应候选/解剖缓存一起使用。在新的私有运行副本中写入评分；若文件夹已经包含逐例预测与评分，冻结脚本会校验现存文件，故应在空的评分输出位置做重新推理。

```bash
# 从 experiments/ 执行。原版首评锁由开发阶段生成，再在测试病例上做冻结预测。
ROOT=phase9_joint/nodule_v2/external_matched
python joint_confirm.py eval --root "$ROOT" --arm anatomy
for seed in 891 1901 2707; do
  for arm in AS AN AU AP BS BN BU BP; do
    python joint_confirm.py eval --root "$ROOT" --arm "$arm" --seed "$seed"
  done
  python joint_proto_raw_audit.py eval --seed "$seed"
done
python joint_confirm.py release --root "$ROOT"
python joint_proto_raw_audit.py release

# 固定模型复验，已有几何收据、候选和缓存时执行。
python joint_replication_eval.py score --shards 1
python joint_replication_eval.py release
```

复验尚未生成缓存时，在合格候选与下载收据就位后执行 `python joint_replication_eval.py cache --shards 1`，在首次评分前执行 `python joint_replication_eval.py lock`。原版复验先平均每个患者三个种子，再对患者等权汇总。零重合衡量预测与指定发布参考的交集是否为空。参考轮廓外体积是相对于指定所见发布掩膜的体积。冻结评分的 `release` 入口会从位打包的预测掩膜重新计分，核对逐例结果再生成队列汇总。

## 重新训练外部定位条件下的模型

以下命令从已准备的肺结节开发协议开始。先按上述路径准备数据、私有清单和外部权重。`A` 使用纯 CT，`B` 使用同次检查报告。`S` 为监督，`N` 为常规 EMA，`U` 为加 Adaptive UOT，`P` 为完整框架。`BR` 由原型对照和冻结原始局部读出共同产生。`joint_nodule_run.py warm-lock` 用训练侧标签拟合尺寸修正，需先于使用其校准的外部候选模型执行。

```bash
python joint_nodule_run.py anatomy
python joint_nodule_run.py warm-lock
python joint_external_nodule.py --role dev
python joint_external_nodule.py --role unlabelled
python joint_external_refine.py lock
for kind in A B; do
  python joint_external_refine.py unlabelled --kind "$kind"
  for seed in 891 1901 2707; do
    python joint_external_refine.py warm --kind "$kind" --seed "$seed"
    for arm in S N U P; do
      python joint_external_refine.py train --kind "$kind" --seed "$seed" --arm "$arm"
    done
  done
done
python joint_metrics.py phase9_joint/nodule_v2/external_matched
```

原型对照的训练入口为 `joint_external_proto_only.py lock` 与 `train --seed`，在原始读出下另跑 `joint_proto_raw_audit.py dev --seed`。开发选择规则在 `joint_nodule_selection.py`，冻结评价及阈值在 `joint_confirm.py` 和 `joint_proto_raw_audit.py` 中落实。对新划分作验证时，开发筛选和锁定先于留出评分；保留 22 人首评和 99 人冻结复验各自的时间点和证据范围。

## 前期实验入口

| 对照 | 源文件 |
|---|---|
| 同卷文本交换、真实报告来源匹配 | `swap_clinical.py`、`swap_eval.py`、`swap_results.py` |
| 同一共享候选框、跨患者结果与标注敏感性 | `matched6_pipeline.py`、`matched6.py`、`matched6_sensitivity.py` |
| 肺不张图文监督低分辨率、滑窗、定位加精分割 | `atelectasis_b0.py`、`atelectasis_sliding.py`、`atelectasis_locate_refine.py` |
| 肺不张图文监督与常规半监督、UOT | `thesis_test_pipeline.py`、`thesis_lobe_model.py`、`thesis_soft_transport.py`、`thesis_evaluate.py` |
| 肺不张局部条件 UOT 和原型约束 | `joint_factorial.py`、`joint_target_audit.py`、`joint_confirm.py` |

历史复现包对应的额外脚本、版本和恢复方法见[胸部 CT 前期实验源码](../studies/earlier_ct/)。新增 65 名肺结节留出的独立性、锁定和逐例算术核对入口见[终稿补证说明](../studies/nodule_followup/)。

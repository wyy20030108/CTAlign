# CTAlign

CTAlign 保存《面向少标注医学影像的多模态半监督分割研究》的胸部 CT 与临床报告实验代码。主任务用同一患者同次检查的报告指明肺结节，在固定 7 份目标域训练掩膜的条件下，比较图文输入、EMA 半监督、Adaptive UOT 和特征原型约束。代码也覆盖前期同卷文本交换、共享候选框和局灶肺不张对照。

## 已核验结果

| 评价队列 | 图文常规半监督 Dice | 图文＋完整框架 Dice | 患者配对差及 95% 区间 |
|---|---:|---:|---:|
| 官方验证分组首评，22 人 | 25.55% | 29.56% | +4.00 个百分点 [−0.67, +9.52] |
| 同源未用患者固定模型复验，99 人 | 19.42% | 22.63% | +3.20 个百分点 [+1.23, +5.23] |

99 人复验中，体素召回由 22.96% 提高至 29.26%，预测与指定参考零重合的比例由 61.62% 降至 56.23%，参考轮廓外预测体积均值由 0.117 增至 0.161 mL。两次评价分别汇总。99 人来自同一数据发布来源的未用患者。统计交互的区间跨零。图文 UOT 还包含在报告有唯一可用尺寸时的条件先验。

- [`experiments/`](experiments/) 保留训练、定位、各组消融、数据筛选、验证和逐体素回读代码。
- [`results/nodule.json`](results/nodule.json) 包含两队列各九组聚合指标和患者配对区间。
- [`results/source-provenance.json`](results/source-provenance.json) 记录初始实验代码出处及 SHA-256。历史实验以 `nodule-v1` 标记，进一步验证的实验可在本仓库记录各自的配置、代码版本和汇总证据。
- [`docs/reproduction.md`](docs/reproduction.md) 说明授权数据准备、冻结模型复跑和开发训练的步骤。
- [`studies/nodule_followup/`](studies/nodule_followup/) 提供终稿补证的文本、尺寸、操作点分析及新增同源留出验证。

## 核对发布结果

使用 Python 3.12 和 NumPy。`--archive` 输入本地原始复现包 `图文与Adaptive-UOT联合增益_复现.zip`。脚本检查代码哈希与编译，并用包内已聚合至患者的原始分数组重新计算两队列的均值、配对差、20,000 次患者 bootstrap 区间及差值为正的人数。

```bash
python -m pip install numpy
python tools/verify_results.py
python tools/verify_results.py --archive '/path/to/图文与Adaptive-UOT联合增益_复现.zip'
```

病例影像、病例报告、参考掩膜、逐例预测和模型权重由原授权数据来源和本地复现包管理。精确复跑原队列还依赖受保护的原始划分清单及缓存或其重建所需源数据。仓库附有这些运行输入的结构和来源说明。

算法承接 PRCV 2026 已录用工作的伪标签校准和原型约束。等预算及阈值独立确定的比较遵循 DASFAA 2027 已投稿工作中的评价原则。对应原研究代码分别见 [DiffUnder](https://github.com/wyy20030108/DiffUnder) 和 [SliceBudget](https://github.com/wyy20030108/SliceBudget)。

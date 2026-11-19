# 肺结节终稿补证

补证沿用 `nodule-v1` 的训练掩膜、无掩膜样本和外部定位器。`official` 和 `replication` 是此前已评分的 22 人和 99 人，回顾性结果分组报告。`fresh` 从同一发布来源新锁病例，排除既有训练、开发、评分及 99 人复验候选，先完成源匹配和几何筛选，再应用冻结模型和开发集确定的阈值。

新增 65 人上，以相同的 7 份人工掩膜训练，BN 和 BP 的 Dice 为 13.23% 和 16.71%。患者配对增益 +3.48 个百分点，95% 区间 [+1.67, +5.35]。使用完整短模板路线时，BP 较 BN 提高 3.49 个百分点。将 BP 阈值在开发集锁定为 0.85 后，新增组相对 BN 的召回差为 −0.95 个百分点，参考外体积差为 −0.051 mL。后者的召回区间覆盖超过预设 2 个百分点容许范围的差值。所有组别、敏感性对照及配对区间见[聚合结果](results.json)。

## 对照设计

| 问题 | 对照 |
|---|---|
| 文本编码器是否提供侧别及肺叶路由之外的信息 | 保留原报告生成的候选、域和解剖特征，只把网络文字输入换成同肺叶短模板、其他患者的同肺叶报告或空文字。 |
| 短模板是否可以直接用于推理 | 使用从原报告提取的肺叶和侧别构造模板，将模板同时用于候选路由和网络输入。对照权重和阈值冻结。 |
| Adaptive UOT 中个体尺寸的贡献 | 以原图文 UOT 和完整组分别对照关闭尺寸先验，以及在对应病例仍启用统一训练中位尺寸。三组匹配初始化、伪标签候选、随机种子、训练轮廓、无掩膜病例和阈值。 |
| 召回接近时是否能减少参考外预测 | 保持 BN 原阈值，从锁定网格中选使开发集 BP 平均召回最接近 BN 的阈值。报告新队列的实际召回差和配对区间。预设点估计的绝对召回差容许范围为 2 个百分点。 |

## 运行顺序

按[主复现说明](../../docs/reproduction.md)恢复训练、开发和既往评估的私有资料、权重和缓存。Python 3.12 和 CUDA 配置沿用主实验。以下命令使用旧实验目录作为工作目录。`$study` 指向本子目录的绝对路径。Linux 下用 `CUDA_VISIBLE_DEVICES` 分配训练任务和评分分片。

```bash
cd /path/to/CTAlign/experiments
study=$(realpath ../studies/nodule_followup)
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python "$study/ctalign_fresh.py" screen
python "$study/ctalign_size.py" lock
python "$study/ctalign_eval.py" register
python "$study/ctalign_structured.py" register
```

新队列用 `ctalign_fresh.py fetch --shard <i> --shards <n>` 下载。维护 `phase10_thesis/fresh/fetch_pids.json` 记录下载进程的 PID，也可在下载结束后写入空对象 `{}` 再启动缓存。`ctalign_fresh.py seal` 封存实际合格参考，`ctalign_eval.py cache` 准备相应的冻结检测器和肺叶缓存。原始数据与逐例文件可保存在研究计算环境中。

对 `off`、`constant`、3 个种子 `891 1901 2707` 与 `U P` 的每个组合，运行 `ctalign_size.py fit --mode <mode> --seed <seed> --arm <arm>`。用 `ctalign_eval.py score --cohort dev` 获取开发阈值曲线；同一入口支持 `official`、`replication`。待训练、几何和缓存收据齐备后运行下列命令。

```bash
python "$study/ctalign_eval.py" lock
python "$study/ctalign_eval.py" score --cohort fresh
python "$study/ctalign_structured.py" score --cohort fresh
python "$study/ctalign_summary.py" --roles dev official replication fresh --destination phase10_thesis/evaluation/followup_aggregate.json
python "$study/ctalign_structured.py" release --roles fresh
```

`ctalign_structured.py score` 与 `release` 也支持已评分的两队列。主要复算要求逐患者重建 22 人和 99 人的 BN、BP 原始指标，并用未舍入的配对差计算区间。补证的零重合与参考外体积均依据发布的指定所见轮廓。

三组结构化对照都评分后，运行 `ctalign_structured.py release --roles official replication fresh`。`ctalign_audit.py` 检查患者隔离、评分时间和体素计数的算术一致性。`ctalign_figures.py` 接收 `followup_aggregate.json` 生成两个群体层面的图。`ctalign_case_review.py` 从指定内部记录重放病例叠加，并核对其对应种子的四项分割指标。

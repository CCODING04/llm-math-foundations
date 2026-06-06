# LLM Math Foundations 改进待办清单

> 基于 `docs/综合改进意见.md`，按阶段跟踪完成情况。

---

## 第一阶段：事实性修复 ✅ (v0.2, commit `01e55ab`)

- [x] 1.1 修正全部 7 章的"下一章预告" — Ch1~Ch8 全部修正为实际下一章内容
- [x] 1.2 修正 Ch6 导航链接（文字+路径） — "概率分布"→"优化理论", "最优化"→"注意力机制", 路径全部修正
- [x] 1.3 修正主 README 的 4 处章节描述 — Ch1(去掉SVD), Ch4(改为特征值/SVD), Ch7(去掉位置编码), Ch8(改为温度/Top-K等)
- [x] 1.4 修正 resources 路径引用 — `README.md` → `further-reading.md`
- [x] 1.5 新增 `requirements.txt` — numpy/matplotlib/scipy/seaborn/torch
- [x] 1.6 修正 Ch3 脚本名不一致 — `distributions.py`→`gaussian_distribution.py`, `mle_demo.py`→`mle_cross_entropy.py`
- [x] 1.7 修正 Ch9 代码 typo — `dip=150`→`dpi=150`
- [x] 1.8 补齐 Ch8 缺失脚本 — 新增 `topk_topp_demo.py`, `sampling_strategies.py`

## 第二阶段：结构优化

- [ ] 2.1 新增 Ch0 数学预备
- [ ] 2.2 Ch9 拆分为 Ch9+Ch10
- [ ] 2.3 处理 Ch3/Ch8 MLE 重叠
- [ ] 2.4 Ch4 加桥接段落
- [ ] 2.5 更新主 README（目录结构、学习路线图）

## 第三阶段：教学体验

- [ ] 3.1 每章加前置知识 + 最小掌握标准
- [ ] 3.2 补全 Ch2/Ch3/Ch5/Ch8/Ch9 思考题答案
- [ ] 3.3 每章加"常见卡点"模块
- [ ] 3.4 Ch7 加完整数值 walkthrough + shape 总览表
- [ ] 3.5 标注核心/深入

## 第四阶段：阶段项目

- [ ] 4.1 项目 1：线性回归（含完整 starter code + 参考答案）
- [ ] 4.2 项目 2：两层神经网络
- [ ] 4.3 项目 3：Mini Attention + 采样器

---

## 完成记录

| 日期 | 阶段 | 完成项 | 提交 hash |
|------|------|--------|-----------|
| 2026-06-07 | 基线 | v0.1 原始快照 | `7369f5a` |
| 2026-06-07 | 一 | v0.2 事实性修复（8/8项） | `01e55ab` |

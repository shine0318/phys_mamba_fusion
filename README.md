README.md
# Phys-MambaFusion 代码实现文档

## 概述

本项目是论文 **"Synergistic Perception of Structural Failure Evolution via Mechanism-Gated Spatiotemporal Mamba"** 的 PyTorch 实现。

核心思想：将高分辨率 2D 表面纹理图像与动态数字图像相关（DIC）应变场融合，通过**力学门控**的方式过滤工业噪声，实现微缺陷检测 + 应力强度因子预测的联合输出。

---

## 目录结构

```
phys_mamba_fusion/
├── __init__.py
├── train.py                    # 训练入口
├── infer.py                    # 推理入口
├── models/
│   ├── backbone_2d.py          # 2D 视觉主干：MobileNetV3 + FPN
│   ├── dic_branch.py           # DIC 力学分支：物理预处理 + 各向异性编码 + 双注意力
│   ├── selective_ssm.py        # S6 选择性状态空间模型 + Bi-SSM
│   ├── cross_modal_gate.py     # 跨模态门控 Mamba
│   ├── ts_mamba.py             # TS-Mamba 完整模块
│   ├── output_head.py          # 三路输出头
│   └── phys_mamba_fusion.py    # 顶层模型集成
└── losses/
    └── physics_loss.py         # 物理约束复合损失函数
```

---

## 整体架构

论文图 1 对应四个阶段：

```
a) 输入
   ├── 2D 表面纹理图像  (B, 3, H, W)
   └── DIC 位移场序列  (B, T, 2, H_d, W_d)
         ↓ 时序同步 (TSS)
b) 特征提取
   ├── MobileNetV3 + FPN  →  Surface Texture Tokens T_2D
   └── DIC Branch         →  Mechanical Evolution Tokens T_DIC
         ↓
c) TS-Mamba 模块
   Modal Embedding → Bi-SSM (B,C=f(x)) → Cross-Modal Gated Mamba
         ↓  h_fused
d) 输出头
   ├── Bbox        (小目标检测)
   ├── Risk Score  (失效风险预测)
   └── K/J         (应力强度因子)
```

---

## 模块详解

### 1. 时序步骤同步 — `TemporalStepSync`

**文件**：`models/phys_mamba_fusion.py`

工业相机（30 fps）与 DIC 传感器（10 fps）采集频率不同，需对齐：

```
sync_idx = floor(t_vision × f_DIC / f_vision)
```

实现上对 DIC 序列沿时间轴做线性插值，使每一帧视觉图像对应一个时间一致的应变演化 token。

```python
x = F.interpolate(x, size=num_vision_frames, mode='linear', align_corners=False)
```

---

### 2. 2D 视觉主干 — `MobileNetV3FPN`

**文件**：`models/backbone_2d.py`

| 子模块 | 作用 |
|--------|------|
| MobileNetV3-Small | 提取多尺度表面纹理特征（stage1/2/3） |
| FPN Neck | 横向连接 + 上采样，融合三级特征图 |
| 空间线性化 | `AdaptiveAvgPool(7×7)` → flatten → `T_2D` tokens |

输出：`T_2D` shape 为 `(B, L_v, C_fpn)`，`L_v = 3 × 49 = 147`。

```python
# FPN 横向连接示意
for i in range(len(laterals) - 1, 0, -1):
    laterals[i-1] += F.interpolate(laterals[i], size=laterals[i-1].shape[-2:])
```

---

### 3. DIC 力学分支 — `DICBranch`

**文件**：`models/dic_branch.py`

#### 3.1 物理预处理 — `PhysicalPreprocessing`

将位移场 $u(x,y,t)$ 转换为应变场 $\varepsilon(x,y,t)$：

```
输入 u: (B, 2, H, W)
  ↓ 固定权重 Sobel 算子（x/y 方向梯度）
  ↓ 应变能密度 = Σ(∂u/∂x)² + Σ(∂u/∂y)²
  ↓ 叠加测量噪声通道
  ↓ 1×1 卷积投影 → 5 通道物理特征图
```

Sobel 核权重固定（`requires_grad=False`），保证物理梯度计算的确定性：

```python
sobel_kx = tensor([[-1,0,1],[-2,0,2],[-1,0,1]])  # 水平梯度
sobel_ky = tensor([[-1,-2,-1],[0,0,0],[1,2,1]])   # 垂直梯度
```

#### 3.2 各向异性编码 — `AnisotropicConv3D`

用三个独立 1D 卷积代替刚性 3D 卷积，捕获裂纹沿不同方向传播的应变集中区：

```
标准 3D 卷积：  K×K×K  — 各向同性，方向不敏感
各向异性分解：  (Kz,1,1) → (1,Ky,1) → (1,1,Kx)  — 可学习方向权重
```

#### 3.3 双注意力 — `DualAttention`

```
通道注意力：  GlobalAvgPool → FC → Sigmoid  → 筛选高物理意义通道
空间注意力：  [AvgPool, MaxPool] 拼接 → Conv7×7 → Sigmoid  → 定位高应变区域
```

#### 3.4 时空聚合输出

```
3D 卷积特征 → AdaptiveAvgPool3d(T→1) → DualAttention → 应变梯度嵌入
                                                       ↓
                                              T_DIC: (B, 1, d_model)
```

---

### 4. 选择性状态空间模型 — `SelectiveSSM` / `BiSSM`

**文件**：`models/selective_ssm.py`

#### S6 核心机制

与 Transformer 全局自注意力（$O(L^2)$）不同，SSM 以 $O(L)$ 复杂度建模长程依赖。

**关键特性**：矩阵 $B$、$C$ 是输入 $x$ 的函数（选择性），对 $\nabla\varepsilon$ 高的 token 分配更高权重：

$$B, C = f(x) \quad \Delta = \text{softplus}(W_\Delta \cdot x)$$

$$\bar{A} = e^{\Delta A}, \quad \bar{B} = \Delta B$$

$$h_t = \bar{A} h_{t-1} + \bar{B} u_t, \quad y_t = C h_t + D u_t$$

代码实现中 `A_log` 保证 $A < 0$（稳定性）：

```python
A = -torch.exp(self.A_log)          # A 恒负，保证状态衰减稳定
dt = F.softplus(self.dt_proj(dt))   # Δt > 0
B, C = f(x)                         # 输入依赖的选择性参数
```

#### Bi-SSM 双向扫描

```
正向扫描：  x[0] → x[1] → ... → x[L]   (捕获早期微位移跳变)
反向扫描：  x[L] → x[L-1] → ... → x[0] (向前追溯宏观裂纹成因)
融合：      merge(fwd, bwd) → LayerNorm
```

---

### 5. 跨模态门控 Mamba — `CrossModalGatedMamba`

**文件**：`models/cross_modal_gate.py`

这是整个框架最核心的创新模块，实现**力学控制律**对视觉特征的调制：

#### 门控逻辑

```
视觉 token T_2D ──┐
                  ├─→ Modal Embedding → [v; d] → Bi-SSM → h_visual, h_DIC
DIC token T_DIC ──┘

力学门控掩码：  G = Sigmoid(W · mean(h_DIC))   ∈ [0, 1]
融合输出：      h_fused = h_visual ⊙ G
```

**物理意义**：
- 当视觉异常与力学奇异点**空间重合**时，$G \approx 1$，信号被放大传播
- 当视觉异常仅为表面噪声（油污、加工纹路）时，力学隐状态 $h_{DIC} \approx 0$，门 $G \approx 0$，信号被抑制

> 论文验证：该机制抑制了 94% 以上的非结构性噪声，FPR < 1%

```python
G = self.gate_proj(dic_context)       # Sigmoid 门
h_fused = h_visual * G                # Hadamard 积过滤
h_fused = self.fusion_norm(h_fused + visual_tokens)  # 残差连接
```

---

### 6. TS-Mamba 完整模块 — `TSMamba`

**文件**：`models/ts_mamba.py`

```
T_2D (B, L_v, C_fpn)  →┐
                         ├→ SpatiotemporalTokenizer → d_model 对齐
T_DIC (B, 1, d_model)  →┘
         ↓
[CrossModalGatedMamba] × num_layers
         ↓
h_fused (B, L_v, d_model)  +  gate_maps  +  h_dic
```

`SpatiotemporalTokenizer` 用 `nn.Linear` 将两路 token 投影到统一 `d_model` 维度，并添加 LayerNorm。

---

### 7. 输出头 — `OutputHead`

**文件**：`models/output_head.py`

三路并行输出，对应论文图 1 的 d) 模块：

| 输出 | 类 | 说明 |
|------|-----|------|
| `bbox` | `BboxHead` | `(B, num_anchors, 5+cls)` 小目标边界框 |
| `risk_mu / risk_var` | `RiskScoreHead` | 失效风险均值 + 方差（贝叶斯不确定性） |
| `k_mu / k_var` | `StressFactorHead` | 应力强度因子 $K_I$ + 预测不确定性 |

**异方差不确定性量化**：模型同时预测均值和方差，SNR 低时置信区间自动展宽：

```python
mu  = Sigmoid(FC(feat))    # 风险/K_I 均值
var = Softplus(FC(feat))   # 方差（恒正）
```

---

### 8. 物理约束复合损失 — `PhysMambaFusionLoss`

**文件**：`losses/physics_loss.py`

$$\mathcal{L} = \mathcal{L}_{det} + \lambda_{risk} \cdot \mathcal{L}_{risk} + \lambda_{phys} \cdot \mathcal{L}_{phys}$$

| 损失项 | 实现 | 物理意义 |
|--------|------|---------|
| $\mathcal{L}_{det}$ | CIoU + Focal Loss | 微小目标精确定位，抑制类别不平衡 |
| $\mathcal{L}_{risk}$ | 异方差 NLL | 因果风险映射，区分良性铸造气孔与高危裂纹 |
| $\mathcal{L}_{phys}$ | 力学一致性 NLL | 强制满足 Saint-Venant 相容方程，惩罚无应力集中的"伪缺陷" |

**力学一致性损失**（$\mathcal{L}_{phys}$）的核心作用：若模型将一处加工纹路误判为缺陷，该位置不存在 $K_I$ 奇异性，则 $\mathcal{L}_{phys}$ 产生大梯度惩罚，迫使主干重新校准注意力到真实力学奇异点。

```python
# 带不确定性权重的 NLL
precision = 1.0 / (k_pred_var + 1e-6)
nll = 0.5 * (precision * (k_pred_mu - k_target)**2 + log(k_pred_var))
```

---

## 快速开始

### 环境依赖

```bash
pip install torch torchvision
```

### 推理示例

```bash
cd phys_mamba_fusion
python infer.py
```

输出示例：
```
=== Phys-MambaFusion Inference Demo ===
Risk Score:        0.5321 ± 0.7594
Stress Factor K_I: 0.6274 ± 0.8369
Bbox predictions:  torch.Size([1, 3, 6])
STATUS: Component within acceptable integrity bounds.
```

### 训练

```bash
cd phys_mamba_fusion
python train.py
```

将 `train.py` 中的 `DummyIndustrialDataset` 替换为真实数据集：
- `img`：缸体表面 2D 图像，shape `(3, 224, 224)`
- `dic_seq`：DIC 位移场序列，shape `(T, 2, H_d, W_d)`
- `targets`：`bbox`（归一化 cx/cy/w/h）、`cls`、`risk`（0~1）、`k_factor`（FEA $K_I$ 值）

### 加载检查点推理

```bash
python infer.py --checkpoint checkpoints/phys_mamba_fusion.pth --device cpu
```

---

## 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `d_model` | 256 | Token 特征维度 |
| `d_state` | 16 | SSM 隐状态维度 $N$ |
| `num_mamba_layers` | 2 | Cross-Modal Gated Mamba 层数 |
| `fpn_out_channels` | 256 | FPN 输出通道数 |
| `dic_base_ch` | 32 | DIC 分支基础通道数 |
| `f_vision` | 30 | 视觉相机帧率 (fps) |
| `f_dic` | 10 | DIC 传感器帧率 (fps) |
| `lambda_risk` | 1.0 | 风险损失权重 |
| `lambda_phys` | 2.0 | 物理约束损失权重（加倍强调力学一致性） |

---

## 与论文的对应关系

| 论文概念 | 代码位置 |
|---------|---------|
| Spatial feature Token flow | `backbone_2d.py` → `T_2D` |
| Mechanical evolution Token flow | `dic_branch.py` → `T_DIC` |
| TSS 时序同步 | `phys_mamba_fusion.py::TemporalStepSync` |
| Bi-SSM, $B,C=f(x)$ | `selective_ssm.py::SelectiveSSM.selective_scan` |
| Cross-Modal Gated Mamba, $G=\sigma(h_{DIC})$ | `cross_modal_gate.py::CrossModalGatedMamba` |
| Mechanical Gating Mask | `cross_modal_gate.py` → `G` 返回值 |
| $h_{fused}$ | `ts_mamba.py` → `h_fused` 输出 |
| Physics-informed loss $\mathcal{L}_{phys}$ | `losses/physics_loss.py::MechanicalConsistencyLoss` |
| Uncertainty Quantification (UQ) | `output_head.py::RiskScoreHead/StressFactorHead` |
| Anisotropic Encoding | `dic_branch.py::AnisotropicConv3D` |
| Dual Attention | `dic_branch.py::DualAttention` |

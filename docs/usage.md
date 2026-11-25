# 从轨迹到界面和频谱

在仓库根目录执行以下命令。Python 3.10 及以上版本适用。

```bash
python -m pip install -r requirements.txt
python examples/make_demo_dump.py output/demo.dump.gz
python -m interfextract output/demo.dump.gz --types 1 --methods hard gaussian neighbor --shape 24 24 40 --level 0.4 --reference-height 9 --radius 3.5 --branch continuous --output output/demo_analysis
python -m unittest discover -s tests -v
```

示例粒子是人工构造的波形液膜。运行真实 MD 时，将 `output/demo.dump.gz` 换成自己的文件，同时设置粒子类型、网格、密度阈值和参考高度。每次使用新的输出目录。

## 输入与参数

读取 LAMMPS `dump custom` 文本及其 `.gz` 压缩文件。盒子采用三向周期的正交边界。每帧需要 `type` 列，以及 `x y z`、`xu yu zu` 或 `xs ys zs` 中的一组坐标。粒子顺序可以随帧变化。

```text
ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0 12
0 12
0 20
ITEM: ATOMS id type x y z
1 1 1.0 1.0 4.0
2 2 1.0 1.0 0.0
```

例如 `--types 1` 只读取类型 1，类型 2 的壁面粒子被排除。聚合物可以通过 `--types 1 3 4` 选择多种流体珠子，邻居计数包含同链珠子。密度使用珠子数密度，单位随坐标选择。

| 参数 | 含义 |
|---|---|
| `--shape Nx Ny Nz` | 整个盒子的三维网格，数组顺序为 x、y、z |
| `--start A --stop B` | 包含两端的步号区间，未指定则使用全部帧 |
| `--level RHO` | 密度等值面的绝对阈值，例如已知体相密度乘以 0.1 |
| `--reference-height H` | 选取主界面附近交点的参考高度 |
| `--branch continuous` | 周期主液膜三角面分量及实际交点选择，默认方式 |
| `--branch nearest` | 每列选择最接近参考高度的实际向下交点 |
| `--gaussian-mode binned` | 先累计硬分箱密度，再进行周期离散高斯卷积，CLI 默认方式 |
| `--gaussian-mode direct` | 每个粒子直接向邻近网格点分配高斯权重 |
| `--width XI` | 高斯标准差，默认 `1/sqrt(20)`，与坐标单位一致 |
| `--radius R --neighbor-threshold RHO` | 邻居球半径及液体粒子分类阈值，默认 3.5 与 0.415 |
| `--weights trapezoid` | 按实际步号间隔作梯形积分，默认方式 |
| `--weights equal` | 各帧权重相同 |
| `--fill harmonic` | 显式要求对无交点列作周期调和重建，默认 `none` |
| `--max-cells N` | 密度网格的单数组格点上限，默认两千万，可按实际内存调整 |
| `--bins N` | 对数环带数量，默认 40 |

密度场的三个方向均使用周期边界。高度交点沿盒内 z 中心插值，调用者需将盒子的竖向切面放在合适的位置，确保所需自由表面位于盒内。当前 CLI 要求整个观察窗口使用同一盒子。

## 三种提取和平均顺序

硬分箱把每个流体珠子计入一个盒子，得到 `count / volume`。在所选时间窗内平均密度，沿 z 线性插值后取阈值交点。

高斯方法有两个可选实现。`gaussian_density()` 是逐粒子直接加权，每个截断核归一化为一个粒子。`gaussian_filter_density()` 是对已有数密度网格作周期卷积，使用 SciPy 的可分离离散核和 4ξ 截断。大场研究中采用过后一种流程。两种方式保留各自的粒子赋值响应，使用时应记录模式和网格。

邻居法在每个流体珠子的中心统计半径 R 内其他珠子的数量，除以球体积，超过阈值的珠子归为液体。每个横向列取最高液体珠子，再对逐帧高度作时间平均。任意一帧缺失的列在时均高度中保留 NaN。

CLI 输出的是**时均密度提取的高度谱**或**时均邻居高度的谱**。研究逐帧功率的集合均值时，应逐帧调用 `surface_from_density()` / `neighbour_surface()` 和 `power_spectrum()`，再平均功率。平均的次序会改变结果。

## FFT 与环向平均

```python
import numpy as np
from interfextract import power_spectrum, radial_spectrum

with np.load("output/demo_analysis/gaussian.npz") as result:
    height = result["height_fft"]
    box = result["box"]

power, qx, qy = power_spectrum(height, box)
radial = radial_spectrum(power, qx, qy, np.geomspace(0.6, 5.0, 30))
```

先减去高度均值，计算完整二维 FFT。

\[
\widehat h_{mn}=\frac{\operatorname{FFT2}(h-\bar h)_{mn}}{N_xN_y},
\qquad P_{mn}=|\widehat h_{mn}|^2,
\qquad \sum_{mn}P_{mn}=\operatorname{var}(h).
\]

波数采用 `2*pi*fftfreq`，单位为弧度/长度。在每个 \(q=\sqrt{q_x^2+q_y^2}\) 环带对实际模态功率作算术平均。`mode_count` 给出所用模态数，包含共轭模态。代码跳过零模和空环带，并将展示范围限制到两个轴中较小的 Nyquist 波数。窄长盒低波数可能仅有单一方向的模态。

环带横坐标采用 \(q_{\mathrm{effective}}=\langle q^{-2}\rangle^{-1/2}\)，因此 \(C/q^2\) 的离散环均恰好落在 \(C/q_{\mathrm{effective}}^2\) 上。CSV 保存每模平均功率，未乘环带模态数或 \(q^2\)。高度的单位为长度时，`power_mean` 的单位为长度平方。

按这里的归一化，白噪声毛细波的每模平衡参照为

\[
P(q)=\frac{k_BT}{\gamma L_xL_yq^2}.
\]

做面积补偿谱时显式乘 \(L_xL_y\)。表面张力需要由相应体系提供。模态数和同一轨迹的帧数分别记录，二者都不能充当独立 seed 数。

## 缺列与探索性补列

默认输出保留 `height_observed` 和 `missing_mask`。若某方法未通过完整界面检查，该方法的 NPZ 仍会保存，`summary.json` 记录 `fft_written=false`。

```bash
python -m interfextract output/demo.dump.gz --types 1 --methods hard gaussian --shape 24 24 40 --level 0.4 --reference-height 9 --branch nearest --fill harmonic --output output/demo_filled
```

`harmonic_fill()` 固定已观测的高度，只在缺列区域求解周期离散拉普拉斯方程。FFT 使用 `height_fft`，原高度和缺失掩膜同时保留。重建引入平滑延拓假设，会影响短波功率。所有列均缺失时，仅保存缺列信息。已有交点发生分支或周期检查问题而未产生缺列时，CLI 保留该失败状态。

高斯核自身也有观测响应。连续线性高斯卷积的功率响应为 \(\exp(-\xi^2q^2)\)，实际离散核还取决于网格和截断。密度阈值取面会进一步引入非线性。对同一原场比较核宽、补列前后可用区域及不同提取方法，才能解释谱下降的来源。

## 输出文件

| 文件 | 内容 |
|---|---|
| `summary.json` | 参数、实际观察步号、帧数、阈值/分支检查、FFT及补列状态、NPZ SHA256 |
| `hard.npz` 等 | 原高度、缺失掩膜、盒子，成功算谱时另含 FFT 高度、二维功率和波数 |
| `hard_radial.csv` 等 | 每环带边界、等效波数、每模平均功率及模态数 |

直接粒子高斯版本适合作为参考实现，运行时间随粒子数和核覆盖格点增长。CLI 的邻居网格沿用 `shape` 的前两个维度，需要不同的邻居分辨率时单独执行该方法。三维网格的双精度数组至少需要 `8*Nx*Ny*Nz` 字节，平均、分支提取和三角面处理会同时使用多个数组。

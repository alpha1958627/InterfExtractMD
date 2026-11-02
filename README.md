# InterfExtractMD

从分子动力学粒子坐标中提取液气界面，比较三种常用观察方式：**硬分箱、高斯密度、邻居密度**。

这三种方法先回答不同的问题：固定盒子里有多少粒子？空间某一点附近的密度如何变化？一个粒子周围是否足够密集？密度的定义不同，最后得到的界面也会不同。

## 三种方法

![三种界面提取方法示意](docs/assets/methods.png)

*上图是方法示意，不是MD采样结果。*

示意图绘制脚本为 `examples/make_schematic.py`，需要Pillow及中文字体；数值计算不依赖该绘图脚本。

| 方法 | 在哪里计算密度 | 怎样得到界面 | 适用场景 |
|---|---|---|---|
| **硬分箱** | 固定的三维小盒子 | 盒内粒子数除以盒体积，再取密度等值面 | 查看局部密度、粗尺度表面，检查分箱和采样的影响 |
| **高斯密度** | 固定空间网格点 | 将附近粒子的高斯权重叠加，再取密度等值面 | 观察连续界面，比较短波起伏和时间平均效果 |
| **邻居密度** | 每个真实粒子的中心 | 统计球形邻域中的粒子，先判别液体粒子，再取横向箱内最高的液体粒子 | 液体粒子识别、局部结构和粒子尺度表面 |

### 硬分箱：固定空间小盒子

将空间划成小盒子，同一粒子只贡献给其中心所在的盒子。盒子的局部数密度为

$$\rho_B(t)=N_B(t)/V_B.$$

需要时间平均时，先累计各帧的盒内密度，再提取界面。沿竖直方向对密度作分段线性插值，求取液体侧向气体侧的阈值交点。

盒子越小，不代表界面一定越准确。某列在整个观察窗口内没有粒子进入时，就没有正密度阈值的交点；代码保留缺失，不补高度。

### 高斯密度：在空间中叠加粒子的贡献

每个粒子对附近多个网格点贡献密度，权重随距离连续减小：

$$\rho_\xi(\mathbf r,t)=\sum_i W_\xi(\mathbf r-\mathbf r_i(t)),\qquad W_\xi\propto\exp[-|\mathbf r|^2/(2\xi^2)].$$

这里采用**直接粒子到网格的高斯权重**，每个截断核归一化后总贡献为一个粒子。它不是先作硬分箱、再对密度图调用模糊滤波。

高斯密度给出连续的密度描述，随后同样通过阈值交点提取界面。核宽是观察尺度，**不改变MD中的粒子运动，也不是OU–SE输入噪声的空间相关长度**。

### 邻居密度：以粒子为中心判别液体

以粒子 $i$ 为中心，在半径 $r_c$ 的球内计数：

$$\rho_i=\frac{\#\{j\ne i:|\mathbf r_j-\mathbf r_i|_{\mathrm{PBC}}\le r_c\}}{4\pi r_c^3/3}.$$

密度超过阈值的粒子标记为液体。再在每个横向箱内，选取最高的液体粒子坐标作为高度。**粒子分类之后，还需要这一步取面规则**；邻居密度本身并不直接给出连续空间等值面。

本实现只统计传入的流体粒子，排除中心自身，保留二聚体的键伙伴，使用三维周期最小镜像距离。图中的二维邻域圆只是三维球的示意。

## 同一截面，三种界面

![三种方法的同一截面对比](docs/assets/interface_comparison.png)

这张图来自同一条 **300σ × 300σ二聚体液膜**轨迹的同一瞬时帧、同一物理截面。灰点是薄层中粒子的投影；密度和邻居计数使用完整的三维粒子构型，并非只使用图中灰点。

- 黑线：硬分箱密度界面。
- 红线：高斯密度界面。
- 蓝线：邻居分类后，横向箱内最高液体粒子形成的界面。

三条曲线没有作竖向对齐。它们的平均位置和局部起伏都可能不同，因为阈值面与最高液体粒子不是同一个界面定义。图展示的是观察方法的差别，不用于给物理准确性排名。

[查看更宽的截面](docs/assets/interface_wide.png)

### 展示图的设置

| 项目 | 设置 |
|---|---|
| 硬分箱 | 盒体积 $1\sigma^3$，横向300 × 300 |
| 高斯密度 | $\xi=\sigma/\sqrt{20}$；横向间距0.25σ，竖向间距0.125σ |
| 两种密度等值面 | $\rho_c=\rho_{\mathrm{bulk}}/10$，沿实际线性交点取连续主液膜分支 |
| 邻居密度 | $r_c=3.5\sigma$；$\rho_i>0.415\sigma^{-3}$；横向箱1σ × 1σ |
| 截面 | 同一时刻、$y=150.5\sigma$；图中 $x=140$–$160\sigma$ |
| 时间处理 | **本截面图三种方法均为瞬时结果，没有混入1τ平均** |

## 运行参考实现

```bash
python -m pip install -r requirements.txt
python examples/synthetic_demo.py
python -m unittest discover -s tests -v
```

示例自动生成一小片**合成粒子液膜**，演示三种方法，并输出界面缺列数及功率归一化检查。它不是MD模拟，也不用于重现上面的私有MD截面图。

自己的数据可以按下面的方式传入：

```python
import numpy as np
from interfextract import (
    hard_bin_density, gaussian_density, surface_from_density,
    neighbour_surface, power_spectrum,
)

# positions: 只含流体粒子的(N, 3)笛卡尔坐标，使用自己的数组。
box = np.array([[0., 12.], [0., 12.], [0., 20.]])
shape = (24, 24, 40)

rho_hard = hard_bin_density(positions, box, shape)
rho_gauss = gaussian_density(positions, box, shape, width=0.4)

h_hard, check_hard = surface_from_density(
    rho_hard, box, level=0.4, reference_height=9., branch="continuous")
h_gauss, check_gauss = surface_from_density(
    rho_gauss, box, level=0.4, reference_height=9., branch="continuous")
h_neighbour, check_neighbour = neighbour_surface(
    positions, box, shape[:2], radius=3.5, density_threshold=0.415)

if check_gauss["valid"]:
    power, qx, qy = power_spectrum(h_gauss, box)
```

上例中的参数用于小域演示；实际分析时应按自己的密度、长度单位和分辨率设置。`box`的格式是 `[[xmin, xmax], [ymin, ymax], [zmin, zmax]]`，当前实现用于**三向周期的正交盒**。不直接支持倾斜盒，也不读取任意格式的LAMMPS文件。

### 时间平均在哪里做？

硬分箱和高斯方法可以先平均密度，再提取界面：

```python
from interfextract import average_density

# frames按时间顺序逐帧提供positions；201帧覆盖一个闭合观察区间。
weights = [0.5] + [1.0] * 199 + [0.5]
rho_mean = average_density(
    frames, box, shape, method="gaussian", weights=weights, width=0.4)
h_mean_density, check = surface_from_density(
    rho_mean, box, level=0.4, reference_height=9., branch="continuous")
```

这与“每帧提取高度后平均高度”以及“每帧算功率后平均功率”是三个不同的量。高斯密度在既有单轨迹的1τ对照中，短波功率相对瞬时结果降低约51%–59%；这是时间平均与界面提取共同产生的观察效果，不能全部归因于分子刚度。

## 界面与功率谱的检查

- 所有选中高度都必须是实际密度的线性阈值交点；不填缺列，不补高度。
- `branch="continuous"`沿用研究程序的周期主液膜分支选择，`branch="nearest"`只选距给定参考高度最近的实际向下交点。参考高度由调用者明确给出，不根据谱的下降程度挑选。
- 主液膜筛选是这里声明的重建选择，不冒充Willis–Freund未公开的作者代码。它也可能额外排除局部小液滴或分离分支，应与原始缺列分开查看。
- 检查返回的 `valid`。缺失高度保留为NaN，`power_spectrum`拒绝对不完整表面作普通FFT。
- 功率按二维FFT的每模幅值平方定义，减去各自平均高度，并检查Parseval关系。需要环均时应取每模平均，不把环带功率总和当作每模谱。
- Python高斯方法是清晰的直接求和参考实现，大粒子数下会慢。大网格必须先估算内存；超过默认 `max_cells` 时需由调用者确认资源后显式提高限制。

## 文件

```text
interfextract/            三种密度定义、界面提取、时间平均与二维功率
examples/synthetic_demo.py  无需私有数据的演示
tests/                   质量、周期边界、真实交点、缺列和FFT检查
docs/assets/             方法示意图与已授权展示的截面图
```

仓库只公开代码、说明和展示图，不包含原始MD轨迹、粒子坐标、密度场、高度场或逐点截面数据。

## 方法来源

- [Willis & Freund, *Physics of Fluids* **22**, 022002 (2010)](https://doi.org/10.1063/1.3326077)：分箱密度、时间平均与阈值界面。
- [Liu, Zhao, Lockerby & Sprittles, *Physical Review E* **107**, 015105 (2023)](https://doi.org/10.1103/PhysRevE.107.015105)：粒子邻居密度分类与最高液体粒子取面。
- 高斯密度与连续主液膜分支为本仓库明确给出的观察实现；三种方法的曲线来自同一粒子构型的对照。

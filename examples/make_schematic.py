"""Draw the conceptual diagram; requires Pillow and a CJK font, no MD data."""
from pathlib import Path
import argparse
from PIL import Image, ImageDraw, ImageFont
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--font", type=Path, help="Path to a Chinese-capable TTF, TTC or OTF")
args = parser.parse_args()
candidates = [args.font, Path("C:/Windows/Fonts/msyh.ttc"),
              Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")]
font = next((p for p in candidates if p and p.exists()), None)
if font is None:
    raise SystemExit("Supply a Chinese font with --font /path/to/font.ttf")

scale = 2
def F(size): return ImageFont.truetype(str(font), size * scale)
im = Image.new("RGB", (1200 * scale, 310 * scale), "white")
draw = ImageDraw.Draw(im)
blue, red, ink = "#1f77b4", "#d62728", "#172b3a"
def text(x, y, s, size=20, color=ink):
    draw.text((x * scale, y * scale), s, font=F(size), fill=color, anchor="mm")
def circle(x, y, r, color):
    draw.ellipse(((x-r)*scale, (y-r)*scale, (x+r)*scale, (y+r)*scale), fill=color)

for x, title in [(200, "硬分箱"), (600, "高斯密度"), (1000, "邻居密度")]:
    text(x, 35, title, 28)
for x in [105,143,181,219,257,295]:
    draw.line((x*scale,91*scale,x*scale,205*scale),fill="#c4c9cd",width=scale)
for y in [91,129,167,205]:
    draw.line((105*scale,y*scale,295*scale,y*scale),fill="#c4c9cd",width=scale)
draw.rectangle((181*scale,129*scale,219*scale,167*scale),outline=red,width=3*scale)
for x,y in [(125,187),(155,108),(194,143),(208,153),(240,116),(276,157)]:circle(x,y,3.5,blue)

# A drawn Gaussian cloud explains weighting; these pixels are not measured density.
xx,yy=np.meshgrid(np.arange(470*scale,730*scale)/scale,np.arange(70*scale,220*scale)/scale)
alpha=np.zeros(xx.shape)
for x,y in [(548,156),(603,125),(651,166)]:
    alpha=1-(1-alpha)*(1-.50*np.exp(-((xx-x)**2+(yy-y)**2)/(2*22**2)))
rgb=255*(1-alpha[:,:,None])+np.array([31,119,180])*alpha[:,:,None]
im.paste(Image.fromarray(np.round(rgb).astype(np.uint8)),(470*scale,70*scale))
draw=ImageDraw.Draw(im)
for x,y in [(548,156),(603,125),(651,166)]:circle(x,y,3.5,blue)
draw.line((590*scale,153*scale,610*scale,153*scale),fill=red,width=2*scale)
draw.line((600*scale,143*scale,600*scale,163*scale),fill=red,width=2*scale)
for start in range(0,360,16):
    draw.arc((940*scale,89*scale,1054*scale,203*scale),start,start+9,fill=red,width=2*scale)
circle(997,146,5,red)
for x,y in [(960,133),(979,108),(1020,125),(1024,171),(980,182),(1069,195),(1087,118),(931,199)]:circle(x,y,3.5,blue)
for x,label,sub in [(200,"固定空间小盒子","盒内粒子数 ÷ 盒体积"),
                    (600,"粒子贡献按距离加权叠加","在网格点上构造连续密度"),
                    (1000,"以粒子为中心的邻域","球内邻居数 ÷ 球体积")]:
    text(x,255,label,20);text(x,287,sub,17,"#555555")
target=Path(__file__).resolve().parents[1]/"docs/assets/methods.png"
im.save(target,optimize=True)
print(target)

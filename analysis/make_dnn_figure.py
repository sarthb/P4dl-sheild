import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrow
from matplotlib import font_manager as fm
for f in fm.findSystemFonts():
    if "Carlito" in f: fm.fontManager.addfont(f)
plt.rcParams["font.family"] = "Carlito"
plt.rcParams["font.weight"] = "bold"

W, H = 13.3, 10.1                      # same aspect as the original (1332 x 1010)
fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off"); ax.set_aspect("equal")

BLUE, YEL, PUR, TEAL, GRN = "#4472c4", "#ffd966", "#8e7cc3", "#76c7c0", "#70ad47"
TBLUE, TYEL, TPUR, TTEAL, TGRN = "#1f4fa8", "#8a6400", "#4a2f8a", "#1d6b65", "#3b6e22"
EDGE = "#000000"; LINE = "#1a1a1a"; R = 0.21

def column(x, ys, fc):
    for y in ys:
        ax.add_patch(Circle((x, y), R, fc=fc, ec=EDGE, lw=1.4, zorder=3))

def ys_with_gap(top, bottom, n_top, n_bot, step):
    a = [top - i * step for i in range(n_top)]
    b = [bottom + i * step for i in range(n_bot)][::-1]
    return a, b

xi, x1, x2, x3, xo = 3.75, 6.1, 8.25, 10.15, 11.45
# input: all 9 features drawn
yi = [7.9 - i * 0.62 for i in range(9)]
# hidden layers: drawn with a gap, as in the original
a1, b1 = ys_with_gap(8.5, 1.75, 7, 3, 0.6)
a2, b2 = ys_with_gap(7.95, 2.55, 5, 3, 0.6)
a3, b3 = ys_with_gap(7.35, 3.2, 3, 2, 0.6)
h1, h2, h3 = a1 + b1, a2 + b2, a3 + b3
yo = 4.95

def connect(xa, ya, xb, yb, lw=1.05):
    for p in ya:
        for q in yb:
            ax.plot([xa + R, xb - R], [p, q], color=LINE, lw=lw, zorder=1)
connect(xi, yi, x1, h1); connect(x1, h1, x2, h2); connect(x2, h2, x3, h3)
connect(x3, h3, xo, [yo], lw=1.3)

column(xi, yi, BLUE); column(x1, h1, YEL); column(x2, h2, PUR); column(x3, h3, TEAL); column(xo, [yo], GRN)

# gaps: vertical dots
for x, ya, yb in [(x1, a1, b1), (x2, a2, b2), (x3, a3, b3)]:
    yc = (ya[-1] + yb[0]) / 2
    for d in (-0.18, 0, 0.18):
        ax.add_patch(Circle((x, yc + d), 0.045, fc="#000000", ec="none", zorder=3))

# titles
def title(x, y, t, sub, col):
    ax.text(x, y, t, ha="center", va="bottom", fontsize=18.5, color=col, fontweight="bold")
    ax.text(x, y - 0.12, sub, ha="center", va="top", fontsize=15, color="black")
title(xi, 9.35, "Input Layer", "(9 Features)", TBLUE)
title(x1, 9.35, "Hidden Layer 1", "(128 units)", TYEL)
title(x2, 9.35, "Hidden Layer 2", "(64 units)", TPUR)
title(x3, 9.35, "Hidden Layer 3", "(32 units)", TTEAL)
title(xo + 0.55, 7.75, "Output Layer", "(1 unit)", TGRN)

# feature names next to the input nodes
names = ["Protocol", "Tag", "Packet length", "Flow packet count", "Flow byte count",
         "Flow duration", "Flow packets/s", "Inter-arrival time", "Window packet count"]
for n, y in zip(names, yi):
    ax.text(xi - R - 0.12, y, n, ha="right", va="center", fontsize=13.5, color="black")

# left brace and input arrow
bx = 1.12
ax.plot([bx + 0.12, bx, bx, bx - 0.1, bx, bx, bx + 0.12],
        [yi[0] + 0.3, yi[0] + 0.2, (yi[0] + yi[-1]) / 2 + 0.1, (yi[0] + yi[-1]) / 2,
         (yi[0] + yi[-1]) / 2 - 0.1, yi[-1] - 0.2, yi[-1] - 0.3], color="#000000", lw=1.6)
ax.add_patch(FancyArrow(0.12, (yi[0] + yi[-1]) / 2, 0.78, 0, width=0.07, head_width=0.26, head_length=0.12,
                        fc=BLUE, ec="#1f3864", lw=1.2, length_includes_head=True))
ax.text(0.5, (yi[0] + yi[-1]) / 2 + 0.22, "Input", ha="center", va="bottom", fontsize=17, fontweight="bold")

# output arrow
ax.add_patch(FancyArrow(xo + R + 0.2, yo, 1.15, 0, width=0.07, head_width=0.26, head_length=0.17,
                        fc=GRN, ec="#375623", lw=1.2, length_includes_head=True))
ax.text(xo + R + 0.78, yo + 0.2, "Output", ha="center", va="bottom", fontsize=19, fontweight="bold")
ax.text(xo + R + 0.78, yo - 0.2, "p(attack)", ha="center", va="top", fontsize=14)

# bottom braces + dashed boxes
def bottom(x, w, text, col):
    yb = 1.05
    ax.plot([x - w / 2, x - w / 2 + 0.05, x - 0.08, x, x + 0.08, x + w / 2 - 0.05, x + w / 2],
            [yb + 0.12, yb, yb, yb - 0.1, yb, yb, yb + 0.12], color="#000000", lw=1.5)
    box_w = w + 0.22
    ax.add_patch(FancyBboxPatch((x - box_w / 2, 0.08), box_w, 0.72, boxstyle="square,pad=0",
                                fc="white", ec=col, lw=1.8, ls=(0, (4, 3))))
    ax.text(x, 0.44, text, ha="center", va="center", fontsize=12.5, linespacing=1.25, color="black")
bottom(xi, 1.5, "Input\n(9 Features)", "#2f528f")
bottom(x1, 1.62, "Fully Connected (128)\n+ BatchNorm + ReLU\n+ Dropout (0.3)", "#b38f00")
bottom(x2, 1.62, "Fully Connected (64)\n+ BatchNorm + ReLU\n+ Dropout (0.3)", "#5b4a9e")
bottom(x3, 1.5, "Fully Connected (32)\n+ ReLU", "#2a8c85")
bottom(xo + 0.62, 1.5, "Fully Connected (1)\n+ Sigmoid\n(attack if p ≥ 0.7)", "#4e8a2e")

fig.savefig("DNN.png", dpi=300, facecolor="white")
fig.savefig("DNN.pdf", facecolor="white")

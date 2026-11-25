import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon, FancyArrowPatch
from matplotlib import font_manager as fm
for f in fm.findSystemFonts():
    if "Carlito" in f: fm.fontManager.addfont(f)
plt.rcParams["font.family"] = "Carlito"

W, H = 12.0, 6.6
fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off"); ax.set_aspect("equal")

SW = "#1f6fa8"; SWL = "#2f8fd0"; INK = "#000000"; MUTED = "#000000"
ATT, ATT_BG = "#c62828", "#fdecea"      # attacker in every scenario
ATT2, ATT2_BG = "#b85c00", "#fff3e0"    # additional attackers (S2) / flood (S9)
VIC, VIC_BG = "#1b5e20", "#e8f5e9"      # victim
BEN, BEN_BG = "#222222", "#f3f4f6"      # benign

def arrow(p, q, lw=1.8, color="#000000", both=True, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="<|-|>" if both else "-|>", mutation_scale=11,
                                 lw=lw, color=color, linestyle=ls, shrinkA=0, shrinkB=0, zorder=2))

def switch(x, y, name, subnet):
    w, h = 1.05, 0.42
    top = Polygon([(x - w/2, y + h/2), (x - w/2 + 0.12, y + h/2 + 0.14), (x + w/2 + 0.12, y + h/2 + 0.14), (x + w/2, y + h/2)],
                  closed=True, fc=SWL, ec=SW, lw=0.8, zorder=3)
    side = Polygon([(x + w/2, y - h/2), (x + w/2 + 0.12, y - h/2 + 0.14), (x + w/2 + 0.12, y + h/2 + 0.14), (x + w/2, y + h/2)],
                   closed=True, fc="#17577f", ec=SW, lw=0.8, zorder=3)
    ax.add_patch(side); ax.add_patch(top)
    ax.add_patch(Rectangle((x - w/2, y - h/2), w, h, fc=SW, ec="#164e75", lw=0.8, zorder=3))
    for dy in (0.07, -0.07):
        ax.annotate("", xy=(x + 0.28, y + dy), xytext=(x - 0.28, y + dy),
                    arrowprops=dict(arrowstyle="<|-|>" if dy > 0 else "<|-|>", color="white", lw=1.0, mutation_scale=7), zorder=4)
    left = x < 6.0
    lx = x - 0.12 if left else x + 0.12
    ha = "right" if left else "left"
    ax.text(lx, y + 0.62, name, ha=ha, va="bottom", fontsize=16.2, color=INK)
    ax.text(lx, y + 0.36, subnet, ha=ha, va="bottom", fontsize=13.8, color=MUTED)

def host(x, y, name, ip, role, col, bg, tag=None):
    # computer icon
    ax.add_patch(Rectangle((x - 0.33, y - 0.05), 0.46, 0.34, fc="white", ec=INK, lw=1.0, zorder=3))
    ax.add_patch(Rectangle((x - 0.29, y - 0.01), 0.38, 0.26, fc="#dfe7ef", ec="none", zorder=3))
    ax.add_patch(Rectangle((x - 0.13, y - 0.12), 0.06, 0.07, fc=INK, ec="none", zorder=3))
    ax.add_patch(Rectangle((x - 0.24, y - 0.15), 0.28, 0.04, fc=INK, ec="none", zorder=3))
    ax.add_patch(Rectangle((x + 0.17, y - 0.15), 0.16, 0.44, fc="white", ec=INK, lw=1.0, zorder=3))
    for k in range(3):
        ax.plot([x + 0.2, x + 0.3], [y + 0.2 - k * 0.06, y + 0.2 - k * 0.06], color=INK, lw=0.6, zorder=4)
    # label card
    cw, ch = 1.38, 1.18
    ax.add_patch(FancyBboxPatch((x - cw/2, y - 0.3 - ch), cw, ch, boxstyle="round,pad=0,rounding_size=0.07",
                                fc=bg, ec=col, lw=1.3, zorder=2))
    ax.text(x, y - 0.42, name, ha="center", va="top", fontsize=16.2, color=INK, fontweight="bold")
    ax.text(x, y - 0.74, ip, ha="center", va="top", fontsize=13.1, color=MUTED)
    ax.text(x, y - 1.01, role, ha="center", va="top", fontsize=13.1, color=("#000000" if col == BEN else col), linespacing=1.1)
    if tag:
        ax.text(x + cw/2 - 0.06, y - 0.3 - 0.06, tag, ha="right", va="top", fontsize=10.6, color="white",
                bbox=dict(boxstyle="round,pad=0.18", fc=col, ec="none"), zorder=5)

# controller
cx, cy = 6.0, 5.75
ax.add_patch(FancyBboxPatch((cx - 0.42, cy - 0.32), 0.84, 0.64, boxstyle="round,pad=0,rounding_size=0.06",
                            fc=SW, ec="#164e75", lw=0.8, zorder=3))
ax.add_patch(plt.Circle((cx, cy + 0.05), 0.13, fc="white", ec="none", zorder=4))
for a in range(8):
    import math
    t = a * math.pi / 4
    ax.plot([cx + 0.17 * math.cos(t), cx + 0.27 * math.cos(t)], [cy + 0.05 + 0.17 * math.sin(t), cy + 0.05 + 0.17 * math.sin(t) * 0.9],
            color="white", lw=1.0, zorder=4)
ax.text(cx, cy + 0.42, "P4Runtime controller", ha="center", va="bottom", fontsize=17.5, color=INK)
ax.text(cx + 0.55, cy - 0.05, "classifier · blocking policy\nflow-churn guard", ha="left", va="center", fontsize=13.1, color=MUTED, linespacing=1.15)

# switches
sy = 3.75
HX = [0.85 + i * 1.475 for i in range(8)]
xs = [(HX[2*i] + HX[2*i+1]) / 2 for i in range(4)]
for i, x in enumerate(xs, 1):
    switch(x, sy, f"s{i}", f"10.0.{i}.0/24")
    arrow((cx + (x - cx) * 0.07, cy - 0.33), (x + (0.18 if x < 6 else -0.18), sy + 0.36), lw=1.6, color="#000000")

# inter-switch links
arrow((xs[0] + 0.62, sy), (xs[1] - 0.55, sy))
arrow((xs[2] + 0.62, sy), (xs[3] - 0.55, sy))
ax.text((xs[0] + xs[1]) / 2, sy - 0.2, "100 Mb/s, 2 ms", ha="center", va="top", fontsize=13.1, color=MUTED)
ax.text((xs[2] + xs[3]) / 2, sy - 0.2, "100 Mb/s, 2 ms", ha="center", va="top", fontsize=13.1, color=MUTED)
arrow((xs[1] + 0.62, sy), (xs[2] - 0.55, sy), lw=3.6, color=ATT)
ax.text((xs[1] + xs[2]) / 2, sy + 0.12, "bottleneck", ha="center", va="bottom", fontsize=15.0, color=ATT, fontweight="bold")
ax.text((xs[1] + xs[2]) / 2, sy - 0.2, "10 Mb/s, 5 ms\nqueue 100", ha="center", va="top", fontsize=13.1, color=ATT)

# hosts
hy = 2.02
hosts = [
    ("h11", "10.0.1.1", "ping", BEN, BEN_BG, None),
    ("h12", "10.0.1.2", "TCP transfer", BEN, BEN_BG, None),
    ("h21", "10.0.2.1", "small-packet\nUDP", BEN, BEN_BG, None),
    ("h22", "10.0.2.2", "attacker", ATT, ATT_BG, "all"),
    ("h31", "10.0.3.1", "attacker (S2)\nflood host (S9)", ATT2, ATT2_BG, "S2, S9"),
    ("h32", "10.0.3.2", "attacker (S2)", ATT2, ATT2_BG, "S2"),
    ("h41", "10.0.4.1", "victim", VIC, VIC_BG, None),
    ("h42", "10.0.4.2", "bursty\nbenign UDP", BEN, BEN_BG, None),
]
for i, (n, ip, role, col, bg, tag) in enumerate(hosts):
    sx = xs[i // 2]; x = HX[i]
    arrow((sx + (x - sx) * 0.15, sy - 0.36), (x - 0.08, hy + 0.36), lw=1.6)
    host(x, hy, n, ip, role, col, bg, tag)

# legend
ly = 0.12
items = [("attacker, every scenario", ATT, ATT_BG), ("added attacker / flood host", ATT2, ATT2_BG),
         ("victim", VIC, VIC_BG), ("benign host", BEN, BEN_BG)]
for (t, c, bg), lx in zip(items, [0.9, 3.75, 6.75, 8.55]):
    ax.add_patch(FancyBboxPatch((lx, ly), 0.26, 0.16, boxstyle="round,pad=0,rounding_size=0.03", fc=bg, ec=c, lw=1.2))
    ax.text(lx + 0.34, ly + 0.08, t, va="center", ha="left", fontsize=13.8, color=INK)

fig.savefig("topo.png", dpi=300, facecolor="white")
fig.savefig("topo.pdf", facecolor="white")

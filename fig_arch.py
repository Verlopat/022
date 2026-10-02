"""System diagram of the camera-to-green-split pipeline (online loop + offline training)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle

import figstyle as fs

fs.setup()

C_PER = '#2a78d6'   # perception
C_DET = '#eb6834'   # detection
C_SCO = '#eda100'   # scoring
C_GRA = '#1baf7a'   # graph
C_GAT = '#4a3aa7'   # estimator
C_TIM = '#008300'   # timing
C_SIG = '#e34948'   # signals
C_OFF = '#e87ba4'   # offline


def tint(hexc, a=0.13):
    hexc = hexc.lstrip('#')
    r, g, b = [int(hexc[i:i + 2], 16) for i in (0, 2, 4)]
    r = int(255 - (255 - r) * a)
    g = int(255 - (255 - g) * a)
    b = int(255 - (255 - b) * a)
    return f'#{r:02x}{g:02x}{b:02x}'


def box(ax, x, y, w, h, color, title, body, tsize=8.2, bsize=7.0):
    p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.0,rounding_size=1.2',
                       linewidth=1.3, edgecolor=color, facecolor=tint(color, 0.12))
    ax.add_patch(p)
    ax.add_patch(Rectangle((x, y + h - 3.6), w, 3.6, facecolor=color, edgecolor='none', alpha=0.95))
    ax.text(x + w / 2, y + h - 1.8, title, ha='center', va='center', fontsize=tsize, color='white',
            fontweight='bold')
    ax.text(x + w / 2, y + (h - 3.6) / 2, body, ha='center', va='center', fontsize=bsize, color=fs.INK,
            linespacing=1.35)


def arrow(ax, x0, y0, x1, y1, color=fs.INK2, lw=1.3, style='-|>', ls='-', rad=0.0):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=9, linewidth=lw,
                        color=color, linestyle=ls, connectionstyle=f'arc3,rad={rad}')
    ax.add_patch(a)


fig = plt.figure(figsize=(7.6, 4.85))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 103)
ax.set_ylim(0, 64.5)
ax.axis('off')
ax.grid(False)

# bands
ax.add_patch(FancyBboxPatch((0.6, 24.0), 98.8, 39.4, boxstyle='round,pad=0,rounding_size=1.5',
                            linewidth=0.9, edgecolor=fs.AXIS, facecolor='#fbfbfa'))
ax.text(1.8, 62.0, 'Online control: strategic layer once per cycle k, tactical layer every second', fontsize=8.0,
        color=fs.INK, fontweight='bold', va='center')
ax.add_patch(FancyBboxPatch((0.6, 0.6), 98.8, 21.4, boxstyle='round,pad=0,rounding_size=1.5',
                            linewidth=0.9, edgecolor=fs.AXIS, facecolor='#fbfbfa'))
ax.text(1.8, 20.4, 'Offline estimator training (simulation, before deployment)', fontsize=8.0, color=fs.INK,
        fontweight='bold', va='center')

# online row
y0, h = 43.3, 16.4
w = 14.6
xs = [1.8, 18.1, 34.4, 50.7, 67.0, 83.3]
box(ax, xs[0], y0, w, h, C_PER, 'Cameras', '12 arm cameras\n60 m field of view\none frame per arm\nper cycle')
box(ax, xs[1], y0, w, h, C_DET, 'YOLO detector', 'UVH-26 weights\n(best.pt)\n14 classes mapped\nto 7 groups')
box(ax, xs[2], y0, w, h, C_SCO, 'PCU scoring', r'$s_{ij}=\sum_c w_c\,\hat{n}_{ij,c}$' + '\nclass-group counts\nfill ratio, history\n14 features per node')
box(ax, xs[3], y0, w, h, C_GRA, 'Junction graph', '12 arm nodes\n3 junction nodes\nintra, hub, trunk\nand feeder edges')
box(ax, xs[4], y0, w, h, C_GAT, 'GAT estimator', '2 layers x 4 heads\narm: ' + r'$\hat{Q}_{ij}$' + ', ' + r'$\hat{A}_{ij}$'
    + '\n(queue, arrivals)\njunction: ' + r'$\hat{\psi}_j$' + '\n(flow ratio)', bsize=6.8)
box(ax, xs[5], y0, w, h, C_TIM, 'Timing law', 'cycle ' + r'$C_{j,k}$' + ' from\nWebster at ' + r'$\hat{\psi}_j$'
    + '\nguard ' + r'$\delta_{ij}$' + ' on\nloaded links\n' + r'$g_{ij}\propto\delta_{ij}(\hat{Q}_{ij}+\hat{A}_{ij}C/H)$',
    bsize=6.6)
for k in range(5):
    arrow(ax, xs[k] + w + 0.1, y0 + h / 2 - 1.8, xs[k + 1] - 0.1, y0 + h / 2 - 1.8, color=fs.INK2)

# tactical layer and signal heads
sy, sh = 28.0, 12.8
box(ax, 55.2, sy, 20.6, sh, C_SIG, 'Tactical check (1 Hz)', '20 m stop-line zone of\nthe green arm; end green\nwhen empty twice or no\ncrossing for 4 s (Eq. 12)',
    tsize=7.6, bsize=6.6)
box(ax, 80.8, sy, 18.0, sh, '#6b5b45', 'Signal heads', 'J1, J2, J3; one arm\nper phase; amber 3 s,\nall-red 2 s; min\ngreen 8 s', tsize=7.6, bsize=6.6)
arrow(ax, 90.6, y0 - 0.1, 90.6, sy + sh + 0.1, color=fs.INK2)
ax.text(91.4, y0 - 1.2, 'plan', fontsize=6.2, color=fs.INK2, va='center')
arrow(ax, 75.9, sy + sh / 2 - 1.6, 80.7, sy + sh / 2 - 1.6, color=C_SIG, lw=1.4)
ax.text(78.3, sy + sh / 2 + 0.2, 'early\nend', fontsize=5.8, color=fs.INK2, ha='center', va='bottom', linespacing=1.1)

# corridor sketch
cx0, cx1, cy = 3.0, 50.5, 35.0
ax.add_patch(Rectangle((cx0, cy - 1.2), cx1 - cx0, 2.4, facecolor='#d9d8d2', edgecolor='none'))
for jx, lab in zip((11.5, 26.5, 41.5), ('J1', 'J2', 'J3')):
    ax.add_patch(Rectangle((jx - 1.4, cy - 5.4), 2.8, 10.8, facecolor='#d9d8d2', edgecolor='none'))
    ax.add_patch(Rectangle((jx - 1.6, cy - 1.6), 3.2, 3.2, facecolor='white', edgecolor=fs.INK2, linewidth=0.8))
    ax.text(jx, cy, lab, ha='center', va='center', fontsize=6.8, color=fs.INK, fontweight='bold')
    for dx, dy in ((-3.6, 0), (3.6, 0), (0, 3.6), (0, -3.6)):
        ax.add_patch(Circle((jx + dx, cy + dy), 0.6, facecolor=C_PER, edgecolor='white', linewidth=0.6))
ax.text(24.0, 27.6, 'corridor: 3 junctions, 4 arms x 2 lanes, camera on every arm',
        ha='center', fontsize=6.3, color=fs.INK2)
# stop-line zones feed the tactical check
arrow(ax, cx1 + 0.3, cy + 1.6, 55.1, cy + 1.6, color=C_SIG, lw=1.2)
# signal heads act on the corridor (route under the tactical box)
ax.plot([89.8, 89.8, 52.8, 52.8], [sy - 0.1, 26.4, 26.4, 30.6], color=fs.INK2, lw=1.3, solid_capstyle='butt')
arrow(ax, 52.8, 30.5, 49.2, 33.4, color=fs.INK2)
ax.text(71.0, 25.3, 'green times to the signal heads of J1, J2, J3', fontsize=6.0, color=fs.INK2, ha='center', va='center')
# feedback to cameras
arrow(ax, 6.0, 40.0, 6.0, y0 - 0.2, color=C_PER, lw=1.3)
ax.text(7.0, 41.4, 'frames at next cycle start', fontsize=6.2, color=fs.INK2, va='center')

# offline row
oy, oh = 2.6, 15.2
ow = 17.6
oxs = [1.8, 21.3, 40.8, 60.3, 79.8]
box(ax, oxs[0], oy, ow, oh, C_OFF, 'Corridor simulator', 'mesoscopic, 7 classes\nspillback, split phasing\ndelay verified against\nWebster formula', bsize=6.8)
box(ax, oxs[1], oy, ow, oh, C_OFF, 'Behaviour policy', r'random $C\in[60,160]$ s' + '\nperturbed splits\nrandom demand, tidal\nratio and incidents', bsize=6.8)
box(ax, oxs[2], oy, ow, oh, C_OFF, 'Detector profiles', 'P0 oracle, P1 UVH-26\nP2 COCO YOLOv8m\nP3 COCO YOLOv8n\nrecall, confusion, FP', bsize=6.8)
box(ax, oxs[3], oy, ow, oh, C_OFF, 'Training set', '400 episodes\n26,745 graph snapshots\ntargets from vehicle\npassage logs (Eq. 7)', bsize=6.8)
box(ax, oxs[4], oy, ow, oh, C_OFF, 'Supervised fit', 'Huber loss on ' + r'$\hat{Q}$, $\hat{A}$' + '\nsquared loss on ' + r'$\hat{\psi}$' + '\nAdam, cosine rate\nearly stopping', bsize=6.8)
for k in range(4):
    arrow(ax, oxs[k] + ow + 0.1, oy + oh / 2 - 1.5, oxs[k + 1] - 0.1, oy + oh / 2 - 1.5, color=fs.INK2)
# weights to the GAT estimator along the right margin
ax.plot([97.5, 101.6, 101.6, 74.3], [oy + oh / 2, oy + oh / 2, 63.3, 63.3], color=C_GAT, lw=1.5,
        solid_capstyle='round')
arrow(ax, 74.3, 63.35, 74.3, y0 + h + 0.1, color=C_GAT, lw=1.5)
ax.text(88.0, 64.0, 'trained weights', fontsize=6.3, color=C_GAT, ha='center', va='center')

fs.save(fig, 'figs/fig01_system.png')
print('ok')

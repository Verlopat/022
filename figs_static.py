"""Static figures: corridor layout, graph structure, class mix, demand profiles, simulator verification."""
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle, Patch
from matplotlib.lines import Line2D

import figstyle as fs
import simcore as sc
import controllers as ct

fs.setup()
PAL = fs.PAL


# ---------------------------------------------------------------------------
def fig_corridor():
    fig, ax = plt.subplots(figsize=(7.3, 3.5))
    ax.set_xlim(-340, 900)
    ax.set_ylim(-230, 185)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.grid(False)
    road = '#d9d8d2'
    half = 22.0   # displayed half-width of a road
    jx = [0.0, 300.0, 550.0]
    ax.add_patch(Rectangle((-320, -half), 1190, 2 * half, facecolor=road, edgecolor='none', zorder=1))
    for x in jx:
        ax.add_patch(Rectangle((x - half, -170), 2 * half, 340, facecolor=road, edgecolor='none', zorder=1))
    # centre lines
    ax.plot([-320, 870], [0, 0], color='white', lw=1.0, zorder=2)
    for x in jx:
        ax.plot([x, x], [-170, 170], color='white', lw=1.0, zorder=2)
    # junction boxes
    for k, x in enumerate(jx):
        ax.add_patch(Rectangle((x - half, -half), 2 * half, 2 * half, facecolor='#f3f2ee', edgecolor=fs.INK2,
                               lw=0.8, zorder=3))
        ax.text(x, 0, f'J{k + 1}', ha='center', va='center', fontsize=8, fontweight='bold', color=fs.INK, zorder=4)
    cam = dict(facecolor=PAL[0], alpha=0.28, edgecolor=PAL[0], lw=0.6, zorder=2.5)
    zone = 60.0
    for x in jx:
        # W approach (eastbound, north half in left-hand traffic)
        ax.add_patch(Rectangle((x - half - zone, 0), zone, half, **cam))
        # E approach (westbound, south half)
        ax.add_patch(Rectangle((x + half, -half), zone, half, **cam))
        # N approach (southbound, east half)
        ax.add_patch(Rectangle((x, half), half, zone, **cam))
        # S approach (northbound, west half)
        ax.add_patch(Rectangle((x - half, -half - zone), half, zone, **cam))
        # camera icons
        for (cx, cy) in ((x - half - zone - 8, half + 10), (x + half + zone + 8, -half - 10),
                         (x + half + 10, half + zone + 8), (x - half - 10, -half - zone - 8)):
            ax.add_patch(Circle((cx, cy), 6.5, facecolor=PAL[0], edgecolor='white', lw=0.8, zorder=5))
    # flow arrows
    def arr(x0, y0, x1, y1, c):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=8, lw=1.4, color=c,
                                     zorder=6))
    for (a, b) in ((-300, -110), (60, 170), (330, 420), (600, 700)):
        arr(a, 11, b, 11, PAL[1])
    for (a, b) in ((860, 700), (500, 420), (250, 170), (-60, -170)):
        arr(a, -11, b, -11, PAL[2])
    for x in jx:
        arr(x + 11, 165, x + 11, 95, PAL[6])
        arr(x - 11, -165, x - 11, -95, PAL[6])
    # dimensions
    def dim(x0, x1, y, txt):
        ax.annotate('', xy=(x0, y), xytext=(x1, y), arrowprops=dict(arrowstyle='<->', color=fs.INK2, lw=0.8))
        ax.text((x0 + x1) / 2, y - 12, txt, ha='center', va='top', fontsize=7.2, color=fs.INK2)
    dim(0, 300, -188, '300 m (85.7 car-spaces)')
    dim(300, 550, -188, '250 m (71.4 car-spaces)')
    for x in jx:
        ax.plot([x, x], [-176, -196], color=fs.INK2, lw=0.6)
    ax.text(-320, 36, 'west entry\n300 m', fontsize=6.8, color=fs.INK2, va='bottom')
    ax.text(870, -36, 'east entry\n300 m', fontsize=6.8, color=fs.INK2, va='top', ha='right')
    # incident marker at J3-W nearside lane
    ix = 550 - half - 30
    ax.plot([ix], [16], marker='X', markersize=9, color=PAL[7], markeredgecolor='white', zorder=7)
    ax.text(425, 112, 'S5: nearside lane\nblocked 1500-2700 s', fontsize=6.8, color=fs.INK, ha='center', va='bottom')
    ax.plot([450, ix - 3], [108, 26], color=fs.INK2, lw=0.7, zorder=6)
    # boundary flows (S2)
    q = sc.scenario_boundary_flows(sc.SCENARIOS['S2'])
    ax.text(-230, -46, f'{q[0]:,.0f} veh/h', fontsize=6.6, color=fs.INK2, ha='center')
    ax.text(780, 30, f'{q[10]:,.0f} veh/h', fontsize=6.6, color=fs.INK2, ha='center')
    handles = [Patch(facecolor=PAL[0], alpha=0.35, edgecolor=PAL[0], label='camera field of view (60 m)'),
               Line2D([0], [0], color=PAL[1], lw=1.6, label='eastbound'),
               Line2D([0], [0], color=PAL[2], lw=1.6, label='westbound'),
               Line2D([0], [0], color=PAL[6], lw=1.6, label='side-street arms'),
               Line2D([0], [0], marker='X', color='none', markerfacecolor=PAL[7], markeredgecolor='white',
                      markersize=8, label='incident location (S5)')]
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize=7.2,
              handlelength=1.6, columnspacing=1.2)
    fs.save(fig, 'figs/fig02_corridor.png')


# ---------------------------------------------------------------------------
def fig_graph():
    net = sc.Network()
    M = ct.build_mask(net, 'full')
    pos = {}
    for j in range(3):
        cx = 6.0 * j
        pos[12 + j] = (cx, 0.0)
        pos[4 * j + 0] = (cx - 1.9, 0.0)
        pos[4 * j + 2] = (cx + 1.9, 0.0)
        pos[4 * j + 1] = (cx, 1.9)
        pos[4 * j + 3] = (cx, -1.9)
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.set_aspect('equal')
    ax.axis('off')
    ax.grid(False)
    jcol = [PAL[0], PAL[1], PAL[2]]
    counts = {'intra': 0, 'hub': 0, 'trunk': 0, 'feeder': 0}
    drawn = set()
    feeder_pairs = []
    for p in range(15):
        for q in range(15):
            if p == q or not M[p, q]:
                continue
            key = tuple(sorted((p, q)))
            if p < 12 and q < 12 and p // 4 == q // 4:
                typ = 'intra'
            elif (p >= 12) != (q >= 12) and (min(p, q) // 4 == max(p, q) - 12):
                typ = 'hub'
            elif p >= 12 and q >= 12:
                typ = 'trunk'
            else:
                typ = 'feeder'
            counts[typ] += 1
            if key in drawn:
                continue
            drawn.add(key)
            (x0, y0), (x1, y1) = pos[key[0]], pos[key[1]]
            if typ == 'intra':
                rad = 0.25 if (abs(x0 - x1) > 0.1 and abs(y0 - y1) > 0.1) else 0.0
                if abs(y0 - y1) < 0.01 or abs(x0 - x1) < 0.01:
                    rad = 0.55
                ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-', connectionstyle=f'arc3,rad={rad}',
                                             color=fs.MUTED, lw=0.9, zorder=1))
            elif typ == 'hub':
                j = max(key) - 12
                ax.plot([x0, x1], [y0, y1], color=jcol[j], lw=1.6, zorder=2)
            elif typ == 'trunk':
                rad = -0.35 if abs(x0 - x1) > 7 else -0.18
                ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-', connectionstyle=f'arc3,rad={rad}',
                                             color=PAL[6], lw=1.8, zorder=2))
            else:
                feeder_pairs.append(key)
    for (a, b) in feeder_pairs:
        (x0, y0), (x1, y1) = pos[a], pos[b]
        rad = 0.22 if y0 + y1 >= 0 else -0.22
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-', connectionstyle=f'arc3,rad={rad}',
                                     color=PAL[7], lw=1.1, zorder=1.5, alpha=0.85))
    names = ['W', 'N', 'E', 'S']
    for n, (x, y) in pos.items():
        if n >= 12:
            ax.add_patch(Circle((x, y), 0.55, facecolor=PAL[6], edgecolor='white', lw=1.5, zorder=4))
            ax.text(x, y, f'J{n - 11}', ha='center', va='center', color='white', fontsize=7.5, fontweight='bold',
                    zorder=5)
        else:
            j, d = divmod(n, 4)
            internal = n in net.internal
            ax.add_patch(Circle((x, y), 0.42, facecolor=jcol[j] if not internal else 'white',
                                edgecolor=jcol[j], lw=1.8, zorder=4))
            ax.text(x, y, f'{names[d]}', ha='center', va='center', fontsize=7,
                    color='white' if not internal else fs.INK, fontweight='bold', zorder=5)
    ax.set_xlim(-2.8, 14.8)
    ax.set_ylim(-3.2, 3.0)
    handles = [Line2D([0], [0], color=fs.MUTED, lw=1.2, label=f'intra-junction ({counts["intra"]})'),
               Line2D([0], [0], color=PAL[0], lw=1.8, label=f'arm-hub ({counts["hub"]}), junction colour'),
               Line2D([0], [0], color=PAL[6], lw=1.8, label=f'hub-hub trunk ({counts["trunk"]})'),
               Line2D([0], [0], color=PAL[7], lw=1.4, label=f'feeder ({counts["feeder"]})'),
               Line2D([0], [0], marker='o', color='none', markerfacecolor=fs.MUTED, markeredgecolor=fs.MUTED,
                      markersize=7, label='boundary arm (filled)'),
               Line2D([0], [0], marker='o', color='none', markerfacecolor='white', markeredgecolor=fs.MUTED,
                      markeredgewidth=1.5, markersize=7, label='internal arm (open)')]
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.03), ncol=3, fontsize=7.2,
              columnspacing=1.2, handlelength=1.8)
    fs.save(fig, 'figs/fig03_graph.png')
    print('directed edge counts (no self loops):', counts, 'total', sum(counts.values()))
    return counts


# ---------------------------------------------------------------------------
def fig_mix():
    labels = ['Two-wheeler', 'Bicycle', 'Three-wheeler (auto)', 'Car (hatchback, sedan, SUV, MUV)',
              'LCV, van, tempo-traveller', 'Bus, mini-bus', 'Truck']
    share = sc.MIX_BASE * 100
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    ax.grid(False)
    cols = [PAL[0], PAL[3], PAL[1], PAL[2], PAL[4], PAL[6], PAL[7]]
    wedges, _ = ax.pie(share, colors=cols, startangle=90, counterclock=False,
                       wedgeprops=dict(edgecolor='white', linewidth=2.0), radius=1.0)
    for wdg, s in zip(wedges, share):
        if s >= 10:
            ang = np.deg2rad((wdg.theta1 + wdg.theta2) / 2)
            ax.text(0.62 * np.cos(ang), 0.62 * np.sin(ang), f'{s:.0f}%', ha='center', va='center',
                    color='white', fontsize=9, fontweight='bold')
    leg = [f'{l}: {s:.1f}%, w = {w:g}' for l, s, w in zip(labels, share, sc.W_DESIGN)]
    ax.legend(wedges, leg, loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=7.8, handlelength=1.2)
    ax.set_aspect('equal')
    fs.save(fig, 'figs/fig04_mix.png')


# ---------------------------------------------------------------------------
def fig_demand():
    t = np.arange(0, 4800, 10)
    prof = np.array([sc.demand_profile(x) for x in t])
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.axvspan(0, 600, color='#efeee9', zorder=0)
    ax.axvspan(1500, 2700, color=PAL[7], alpha=0.10, zorder=0)
    ax.text(300, 0.93, 'warm-up', transform=ax.get_xaxis_transform(), ha='center', fontsize=7.5, color=fs.INK2)
    ax.text(2100, 0.02, 'S5 incident window', transform=ax.get_xaxis_transform(), ha='center', fontsize=7.5,
            color=fs.INK2)
    order = ['S1', 'S2', 'S3', 'S4']
    for k, s in enumerate(order):
        q = sc.scenario_boundary_flows(sc.SCENARIOS[s])
        tot = sum(q.values())
        lab = f"{s} {sc.SCENARIOS[s]['name']}" + (' (S5 uses S2 demand)' if s == 'S2' else '')
        ax.plot(t / 60.0 * 60, tot * prof, color=PAL[k], lw=1.9, label=lab)
    ax.set_xlabel('Simulation time (s)')
    ax.set_ylabel('Network inflow (veh/h)')
    ax.set_xlim(0, 4800)
    ax.axvline(3600, color=fs.AXIS, lw=0.9)
    ax.text(3620, 0.93, 'end of cohort\nrecording', transform=ax.get_xaxis_transform(), fontsize=7.2, color=fs.INK2,
            va='top')
    ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.22), ncol=2, fontsize=7.6)
    fs.save(fig, 'figs/fig05_demand.png')


# ---------------------------------------------------------------------------
def fig_webster():
    d = json.load(open('results/verify_webster.json'))
    x = np.array([r['x'] for r in d])
    ws = np.array([r['webster'] for r in d])
    sm = np.array([r['sim_mean'] for r in d])
    ci = np.array([r['sim_ci'] for r in d])
    # smooth Webster curve
    C, g = 96, 19
    ge = g - 1.0
    lam = ge / C
    xx = np.linspace(0.25, 0.955, 200)
    q = xx * lam
    wc = C * (1 - lam) ** 2 / (2 * (1 - lam * xx)) + xx ** 2 / (2 * q * (1 - xx)) - 0.65 * (C / q ** 2) ** (1 / 3) * xx ** (2 + 5 * lam)
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.plot(xx, wc, color=PAL[0], lw=2.0, label="Webster's delay formula")
    ax.errorbar(x, sm, yerr=ci, fmt='o', color=PAL[1], ecolor=PAL[1], elinewidth=1.2, capsize=3, markersize=5.5,
                markeredgecolor='white', markeredgewidth=1.0, label='simulator, mean and 95% CI (16 runs x 8 h)')
    ax.set_xlabel('Degree of saturation x')
    ax.set_ylabel('Average control delay (s/veh)')
    ax.set_xlim(0.25, 0.98)
    ax.legend(loc='upper left')
    fs.save(fig, 'figs/fig06_webster.png')
    err = (sm - ws) / ws * 100
    print('relative error %', np.round(err, 1))


if __name__ == '__main__':
    fig_corridor()
    fig_graph()
    fig_mix()
    fig_demand()
    fig_webster()

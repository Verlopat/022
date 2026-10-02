import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
INK = '#0b0b0b'
INK2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'
BLUE_SEQ = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6',
            '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']

CTRL_ORDER = ['FT-W', 'ACT', 'MP-A', 'LPS', 'LPS-T', 'GATC-S', 'GATC', 'MP-C']
CTRL_COLOR = {c: PAL[i] for i, c in enumerate(CTRL_ORDER)}
CTRL_COLOR['DQN-C'] = '#8a5a2b'
CTRL_COLOR['DQN-O'] = '#17a2b8'
CTRL_LABEL = {'FT-W': 'FT-W (design-day fixed time)', 'ACT': 'ACT (loop actuated)', 'MP-C': 'MP-C (cyclic max-pressure)',
              'MP-A': 'MP-A (acyclic max-pressure)', 'LPS': 'LPS (local proportional)', 'LPS-T': 'LPS-T (LPS + 1 Hz check)',
              'GATC-S': 'GATC-S (proposed, snapshot)', 'GATC': 'GATC (proposed, full)',
              'DQN-C': 'DQN-C (deep Q-network, camera)', 'DQN-O': 'DQN-O (deep Q-network, exact state)'}


def setup():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8.5,
        'ytick.labelsize': 8.5,
        'legend.fontsize': 8.5,
        'axes.edgecolor': AXIS,
        'axes.labelcolor': INK,
        'xtick.color': INK2,
        'ytick.color': INK2,
        'axes.grid': True,
        'grid.color': GRID,
        'grid.linewidth': 0.8,
        'grid.linestyle': '-',
        'axes.axisbelow': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'legend.frameon': False,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.04,
        'lines.linewidth': 1.8,
        'lines.solid_capstyle': 'round',
    })


def save(fig, path):
    fig.savefig(path, facecolor='white')
    plt.close(fig)

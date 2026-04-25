import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── Load data ──────────────────────────────────────────────────────────
def load(path):
    df = pd.read_csv(path)
    df.columns = ['walltime', 'step', 'value']
    return df

r_return = load('/mnt/user-data/uploads/all_random_total_return_per_episode.csv')
m_return = load('/mnt/user-data/uploads/mix_total_reutrn_per_episode.csv')
r_items  = load('/mnt/user-data/uploads/all_random_items_per_episode.csv')
m_items  = load('/mnt/user-data/uploads/mix_items_per_episode.csv')
r_wait   = load('/mnt/user-data/uploads/all_random_fraction_of_wait_moves_per_episode.csv')
m_wait   = load('/mnt/user-data/uploads/mix_fraction_of_wait_moves_per_episode.csv')

# ── Smoothing ──────────────────────────────────────────────────────────
def smooth(values, window=30):
    return pd.Series(values).rolling(window, min_periods=1, center=True).mean().values

# ── Style ──────────────────────────────────────────────────────────────
MIX_COLOR    = '#2563EB'   # blue
RANDOM_COLOR = '#DC2626'   # red
ALPHA_RAW    = 0.15
LW           = 2.0

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
    'figure.dpi': 150,
})

# ── Plot 1: Total Return ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))

rx, ry = r_return['step'].values, r_return['value'].values
mx, my = m_return['step'].values, m_return['value'].values

ax.plot(rx, ry, color=RANDOM_COLOR, alpha=ALPHA_RAW, linewidth=0.6)
ax.plot(mx, my, color=MIX_COLOR,    alpha=ALPHA_RAW, linewidth=0.6)
ax.plot(rx, smooth(ry), color=RANDOM_COLOR, linewidth=LW, label='Random')
ax.plot(mx, smooth(my), color=MIX_COLOR,    linewidth=LW, label='Mix')

ax.set_title('Total Return per Episode', fontsize=13, fontweight='500', pad=10)
ax.set_xlabel('Episode', fontsize=11)
ax.set_ylabel('Cumulative Reward', fontsize=11)
ax.legend(frameon=False, fontsize=11)
ax.set_xlim(left=0)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/plot_return.png', bbox_inches='tight', dpi=200)
plt.close()
print("Saved plot_return.png")

# ── Plot 2: Goals Collected ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))

rx, ry = r_items['step'].values, r_items['value'].values
mx, my = m_items['step'].values, m_items['value'].values

ax.plot(rx, ry, color=RANDOM_COLOR, alpha=ALPHA_RAW, linewidth=0.6)
ax.plot(mx, my, color=MIX_COLOR,    alpha=ALPHA_RAW, linewidth=0.6)
ax.plot(rx, smooth(ry), color=RANDOM_COLOR, linewidth=LW, label='Random')
ax.plot(mx, smooth(my), color=MIX_COLOR,    linewidth=LW, label='Mix')

ax.set_title('Goals Collected per Episode', fontsize=13, fontweight='500', pad=10)
ax.set_xlabel('Episode', fontsize=11)
ax.set_ylabel('Goals Collected', fontsize=11)
ax.legend(frameon=False, fontsize=11)
ax.set_xlim(left=0)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/plot_goals.png', bbox_inches='tight', dpi=200)
plt.close()
print("Saved plot_goals.png")

# ── Plot 3: Wait Action Fraction ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))

rx, ry = r_wait['step'].values, r_wait['value'].values
mx, my = m_wait['step'].values, m_wait['value'].values

ax.plot(rx, ry, color=RANDOM_COLOR, alpha=ALPHA_RAW, linewidth=0.6)
ax.plot(mx, my, color=MIX_COLOR,    alpha=ALPHA_RAW, linewidth=0.6)
ax.plot(rx, smooth(ry), color=RANDOM_COLOR, linewidth=LW, label='Random')
ax.plot(mx, smooth(my), color=MIX_COLOR,    linewidth=LW, label='Mix')
ax.axhline(0.25, color='gray', linestyle=':', linewidth=1.4, label='Uniform baseline (0.25)')

ax.set_title('Wait Action Fraction per Episode', fontsize=13, fontweight='500', pad=10)
ax.set_xlabel('Episode', fontsize=11)
ax.set_ylabel('Fraction of Wait Actions', fontsize=11)
ax.legend(frameon=False, fontsize=11)
ax.set_xlim(left=0)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/plot_wait.png', bbox_inches='tight', dpi=200)
plt.close()
print("Saved plot_wait.png")
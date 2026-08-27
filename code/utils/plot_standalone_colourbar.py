# %%
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['CMU']
rcParams['text.usetex'] = True
rcParams['axes.labelsize'] = 9
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 9
mm2inches = 0.039371

fig = plt.figure(figsize=(20*mm2inches, 8*mm2inches))
# Leave more space around the colorbar for labels
ax1 = fig.add_axes([0.1, 0.3, 0.8, 0.4])

cmap = mpl.cm.coolwarm
norm = mpl.colors.Normalize(vmin=-0.02, vmax=0.02)

cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,
norm=norm,
orientation='horizontal')

# Adjust tick label size to fit better
cb1.ax.tick_params(labelsize=8)

plt.savefig('/Users/joebathelt/Desktop/colorbar.png', dpi=300, bbox_inches='tight', 
            pad_inches=0.1)

# %%
fig = plt.figure(figsize=(15*mm2inches, 8*mm2inches))
# Leave more space around the colorbar for labels
ax1 = fig.add_axes([0.1, 0.3, 0.8, 0.4])

cmap = mpl.cm.RdBu_r
norm = mpl.colors.Normalize(vmin=-0.3, vmax=0.3)

cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,
norm=norm,
orientation='horizontal')

# Adjust tick label size to fit better
cb1.ax.tick_params(labelsize=8)
cb1.ax.set_xticks([-0.3, 0.3])

# Set the background colour to transparent
cb1.ax.set_facecolor('none')

plt.savefig('/Users/joebathelt/Desktop/colorbar.png', dpi=300, bbox_inches='tight', 
            pad_inches=0.1)

# %%

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

fig = plt.figure()
handles = [Line2D([0], [0], label=str(i)) for i in range(6)]
# 0, 1, 2, 3, 4, 5
fig.legend(handles=handles, ncol=3, loc='center')
plt.savefig('legend_test.png')

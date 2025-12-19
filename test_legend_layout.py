import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Test different orderings to understand the fill pattern
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Test 1: Simple 1-6 ordering
handles1 = [Line2D([0], [0], marker='o', color='w', label=f'Item {i}', 
                   markerfacecolor='red', markersize=10) for i in range(1, 7)]
axes[0, 0].set_title('Test 1: Items 1-6, ncol=3')
axes[0, 0].legend(handles=handles1, ncol=3, loc='center')
axes[0, 0].axis('off')

# Test 2: With blank at position 4 (row-major would put it at row 2, col 1)
handles2 = [Line2D([0], [0], marker='o', color='w', label=f'Item {i}', 
                   markerfacecolor='red', markersize=10) for i in [1, 2, 3]]
handles2.append(Line2D([0], [0], color='none', marker='None', label=' '))
handles2.extend([Line2D([0], [0], marker='o', color='w', label=f'Item {i}', 
                        markerfacecolor='red', markersize=10) for i in [4, 5]])
axes[0, 1].set_title('Test 2: [1,2,3,blank,4,5], ncol=3')
axes[0, 1].legend(handles=handles2, ncol=3, loc='center')
axes[0, 1].axis('off')

# Test 3: Column-major ordering [1, blank, 2, 4, 3, 5]
handles3 = []
handles3.append(Line2D([0], [0], marker='o', color='w', label='Item 1', 
                      markerfacecolor='red', markersize=10))
handles3.append(Line2D([0], [0], color='none', marker='None', label=' '))
handles3.append(Line2D([0], [0], marker='o', color='w', label='Item 2', 
                      markerfacecolor='blue', markersize=10))
handles3.append(Line2D([0], [0], marker='o', color='w', label='Item 4', 
                      markerfacecolor='blue', markersize=10))
handles3.append(Line2D([0], [0], marker='o', color='w', label='Item 3', 
                      markerfacecolor='green', markersize=10))
handles3.append(Line2D([0], [0], marker='o', color='w', label='Item 5', 
                      markerfacecolor='green', markersize=10))
axes[1, 0].set_title('Test 3: [1,blank,2,4,3,5], ncol=3')
axes[1, 0].legend(handles=handles3, ncol=3, loc='center')
axes[1, 0].axis('off')

# Test 4: What the user wants - try mode='expand' or different approach
# Target: Col1=[a], Col2=[b,d], Col3=[c,none]
# Try: a in first col alone, then b,c in row, then d,none in row
handles4 = []
handles4.append(Line2D([0], [0], marker='o', color='w', label='a (Original T10)', 
                      markerfacecolor='red', markersize=10))
handles4.append(Line2D([0], [0], marker='o', color='w', label='b (Zero-DP)', 
                      markerfacecolor='blue', markersize=10))
handles4.append(Line2D([0], [0], marker='o', color='w', label='c (Fused)', 
                      markerfacecolor='green', markersize=10))
handles4.append(Line2D([0], [0], marker='o', color='w', label='d (Limited)', 
                      markerfacecolor='purple', markersize=10))
handles4.append(Line2D([0], [0], marker='o', color='w', label='none (Stricter)', 
                      markerfacecolor='black', markersize=10))
axes[1, 1].set_title('Test 4: ncol=2 approach')
axes[1, 1].legend(handles=handles4, ncol=2, loc='center')
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'legend_layout_test.png'), dpi=150)
print("Saved test figure to legend_layout_test.png")

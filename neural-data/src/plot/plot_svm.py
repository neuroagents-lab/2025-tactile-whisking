import matplotlib.pyplot as plt
import numpy as np
from utils.plotting import COLORS, FONTSIZE

num_categories = [2, 3, 4]
means_1000hz = [0.8275, 0.8148, 0.7123]
stds_1000hz = [0.0402, 0.0794, 0.0597]
means_110hz = [0.8759, 0.8713, 0.8333]
stds_110hz = [0.0124, 0.0200, 0.0170]

x = np.arange(len(num_categories))
width = 0.35
fontsize = 11

fig, ax = plt.subplots()
fig.set_size_inches(3, 2)
fig.subplots_adjust(top=0.8)
bars1 = ax.bar(x - width/2, means_1000hz, width, yerr=stds_1000hz, label='Low-Var High-Fid', capsize=5, color=COLORS["red"])
bars2 = ax.bar(x + width/2, means_110hz, width, yerr=stds_110hz, label='High-Var Low-Fid', capsize=5, color=COLORS["blue"])

pos = ax.get_position()
ax.set_position([pos.x0, pos.y0 - 0.05, pos.width, pos.height * 0.95])

ax.set_ylabel('Classification Accuracy')
ax.set_xticks(x)
ax.set_xlabel("Number of Classes")
ax.set_xticklabels(num_categories)
ax.set_ylim(0, 1)

for i, c in enumerate(num_categories):
    x = i*1
    print(x, c)
    ax.hlines(y=1/c, xmin=x-0.5, xmax=x+0.5, color=COLORS["yellow"], linestyle="--")

ax.plot([0, 0], [0, 0], color=COLORS["yellow"], linestyle="--", label="Random Chance")
ax.legend(bbox_to_anchor=(1, 1.25), loc='upper right', fontsize=FONTSIZE-2)

plt.title("SVM Classification on Whisk Datasets", y=1.25)

plt.savefig("out/svm_accuracy.pdf", transparent=True)
print("saved to out/svm_accuracy.pdf")

# plt.savefig("out/svm_accuracy.png", dpi=300, bbox_inches='tight')
# print("saved to out/svm_accuracy.png")
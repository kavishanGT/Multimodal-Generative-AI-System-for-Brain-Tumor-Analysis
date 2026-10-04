# visualize_samples.py

import matplotlib.pyplot as plt
from data_loader import train_loader
data_iter = iter(train_loader)

images, labels = next(data_iter)

fig, axes = plt.subplots(1, 4)

for i in range(4):

    img = images[i].permute(1, 2, 0)

    axes[i].imshow(img)

    axes[i].set_title(labels[i].item())

    axes[i].axis("off")

plt.show()
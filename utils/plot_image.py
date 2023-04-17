import numpy as np
import matplotlib.pyplot as plt

def plot(imgs:list, titles:list, shape):
    fig = plt.figure()
    for idx in range(len(imgs)):
        ax = fig.add_subplot(*shape, idx+1)
        ax.imshow(imgs[idx])
        ax.set_title(titles[idx])
        ax.axis("off")
    plt.show()
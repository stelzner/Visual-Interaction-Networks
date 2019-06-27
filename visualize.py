import os
import matplotlib.pyplot as plt
import visdom
import numpy as np
import imageio

vis = visdom.Visdom()


def plot_positions(xy, img_folder, prefix, save=True, size=10):
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)
    fig_num = len(xy)
    mydpi = 100
    fig = plt.figure(figsize=(128/mydpi, 128/mydpi))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks([])
    plt.yticks([])

    color = ['r', 'b', 'g', 'k', 'y', 'm', 'c']
    for i in range(fig_num):
        for j in range(len(xy[0])):
            plt.scatter(xy[i, j, 1], xy[i, j, 0],
                        c=color[j % len(color)], s=size, alpha=(i+1)/fig_num)

    if save:
        fig.savefig(img_folder+prefix+".pdf", dpi=mydpi, transparent=True)
        vis.matplot(fig)

    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return image


def animate(states, img_folder, prefix):
    images = []
    for i in range(len(states)):
        images.append(plot_positions(states[i:i + 1], img_folder,
                                     prefix, save=False, size=270))
    imageio.mimsave(img_folder+prefix+'.gif', images, fps=24)

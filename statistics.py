import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

o1 = "opacity_anim.mp4"

def make_animation(iterations_id, file_name, fig_title_base, x_label, out_name):
    bins = np.load(file_name)
    bins_lows = np.arange(start=0, stop=1.01, step=0.01)
    bins_tops = np.arange(start=0.01, stop=1.01, step=0.01)

    bins_names = list(map(lambda x: str(x[0])[:4] + "-" + str(x[1])[:4], zip(bins_lows, bins_tops)))

    fig = plt.figure(figsize=(18, 10))
    ax = fig.add_subplot(111)
    plots = []
    for i, iteration in enumerate(bins):
        ttl = plt.text(0.5, 1.01, fig_title_base + f"Iteration {iterations_id[i]}", horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes)
        txt = ax.text(iterations_id[i], iterations_id[i], iterations_id[i])
        # ax.text(10, 90, f"Radiuses distribution of circles placed into background.\nIteration {iterations_id[i]}")
        plt.xticks(rotation=30)
        bar = ax.bar(bins_names[:30], iteration[:30], color="dodgerblue")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Number of circles")
        plots.append([b for b in bar] + [txt, ttl])

    anim = animation.ArtistAnimation(fig, plots, interval=50, repeat=False, blit=False)
    anim.save(out_name)

iterations_id = np.load("iteration_update.npy")
files = ["opacity_bins.npy", "radius_bins.npy"]
titles = ["Opacity distribution of circles placed into background.\n", "Radiuses distribution of circles placed into background.\n"]
x_labels = ["Opacity", "Percentage of target image diagonal"]
out_names = [ "opacity_anim.mp4", "radiuses_anim.mp4"]


for file_name, title, x_label, out_name in zip(files, titles, x_labels, out_names):
    make_animation(iterations_id, file_name, title, x_label, out_name)

# bins = np.load("radius_bins.npy")
# bins_lows = np.arange(start=0, stop=1.01, step=0.01)
# bins_tops = np.arange(start=0.01, stop=1.01, step=0.01)

# bins_names = list(map(lambda x: str(x[0])[:4] + "-" + str(x[1])[:4], zip(bins_lows, bins_tops)))
# iterations_id = np.load("iteration_update.npy")

# fig = plt.figure(figsize=(18, 10))
# ax = fig.add_subplot(111)
# plots = []
# for i, iteration in enumerate(bins):
#     ttl = plt.text(0.5, 1.01, f"Radiuses distribution of circles placed into background.\nIteration {iterations_id[i]}", horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes)
#     txt = ax.text(iterations_id[i], iterations_id[i], iterations_id[i])
#     # ax.text(10, 90, f"Radiuses distribution of circles placed into background.\nIteration {iterations_id[i]}")
#     plt.xticks(rotation=30)
#     bar = ax.bar(bins_names[:30], iteration[:30], color="dodgerblue")
#     ax.set_xlabel("Percentage of target image diagonal")
#     ax.set_ylabel("Number of circles")
#     plots.append([b for b in bar] + [txt, ttl])

# anim = animation.ArtistAnimation(fig, plots, interval=50, repeat=False, blit=False)
# anim.save("radiuses_anim.mp4")


# bins = np.load("opacity_bins.npy")
# bins_lows = np.arange(start=0, stop=1.01, step=0.01)
# bins_tops = np.arange(start=0.01, stop=1.01, step=0.01)

# bins_names = list(map(lambda x: str(x[0])[:4] + "-" + str(x[1])[:4], zip(bins_lows, bins_tops)))
# iterations_id = np.load("iteration_update.npy")

# fig = plt.figure(figsize=(18, 10))
# ax = fig.add_subplot(111)
# plots = []
# for i, iteration in enumerate(bins):
#     ttl = plt.text(0.5, 1.01, f"Opacity distribution of circles placed into background.\nIteration {iterations_id[i]}", horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes)
#     txt = ax.text(iterations_id[i], iterations_id[i], iterations_id[i])
#     # ax.text(10, 90, f"Radiuses distribution of circles placed into background.\nIteration {iterations_id[i]}")
#     plt.xticks(rotation=30)
#     bar = ax.bar(bins_names[:30], iteration[:30], color="dodgerblue")
#     ax.set_xlabel("Opacity")
#     ax.set_ylabel("Number of circles")
#     plots.append([b for b in bar] + [txt, ttl])

# anim = animation.ArtistAnimation(fig, plots, interval=50, repeat=False, blit=False)
# anim.save("opacity_anim.mp4")
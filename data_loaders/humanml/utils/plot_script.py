import math
from textwrap import wrap

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage


def list_cut_average(ll, intervals):
    if intervals == 1:
        return ll

    bins = math.ceil(len(ll) * 1.0 / intervals)
    ll_new = []
    for i in range(bins):
        l_low = intervals * i
        l_high = l_low + intervals
        l_high = l_high if l_high < len(ll) else len(ll)
        ll_new.append(np.mean(ll[l_low:l_high]))
    return ll_new


def plot_3d_motion(
    save_path,
    kinematic_tree,
    joints,
    title,
    dataset,
    figsize=(3, 3),
    fps=120,
    radius=3,
    vis_mode="default",
    gt_frames=None,
):
    matplotlib.use("Agg")
    if gt_frames is None:
        gt_frames = []

    title_per_frame = isinstance(title, list)
    wrap_width = 14
    if title_per_frame:
        assert len(title) == len(joints), "Title length should match the number of frames"
        title = ["\n".join(wrap(s, wrap_width)) for s in title]
    else:
        title = "\n".join(wrap(str(title), wrap_width))

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3.0, radius * 2 / 3.0])
        ax.grid(False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz],
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    data = joints.copy().reshape(len(joints), -1, 3)

    if dataset == "kit":
        data *= 0.003
    elif dataset in ["humanml", "twen"]:
        data *= 1.3
    elif dataset in ["humanact12", "uestc"]:
        data *= -1.5

    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    # Matplotlib 3.9+ needs explicit 3D axis registration.
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(right=0.68)
    title_text = fig.text(0.70, 0.5, "", va="center", ha="left", fontsize=9)
    init()

    mins = data.min(axis=0).min(axis=0)
    maxs = data.max(axis=0).max(axis=0)

    colors_blue = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]
    colors_orange = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]
    colors = colors_orange
    if vis_mode == "upper_body":
        colors[0] = colors_blue[0]
        colors[1] = colors_blue[1]
    elif vis_mode == "gt":
        colors = colors_blue

    n_frames = data.shape[0]
    height_offset = mins[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    global_min = data.min(axis=(0, 1))
    global_max = data.max(axis=(0, 1))
    range_eps = 1e-6
    x_lims = (global_min[0], global_max[0])
    y_lims = (0, global_max[1])
    z_lims = (global_min[2], global_max[2])
    box_aspect = (
        max(x_lims[1] - x_lims[0], range_eps),
        max(y_lims[1] - y_lims[0], range_eps),
        max(z_lims[1] - z_lims[0], range_eps),
    )

    def update(t):
        idx = min(n_frames - 1, int(t * fps))
        ax.clear()
        ax.set_xlim3d(*x_lims)
        ax.set_ylim3d(*y_lims)
        ax.set_zlim3d(*z_lims)
        ax.set_box_aspect(box_aspect)
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5

        if title_per_frame:
            frame_title = title[idx]
        else:
            frame_title = title
        frame_title += f"\n[{idx}]"
        title_text.set_text(frame_title)

        plot_xzPlane(
            mins[0] - trajec[idx, 0],
            maxs[0] - trajec[idx, 0],
            0,
            mins[2] - trajec[idx, 1],
            maxs[2] - trajec[idx, 1],
        )

        used_colors = colors_blue if idx in gt_frames else colors
        for i, (chain, color) in enumerate(zip(kinematic_tree, used_colors)):
            linewidth = 4.0 if i < 5 else 2.0
            ax.plot3D(
                data[idx, chain, 0],
                data[idx, chain, 1],
                data[idx, chain, 2],
                linewidth=linewidth,
                color=color,
            )

        plt.axis("off")
        ax.set_axis_off()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        return mplfig_to_npimage(fig)

    ani = VideoClip(update, duration=n_frames / fps)
    plt.close()
    return ani

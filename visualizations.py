"""
Methods for data visualization
"""
import os
import numpy as np
import torch
from matplotlib import pyplot as plt
import imageio
from webcolors import name_to_rgb
import torch.nn.functional as F
from torchvision.utils import draw_segmentation_masks


COLORS = ["blue", "green", "olive", "red", "yellow", "purple", "orange", "cyan",
          "brown", "pink", "darkorange", "goldenrod", "forestgreen", "springgreen",
          "aqua", "royalblue", "navy", "darkviolet", "plum", "magenta", "slategray",
          "maroon", "gold", "peachpuff", "silver", "aquamarine", "indianred", "greenyellow",
          "darkcyan", "sandybrown"]


def make_gif(frames, savepath, n_seed=4):
    """ Making a GIF with the frames """
    with imageio.get_writer(savepath, mode='I') as writer:
        for i, frame in enumerate(frames):
            up_frame = F.upsample(frame.unsqueeze(0), scale_factor=2)[0]  # to make it larger
            up_frame = up_frame.permute(1, 2, 0).cpu().detach().clamp(0, 1)
            disp_frame = add_border(up_frame, color="green") if i < n_seed else add_border(up_frame, color="red")
            writer.append_data(disp_frame)


def add_border(x, color, pad=2, extend=True):
    """
    Adds colored border to frames.

    Args:
    -----
    x: numpy array
        image to add the border to
    color: string
        Color of the border
    pad: integer
    extend: boolean
        Extend the image by padding or not.
    number of pixels to pad each side
    """
    common, (_, H, W) = x.shape[:-3], x.shape[-3:]
    px_h, px_w = (H + 2 * pad, W + 2 * pad) if extend else (H, W)
    px = torch.zeros((*common, 3, px_h, px_w))
    color_rgb = name_to_rgb(color)
    for c in range(3):
        px[..., c, :, :] = color_rgb[c] / 255.

    p = 0 if extend else pad
    x_ = x[..., p:H-p, p:W-p]
    for c in range(3):
        px[..., c, pad:px_h-pad, pad:px_w-pad] = x_[..., c, :, :]
    return px


def visualize_sequence(sequence, savepath=None, seq_id=None, gt=False, add_title=True,
                       add_axis=False, n_cols=10, **kwargs):
    """
    Visualizing a grid with all frames from a sequence
    """
    if seq_id is not None:
        suffix = "_gt" if gt is True else ""
        seq_path = os.path.dirname(savepath) + f"/seq_{seq_id}{suffix}/"
        os.makedirs(seq_path, exist_ok=True)
    n_frames = sequence.shape[0]
    n_rows = int(np.ceil(n_frames / n_cols))
    fig, ax = plt.subplots(n_rows, n_cols)
    fig.set_size_inches(3*n_cols, 3*n_rows)

    for i in range(n_frames):
        row, col = i // n_cols, i % n_cols
        a = ax[row, col] if n_frames > n_cols else ax[col]
        img = sequence[i].permute(1, 2, 0).cpu().detach()
        img = torch.squeeze(img, dim=2)
        img = img.clamp(0, 1).numpy()
        cmap = "gray" if img.ndim == 2 else None
        a.imshow(img, cmap=cmap)
        if (add_title):
            a.set_title(f"Frame {i}")
        if seq_id is not None:
            plt.imsave(seq_path + f"img_{i}.png", img)

    for i in range(n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        a = ax[row, col] if n_frames > n_cols else ax[col]
        if (not add_axis):
            a.axis("off")

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
        fig.clear()
    return


def overlay_segmentations(frames, segmentations, colors, num_classes, alpha=0.7):
    """
    Overlaying the segmentation on a sequence of images
    """
    if frames.max() <= 1:
        frames = frames * 255
    frames = frames.to(torch.uint8)

    imgs = []
    for frame, segmentation in zip(frames, segmentations):
        img = overlay_segmentation(frame, segmentation, colors, num_classes, alpha)
        imgs.append(img)
    imgs = torch.stack(imgs)
    return imgs


def overlay_segmentation(img, segmentation, colors, num_classes, alpha=0.7):
    """
    Overlaying the segmentation on an image
    """
    if img.max() <= 1:
        img = img * 255
    img = img.to(torch.uint8)
    seg_masks = (segmentation[0] == torch.arange(num_classes)[:, None, None].to(segmentation.device))
    img_with_seg = draw_segmentation_masks(
            img,
            masks=seg_masks,
            alpha=alpha,
            colors=colors
        )
    return img_with_seg / 255



#

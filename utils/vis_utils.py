import cv2
import matplotlib.pyplot as plt
import numpy as np


# plt.switch_backend('agg')

def visualize_sequences(batch, seq_len, return_fig=True):
    """
    visualize a sequence (imgs or flows)
    """
    sequences = []
    if seq_len == 0:
        return
    channels_per_frame = batch.shape[-1] // seq_len
    for i in range(batch.shape[0]):
        cur_sample = batch[i]  # [H,W,channels_per_frame * seq_len]
        if channels_per_frame == 1:
            sequence = [cur_sample[:, :, j * channels_per_frame:(j + 1) * channels_per_frame][:, :, ::-1]
                        for j in range(seq_len)]
        elif channels_per_frame == 2:
            pass
        elif channels_per_frame == 3:
            # to RGB
            sequence = [cur_sample[:, :, j * channels_per_frame:(j + 1) * channels_per_frame][:, :, ::-1]
                        for j in range(seq_len)]
        elif channels_per_frame == 5:
            pass
        elif channels_per_frame == 6:
            # to RGB/pose
            sequence_frame = [cur_sample[:, :, j * channels_per_frame:((j + 1) * channels_per_frame) - 3][:, :, ::-1]
                        for j in range(seq_len)] 
            sequence_pose = [cur_sample[:, :, (j * channels_per_frame) + 3:(j + 1) * channels_per_frame][:, :, ::-1]
                        for j in range(seq_len)]
            sequence = np.concatenate((sequence_frame, sequence_pose), axis=2)
        elif channels_per_frame == 8:
            pass
        sequences.append(np.hstack(sequence))
    sequences = np.vstack(sequences)

    if return_fig:
        fig = plt.figure()
        plt.imshow(sequences)
        return fig
    else:
        return sequences

def mse2img(mseimg):
    # img: h,w,c
    mseimg = mseimg.transpose(1,2,0)
    mseimg = (mseimg - np.min(mseimg)) / (np.max(mseimg)-np.min(mseimg))
    mseimg = mseimg * 255
    mseimg = mseimg.astype(dtype=np.uint8)
    color_mseimg = cv2.applyColorMap(mseimg, cv2.COLORMAP_JET)
    return color_mseimg

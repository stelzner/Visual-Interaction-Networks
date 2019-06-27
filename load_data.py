from torch.utils.data import Dataset
from scipy.io import loadmat
import numpy as np
import torch


def clips_from_episodes(images, labels, visible_l, rollout_l, step):
    """
    Rearrange episodic observations into shorter clips
    :param images: Episodes of images of shape (n, fr, c, h, w)
    :param labels: Episodes of accompanying data for the images (n, fr, obj, d)
    :param visible_l: Number of frames in each clip
    :param rollout_l: Number of future frames for which labels are returned
    :param step: Stepsize for taking frames from the given episodes
    :return: A number of shorter clips (_, visible_l, c, h, w),
             the corresponding labels (_, visible_l, obj, d),
             and future labels (_, rollout_l, obj, d).
    """
    (num_episodes, num_frames, height, width, channels) = images.shape
    num_obj = labels.shape[-2]

    clips_per_episode = num_frames - (rollout_l + visible_l) * step + 1
    num_clips = num_episodes * clips_per_episode

    clips = np.zeros((num_clips, visible_l, height, width, channels))
    present_labels = np.zeros((num_clips, visible_l - 2, num_obj, 4))
    future_labels = np.zeros((num_clips, rollout_l, num_obj, 4))

    for i in range(num_episodes):
        for j in range(clips_per_episode):
            clip_idx = i * clips_per_episode + j

            end_visible = j + visible_l * step
            end_rollout = end_visible + rollout_l * step

            clips[clip_idx] = images[i, j:end_visible:step]
            present_labels[clip_idx] = labels[i, j + 2*step:end_visible:step]
            future_labels[clip_idx] = labels[i, end_visible:end_rollout:step]

    # shuffle
    perm_idx = np.random.permutation(num_clips)
    return clips[perm_idx], present_labels[perm_idx], future_labels[perm_idx]


class VinDataset(Dataset):
    def __init__(self, config, transform=None, test=False):
        self.config = config
        self.transform = transform

        if test:
            data = loadmat(config.testdata)
        else:
            data = loadmat(config.traindata)

        self.total_img = data['X'][:config.num_episodes]
        # Transpose, as PyTorch images have shape (c, h, w)
        self.total_img = np.transpose(self.total_img, (0, 1, 4, 2, 3))
        self.total_data = data['y'][:config.num_episodes]
        self.total_data[..., :2] /= 10
        self.total_data[..., 2:] *= 2

        num_eps, num_frames = self.total_img.shape[0:2]
        clips_per_ep = num_frames - ((config.num_visible +
                                     config.num_rollout) *
                                     config.frame_step) + 1

        idx_ep, idx_fr = np.meshgrid(list(range(num_eps)),
                                     list(range(clips_per_ep)))

        self.idxs = np.reshape(np.stack([idx_ep, idx_fr], 2), (-1, 2))

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        conf = self.config
        step = conf.frame_step

        i, j = self.idxs[idx, 0], self.idxs[idx, 1]

        end_visible = j + conf.num_visible * step
        end_rollout = end_visible + conf.num_rollout * step
        image = self.total_img[i, j:end_visible:step]
        present = self.total_data[i, j + 2 * step:end_visible:step]
        future = self.total_data[i, end_visible:end_rollout:step]

        sample = {'image': torch.from_numpy(image),
                  'future_labels': torch.from_numpy(future),
                  'present_labels': torch.from_numpy(present)}

        return sample

import os

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

from load_data import VinDataset
from visualize import plot_positions, animate


class Trainer:
    def __init__(self, config, net):
        self.net = net
        self.params = net.parameters()
        self.initial_values = {}
        self.config = config

        train_dataset = VinDataset(self.config)
        self.dataloader = DataLoader(train_dataset,
                                     batch_size=self.config.batch_size,
                                     shuffle=True,
                                     num_workers=4,
                                     drop_last=True)

        self.test_dataset = VinDataset(self.config,
                                       test=True)
        self.test_dataloader = DataLoader(self.test_dataset,
                                          batch_size=self.config.batch_size,
                                          shuffle=True,
                                          num_workers=4,
                                          drop_last=True)

        self.optimizer = optim.Adam(self.net.parameters(), lr=0.0005)
        if config.load:
            self.load()

    def save(self):
        torch.save(self.net.state_dict(), os.path.join(
            self.config.checkpoint_dir, "checkpoint"))
        print('Parameters saved')

    def load(self):
        try:
            self.net.load_state_dict(torch.load(os.path.join(
                self.config.checkpoint_dir, "checkpoint")))
            print('Parameters loaded')
        except RuntimeError:
            print('Loading parameters failed, training from scratch...')

    def compute_loss(self, present_labels, future_labels, recons, preds):
        loss = nn.MSELoss()
        df = self.config.discount_factor
        pred_loss = 0.0
        for delta_t in range(0, self.config.num_rollout):
            pred_loss += (df ** (delta_t + 1)) * \
                loss(preds[:, delta_t], future_labels[:, delta_t])

        recon_loss = loss(recons, present_labels)
        total_loss = pred_loss + recon_loss

        return total_loss, pred_loss, recon_loss

    def train(self):
        step_counter = 0
        num_rollout = self.config.num_rollout
        for epoch in range(100):
            print("testing................")
            self.test()
            for i, data in enumerate(self.dataloader, 0):
                step_counter += 1
                images = data['image']
                present_labels = data['present_labels']
                future_labels = data['future_labels']

                images = images.cuda()
                present_labels = present_labels.cuda()
                future_labels = future_labels.cuda()

                if self.config.visual:
                    vin_input = images
                else:
                    vin_input = present_labels

                self.optimizer.zero_grad()
                state_pred, state_recon = self.net(vin_input,
                                                   num_rollout=num_rollout,
                                                   visual=self.config.visual)

                total_loss, pred_loss, recon_loss = \
                    self.compute_loss(present_labels, future_labels,
                                      state_recon, state_pred)

                total_loss.backward()
                self.optimizer.step()

                # print loss
                if step_counter % 20 == 0:
                    print('{:5d} {:5f} {:5f} {:5f}'.format(step_counter,
                                                           total_loss.item(),
                                                           recon_loss.item(),
                                                           pred_loss.item()))
                # Draw example
                if step_counter % 200 == 0:
                    real = torch.cat([present_labels[0], future_labels[0]])
                    simu = torch.cat([state_recon[0], state_pred[0]]).detach()
                    plot_positions(real, self.config.img_folder, 'real')
                    plot_positions(simu, self.config.img_folder, 'rollout')
                # Save parameters
                if (step_counter + 1) % 1000 == 0:
                    self.save()

            print("epoch ", epoch, " Finished")
        print('Finished Training')

    def test(self):
        total_loss = 0.0
        for i, data in enumerate(self.test_dataloader, 0):
            images, future_labels, present_labels = \
                data['image'], data['future_labels'], data['present_labels']

            images = images.cuda()
            present_labels = present_labels.cuda()
            future_labels = future_labels.cuda()

            vin_input = images if self.config.visual else present_labels

            pred, recon = self.net(vin_input,
                                   num_rollout=self.config.num_rollout,
                                   visual=self.config.visual)

            total_loss, pred_loss, recon_loss = \
                self.compute_loss(present_labels, future_labels,
                                  recon, pred)

        print('total test loss {:5f}'.format(total_loss.item()))

        # Create one long rollout and save it as an animated GIF
        total_images = self.test_dataset.total_img
        total_labels = self.test_dataset.total_data
        step = self.config.frame_step
        visible = self.config.num_visible
        batch_size = self.config.batch_size

        long_rollout_length = self.config.num_frames // step - visible

        if self.config.visual:
            vin_input = total_images[:batch_size, :visible*step:step]
        else:
            vin_input = total_labels[:batch_size, 2*step:visible*step:step]

        vin_input = torch.tensor(vin_input).cuda()

        pred, recon = self.net(vin_input, long_rollout_length,
                               visual=self.config.visual)

        simu_rollout = pred[0].detach().cpu().numpy()
        simu_recon = recon[0].detach().cpu().numpy()
        simu = np.concatenate((simu_recon, simu_rollout), axis=0)

        # Saving
        print("Make GIFs")
        animate(total_labels[0, 2:], self.config.img_folder, 'real')
        animate(simu, self.config.img_folder, 'rollout')
        print("Done")

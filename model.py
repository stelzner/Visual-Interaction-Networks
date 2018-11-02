import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()
        self.config = config
        self.pool = nn.MaxPool2d(2, 2)
        self.x_coord, self.y_coord = self.construct_coord_dims()
        cl = config.cl

        # Visual Encoder Modules
        self.conv1 = nn.Conv2d(config.channels * 2 + 2, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv4 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv5 = nn.Conv2d(32, 32, 3, padding=1)
        # shared linear layer to get pair codes of shape N_obj*cl
        self.fc1 = nn.Linear(32, 3 * cl)

        # shared MLP to encode pairs of pair codes as state codes N_obj*cl
        self.fc2 = nn.Linear(cl * 2, cl)
        self.fc3 = nn.Linear(cl, cl)
        # end of visual encoder

        # Interaction Net Core Modules
        # Self-dynamics MLP
        self.self_cores = nn.ModuleList()
        for i in range(3):
            self.self_cores.append(nn.ModuleList())
            self.self_cores[i].append(nn.Linear(cl, cl).double().cuda())
            self.self_cores[i].append(nn.Linear(cl, cl).double().cuda())

        # Relation MLP
        self.rel_cores = nn.ModuleList()
        for i in range(3):
            self.rel_cores.append(nn.ModuleList())
            self.rel_cores[i].append(nn.Linear(cl * 2, 2 * cl).double().cuda())
            self.rel_cores[i].append(nn.Linear(2 * cl, cl).double().cuda())
            self.rel_cores[i].append(nn.Linear(cl, cl).double().cuda())

        # Affector MLP
        self.affector = nn.ModuleList()
        for i in range(3):
            self.affector.append(nn.ModuleList())
            self.affector[i].append(nn.Linear(cl, cl).double().cuda())
            self.affector[i].append(nn.Linear(cl, cl).double().cuda())
            self.affector[i].append(nn.Linear(cl, cl).double().cuda())

        # Core output MLP
        self.out = nn.ModuleList()
        for i in range(3):
            self.out.append(nn.ModuleList())
            self.out[i].append(nn.Linear(cl + cl, cl).double().cuda())
            self.out[i].append(nn.Linear(cl, cl).double().cuda())

        # Aggregator MLP for aggregating core predictions
        self.aggregator1 = nn.Linear(cl * 3, cl)
        self.aggregator2 = nn.Linear(cl, cl)

        # decoder mapping state codes to actual states
        self.state_decoder = nn.Linear(cl, 4)
        # encoder for the non-visual case
        self.state_encoder = nn.Linear(4, cl)

    def construct_coord_dims(self):
        """
        Build a meshgrid of x, y coordinates to be used as additional channels
        """
        x = np.linspace(0, 1, self.config.width)
        y = np.linspace(0, 1, self.config.height)
        xv, yv = np.meshgrid(x, y)
        xv = np.reshape(xv, [1, 1, self.config.height, self.config.width])
        yv = np.reshape(yv, [1, 1, self.config.height, self.config.width])
        x_coord = Variable(torch.from_numpy(xv)).cuda()
        y_coord = Variable(torch.from_numpy(yv)).cuda()
        x_coord = x_coord.expand(self.config.batch_size * 5, -1, -1, -1)
        y_coord = y_coord.expand(self.config.batch_size * 5, -1, -1, -1)
        return x_coord, y_coord

    def core(self, s, core_idx):
        """
        Applies an interaction network core
        :param s: A state code of shape (n, o, cl)
        :param core_idx: The index of the set of parameters to apply (0, 1, 2)
        :return: Prediction of a future state code (n, o, cl)
        """
        objects = [s[:, i] for i in range(3)]

        self_sd_h1 = F.relu(self.self_cores[core_idx][0](s))
        self_dynamic = self.self_cores[core_idx][1](self_sd_h1) + self_sd_h1

        rel_combination = []
        for i in range(6):
            row_idx = i // 2
            # pick the two other objects
            col_idx = (row_idx + 1 + (i % 2)) % 3
            rel_combination.append(
                torch.cat([objects[row_idx], objects[col_idx]], 1))
        # 6 combinations of the 3 objects, (n, 6, 2*cl)
        rel_combination = torch.stack(rel_combination, 1)
        rel_sd_h1 = F.relu(self.rel_cores[core_idx][0](rel_combination))
        rel_sd_h2 = F.relu(self.rel_cores[core_idx][1](rel_sd_h1))
        rel_factors = self.rel_cores[core_idx][2](rel_sd_h2) + rel_sd_h2
        obj1 = rel_factors[:, 0] + rel_factors[:, 1]
        obj2 = rel_factors[:, 2] + rel_factors[:, 3]
        obj3 = rel_factors[:, 4] + rel_factors[:, 5]
        # relational dynamics per object, (n, o, cl)
        rel_dynamic = torch.stack([obj1, obj2, obj3], 1)
        # total dynamics
        dynamic_pred = self_dynamic + rel_dynamic

        aff1 = F.relu(self.affector[core_idx][0](dynamic_pred))
        aff2 = F.relu(self.affector[core_idx][1](aff1) + aff1)
        aff3 = self.affector[core_idx][2](aff2)

        aff_s = torch.cat([aff3, s], 2)
        out1 = F.relu(self.out[core_idx][0](aff_s))
        out2 = self.out[core_idx][1](out1) + out1
        return out2

    def frames_to_states(self, frames):
        """
        Apply visual encoder
        :param frames: Groups of six input frames of shape (n, 6, c, w, h)
        :return: State codes of shape (n, 4, o, cl)
        """
        batch_size = self.config.batch_size
        cl = self.config.cl
        num_obj = self.config.num_obj

        pairs = []
        for i in range(frames.shape[1] - 1):
            # pair consecutive frames (n, 2c, w, h)
            pair = torch.cat((frames[:, i], frames[:, i+1]), 1)
            pairs.append(pair)

        num_pairs = len(pairs)
        pairs = torch.cat(pairs, 0)
        # add coord channels (n * num_pairs, 2c + 2, w, h)
        pairs = torch.cat([pairs, self.x_coord, self.y_coord], dim=1)

        # apply ConvNet to pairs
        ve_h1 = F.relu(self.conv1(pairs))
        ve_h1 = self.pool(ve_h1)
        ve_h2 = F.relu(self.conv2(ve_h1))
        ve_h2 = self.pool(ve_h2)
        ve_h3 = F.relu(self.conv3(ve_h2))
        ve_h3 = self.pool(ve_h3)
        ve_h4 = F.relu(self.conv4(ve_h3))
        ve_h4 = self.pool(ve_h4)
        ve_h5 = F.relu(self.conv5(ve_h4))
        ve_h5 = self.pool(ve_h5)

        # pooled to 1x1, 32 channels: (n * num_pairs, 32)
        encoded_pairs = torch.squeeze(ve_h5)
        # final pair encoding (n * num_pairs, o, cl)
        encoded_pairs = self.fc1(encoded_pairs)
        encoded_pairs = encoded_pairs.view(batch_size * num_pairs, num_obj, cl)
        # chunk pairs encoding, each is (n, o, cl)
        encoded_pairs = torch.chunk(encoded_pairs, num_pairs)

        triples = []
        for i in range(num_pairs - 1):
            # pair consecutive pairs to obtain encodings for triples
            triple = torch.cat([encoded_pairs[i], encoded_pairs[i+1]], 2)
            triples.append(triple)

        # the triples together, i.e. (n, num_pairs - 1, o, 2 * cl)
        triples = torch.stack(triples, 1)
        # apply MLP to triples
        shared_h1 = F.relu(self.fc2(triples))
        state_codes = self.fc3(shared_h1)
        return state_codes

    def forward(self, x, num_rollout=8, visual=True):
        """
        Rollout a given sequence of observations using the model
        :param x: The given sequence of observations.
        If visual is True, it should be images of shape (n, 6, c, w, h),
                  otherwise states of shape (n, 4, o, 4).
        :param num_rollout: The number of future states to be predicted
        :param visual: Boolean determining the type of input
        :return: rollout_states: predicted future states (n, roll_len, o, 4)
                 present_states: predicted states at the time of
                                 the given observations, (n, 4, o, 4)
        """
        # get encoded states
        if visual:
            state_codes = self.frames_to_states(x)
        else:
            state_codes = self.state_encoder(x)
        # the 4 state codes (n, o, cl)
        s1, s2, s3, s4 = [state_codes[:, i] for i in range(4)]
        rollouts = []
        for i in range(num_rollout):
            # use cores to predict next state using delta_t = 1, 2, 4
            c1 = self.core(s4, 0)
            c2 = self.core(s3, 1)
            c4 = self.core(s1, 2)
            all_c = torch.cat([c1, c2, c4], 2)
            aggregator1 = F.relu(self.aggregator1(all_c))
            state_prediction = self.aggregator2(aggregator1)
            rollouts.append(state_prediction)
            s1, s2, s3, s4 = s2, s3, s4, state_prediction
        rollouts = torch.stack(rollouts, 1)

        present_states = self.state_decoder(state_codes)
        rollout_states = self.state_decoder(rollouts)

        return rollout_states, present_states

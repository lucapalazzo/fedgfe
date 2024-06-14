# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import copy
import torch
import numpy as np
import time
import wandb
from flcore.clients.clientRewind import clientRewind
from utils.privacy import *


class clientAVGRew(clientRewind):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

    def train(self):

        if self.optimizer.param_groups[0]['lr'] != self.learning_rate:
            print(f"Client {self.id} resetting learning rate from {self.optimizer.param_groups[0]['lr']} to  {self.learning_rate}")
            self.optimizer.param_groups[0]['lr'] = self.learning_rate
        last_lr = self.learning_rate
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()

        # differential privacy
        if self.privacy:
            model_origin = copy.deepcopy(self.model)
            self.model, self.optimizer, trainloader, privacy_engine = \
                initialize_dp(self.model, self.optimizer, trainloader, self.dp_sigma)
        
        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)
        rewind_epochs, local_epochs, rewind_nodes_count = self.prepare_rewind(max_local_epochs)

        for epoch in range(local_epochs):
            if ( ( self.rewind_strategy == "halfway" or self.rewind_strategy == "interval" or self.rewind_strategy == "atend_pre"  ) and len(self.rewind_previous_node) > 0 ):
                self.rewind(epoch, max_local_epochs, rewind_epochs, rewind_nodes_count)

            for i, (x, y) in enumerate(trainloader):
                if self.check_batch(x, y) == False:
                    print (f"Client {self.id} batch {i} has wrong shape, skipping {x.shape} {y.shape}")
                    continue

                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if self.learning_rate_schedule:
                self.learning_rate_scheduler.step(loss)
                new_lr = self.learning_rate_scheduler.get_last_lr()[0]
                if new_lr != self.learning_rate and new_lr != last_lr:
                    last_lr = new_lr
                    print(f"Client {self.id} learning rate: {self.learning_rate_scheduler.get_last_lr()} loss {loss}")

            if ( self.rewind_strategy == "atend" and rewind_nodes_count ):
                self.rewind(epoch, max_local_epochs, rewind_epochs, rewind_nodes_count)

        if len(self.rewind_previous_node) > 0:
            rewind_node = self.rewind_previous_node[-1]
            local_loss, rw_loss = self.rewind_train_metrics(rewind_node)
            if not self.no_wandb:
                wandb.log({f"train/model_{self.model.id}/atend_loss_on_local": local_loss, "round": self.round})
                wandb.log({f"train/model_{self.model.id}/atend_loss_on_previous": rw_loss, "round": self.round})

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        if self.privacy:
            eps, DELTA = get_dp_params(privacy_engine)
            print(f"Client {self.id}", f"epsilon = {eps:.2f}, sigma = {DELTA}")

            for param, param_dp in zip(model_origin.parameters(), self.model.parameters()):
                param.data = param_dp.data.clone()
            self.model = model_origin
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

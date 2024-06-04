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

from sklearn.metrics import balanced_accuracy_score
import torch
import os
import numpy as np
import h5py
import copy
import time
import random
import wandb
from utils.data_utils import read_client_data
from utils.dlg import DLG
import torch.nn.functional as F
import uuid
# from ignite.metrics import ConfusionMatrix
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay

class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.uuid = uuid.uuid4()
        self.save_folder_name = str(self.uuid)

        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.dataset_limit = args.dataset_limit
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.top_cnt = 100
        self.auto_break = args.auto_break
        self.reduce_memory_footprint = args.reduce_memory_footprint

        self.routing_scored_average = args.routing_scored_average

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []
        self.loss_weighted = args.loss_weighted

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.dlg_eval = args.dlg_eval
        self.dlg_gap = args.dlg_gap
        self.batch_num_per_client = args.batch_num_per_client

        self.num_new_clients = args.num_new_clients
        self.new_clients = []
        self.eval_new_clients = False
        self.fine_tuning_epoch_new = args.fine_tuning_epoch_new
        self.round = 0

        self.no_wandb = args.no_wandb
        self.gpus = list(map(int, args.device_ids.split(',')))

        self.round_test_stats = self.num_clients * [None]
        self.round_test_on_train_stats = self.num_clients * [None]
        self.round_train_stats = self.num_clients * [None]

    def save_checkpoint(self):
        if self.save_folder_name == None:
            self.save_folder_name = os.path.join(self.uuid)
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        
        # torch.save(self, os.path.join(self.save_folder_name, "server.pt"))

        for client in self.clients:
            torch.save(client.model, os.path.join(self.save_folder_name, "client_" + str(client.model.id) + ".pt"))
            # torch.save(client.node_data, os.path.join(self.save_folder_name, "client_data_" + str(client.id) + ".pt"))

    def get_routes(self):
        if self.routing is not None:
            return self.routing.get_routes()

        return self.routes
    
    def set_clients(self, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=train_slow, 
                            send_slow=send_slow)
            self.clients.append(client)
        self.define_metrics()

    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))

        return selected_clients

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.set_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def save_global_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def model_exists(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)
        
    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = os.path.join( self.save_folder_name, "results/" )
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)

    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))
    
    def define_metrics(self):
        if not self.no_wandb:
            print ( "Defining common metrics for all experiments, please CUSTOMIZE HERE you metrics ")
            wandb.define_metric(f"federation/acc_std", step_metric="round") 
            wandb.define_metric(f"federation/acc_std_on_train", step_metric="round") 
            wandb.define_metric(f"federation/Federation Test Accuracy Mean", step_metric="round")
            wandb.define_metric(f"federation/Federation Balanced Test Accuracy Mean", step_metric="round")
            for client in self.clients:
                wandb.define_metric(f"test/client_{client.id}/acc", step_metric="round")
                wandb.define_metric(f"test/client_{client.id}/bal", step_metric="round")
                for t in range(self.num_clients):
                    wandb.define_metric(f"test/model_{client.id}/round_test_acc_{t}", step_metric="round")
                    wandb.define_metric(f"test/model_{client.id}/round_test_acc_on_train_{t}", step_metric="round")
            #     wandb.define_metric(f"train/client_{client.id}/round_train_loss_{client.id}", step_metric="round")
            #     wandb.define_metric(f"test/client_{client.id}/round_test_acc_{client.id}", step_metric="round")
            #     if self.rewind_ratio > 0 or self.rewind_epochs > 0:
            #         wandb.define_metric(f"rewind/rewind_phase_loss_{client.id}", step_metric="round")

            #     for test_client in self.clients:
            #         if ( test_client.id != client.id):
            #             wandb.define_metric(f"train/client_{client.id}/round_train_loss_{client.id}_on_{test_client.id}", step_metric="round")
            #             wandb.define_metric(f"test/client_{client.id}/round_test_acc_{client.id}_on_{test_client.id}", step_metric="round")

    def test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
        
        test_num_samples = []
        test_tot_correct = []
        test_tot_auc = []

        test_y_t =[]
        test_y_p = []

        train_num_samples = []
        train_tot_correct = []
        train_tot_auc = []

        train_y_t =[]
        train_y_p = []
        test_clients_stats = self.num_clients * [None]
        train_clients_stats = self.num_clients * [None] 

        # metric = ConfusionMatrix(num_classes=10)
        # metric.attach(default_evaluator, 'cm')
        for client_index,c in enumerate(self.clients):
            test_client_stats = []
            train_client_stats = []
            if ( client_index < len(self.clients) - 1):
                self.clients[client_index+1].node_data.load_test_data(self.args.batch_size)
            for t in self.clients:
                test = []
                test_ct, test_ns, test_auc, test_y_true, test_y_prob = c.test_metrics(t)
                train_ct, train_ns, train_auc, train_y_true, train_y_prob = c.test_metrics(t, on_train=True)
                test_tot_correct.append(test_ct*1.0)
                test_tot_auc.append(test_auc*test_ns)
                test_num_samples.append(test_ns)
                test_y_t.append(test_y_true)
                test_y_p.append(test_y_prob)
                test.append(test_ct*1.0)
                test.append(test_auc*test_ns)
                test.append(test_ns)
                test.append(test_y_true)
                test.append(test_y_prob)
                test_client_stats.append(test)
                if self.reduce_memory_footprint == True:
                    t.node_data.unload_test_data()
                    t.node_data.unload_train_data()

                train =[]
                train_tot_correct.append(train_ct*1.0)
                train_tot_auc.append(train_auc*train_ns)
                train_num_samples.append(train_ns)
                train_y_t.append(train_y_true)
                train_y_p.append(train_y_prob)
                train.append(train_ct*1.0)
                train.append(train_auc*train_ns)
                train.append(train_ns)
                train.append(train_y_true)
                train.append(train_y_prob)
                train_client_stats.append(train)
            
            test_clients_stats[c.model.id] = test_client_stats
            train_clients_stats[c.model.id] = train_client_stats

            if ( client_index > 0 and self.reduce_memory_footprint == True):
                c.node_data.unload_test_data()
            

            
            # if not self.no_wandb:
            #     wandb.log({f'test_acc_{c.id}': ct*1.0/ns})

        ids = [c.id for c in self.clients]

        return ids, test_num_samples, test_tot_correct, test_tot_auc, test_y_t, test_y_p, test_clients_stats, train_clients_stats

    def train_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]
        
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)
            if not self.no_wandb:
                wandb.log({f'train_loss_{c.id}': cl*1.0/ns})

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    # evaluate selected clients
    def evaluate(self, acc=None, loss=None):

        stats = self.test_metrics()
        stats_train = self.train_metrics()

        round_test_stats = stats[6]
        round_test_on_train_stats = stats[7]
        self.round_test_stats.insert(self.round, round_test_stats)
        self.round_test_on_train_stats.insert(self.round, round_test_on_train_stats)
        self.round_train_stats.insert(self.round, stats_train)

        fed_test_acc = sum(stats[2])*1.0 / sum(stats[1])
        fed_test_auc = sum(stats[3])*1.0 / sum(stats[1])
        y_true_tot = np.concatenate(stats[4])
        y_prob_tot = np.concatenate(stats[5])
        y_prob_tot_sm = F.softmax(torch.tensor(y_prob_tot), dim=1).numpy()
        y_prob_tot_sm_am = np.argmax(y_prob_tot_sm, axis=1)
        y_trues = stats[4]
        y_probs = stats[5]

        # stats_index = 0
        # for y_true, y_prob in zip(y_trues, y_probs):
        #     client_index = stats_index // self.num_clients
        #     test_index = stats_index % self.num_clients
        #     y_prob_sm = F.softmax(torch.tensor(y_prob), dim=1).numpy()
        #     y_prob_sm_am = np.argmax(y_prob_sm, axis=1)
        #     # cm = confusion_matrix(y_true, y_prob_sm_am)
        #     # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(self.num_classes))
        #     # disp.plot()
        #     # destination_dir = str(self.uuid)
        #     # if ( not os.path.exists(destination_dir)):
        #     #     os.makedirs(destination_dir)
        #     # disp.figure_.savefig(f'{destination_dir}/confusion_matrix_{self.round}_{client_index}_{test_index}.png')
        #     # print(cm)
        #     print("balanced accuracy score: ", balanced_accuracy_score(y_true, y_prob_sm_am))
        #     print("accuracy score: ", sum(y_true == y_prob_sm_am) / len(y_true))
        #     stats_index += 1


        fed_test_acc_balanced = balanced_accuracy_score(y_true_tot, y_prob_tot_sm_am)

        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]

        for idx in range(len(stats[0])):
            client_id = stats[0][idx]
            self.data_log({f'test/client_{client_id}/acc': accs[idx], 'round' : self.round})
            y_true = stats[4][idx]
            y_prob = stats[5][idx]
            y_prob_sm = F.softmax(torch.tensor(y_prob), dim=1).numpy()
            y_prob_sm_am = np.argmax(y_prob_sm, axis=1)
            test_acc_balanced = balanced_accuracy_score(y_true,  y_prob_sm_am)
            self.data_log({f'test/client_{client_id}/bal_acc': test_acc_balanced, 'round' : self.round})

        
        if acc == None:
            self.rs_test_acc.append(fed_test_acc)
        else:
            acc.append(fed_test_acc)
        
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        # for stats in client_stats:
        #     for client_stats in stats:
        #         round_acc = client_stats[0]/client_stats[2]
        test_accuracies = []
        for model_id in range(len(round_test_stats)):
            model_stats = round_test_stats[model_id]
            model_accs = []
            for test_client_id in range(len(model_stats)):
                round_acc = model_stats[test_client_id][0]/model_stats[test_client_id][2]
                test_accuracies.append(round_acc)
                model_accs.append(round_acc)
                self.data_log({f"test/model_{model_id}/round_test_acc_{test_client_id}": round_acc, "round":self.round})
            round_acc_std = np.std(test_accuracies)
            self.data_log({f"test/model_{model_id}/test_std": round_acc_std, "round":self.round})
       
        test_accuracies_on_train = []
        for model_id in range(len(round_test_on_train_stats)):
            model_accs_on_train = []
            model_stats = round_test_on_train_stats[model_id]
            for test_client_id in range(len(model_stats)):
                round_acc = model_stats[test_client_id][0]/model_stats[test_client_id][2]
                test_accuracies_on_train.append(round_acc)
                model_accs_on_train.append(round_acc)
                self.data_log({f"test/model_{model_id}/round_test_acc_on_train_{test_client_id}": round_acc, "round":self.round})
            round_acc_on_train_std = np.std(test_accuracies_on_train)
            self.data_log({f"test/model_{model_id}/test_std_on_train": round_acc_on_train_std, "round":self.round})



        fed_acc_std = np.std(test_accuracies)
        fed_acc_std_on_train = np.std(test_accuracies_on_train)
        self.data_log({"federation/acc_std": fed_acc_std, "round":self.round})
        self.data_log({"federation/acc_std_on_train": fed_acc_std_on_train, "round":self.round})

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accuracy: {:.4f}".format(fed_test_acc))
        print("Averaged Balanced Test Accurancy : {:.4f}".format(fed_test_acc_balanced))
        
        self.data_log({"federation/Federation Test Accuracy Mean": fed_test_acc, "round":self.round})
        self.data_log({"federation/Federation Balanced Test Accuracy Mean": fed_test_acc_balanced, "round":self.round})
        
        print("Averaged Test AUC: {:.4f}".format(fed_test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accuracy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))

        # for client in self.clients:
        #     for test_client in self.clients:
        #         if ( client.model.id != test_client.model.id):
        #             acc, test_num, auc, y_true, y_prob = client.test_metrics_other(test_client)
        #             round_acc = acc/test_num
        #             # print("Accuracy of model from node %d on dataset from node %d test set accuracy %02f" % (client.id, test_client.id, round_acc ))
        #             self.data_log( {f"test/model_{client.id}/round_test_acc_{client.id}_on_{test_client.id}": round_acc, "round":self.round})
        self.round += 1

    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accurancy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt != None and div_value != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value != None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

    def call_dlg(self, R):
        # items = []
        cnt = 0
        psnr_val = 0
        for cid, client_model in zip(self.uploaded_ids, self.uploaded_models):
            client_model.eval()
            origin_grad = []
            for gp, pp in zip(self.global_model.parameters(), client_model.parameters()):
                origin_grad.append(gp.data - pp.data)

            target_inputs = []
            trainloader = self.clients[cid].load_train_data()
            with torch.no_grad():
                for i, (x, y) in enumerate(trainloader):
                    if i >= self.batch_num_per_client:
                        break

                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = client_model(x)
                    target_inputs.append((x, output))

            d = DLG(client_model, origin_grad, target_inputs)
            if d is not None:
                psnr_val += d
                cnt += 1
            
            # items.append((client_model, origin_grad, target_inputs))
                
        if cnt > 0:
            print('PSNR value is {:.2f} dB'.format(psnr_val / cnt))
        else:
            print('PSNR error')

        # self.save_item(items, f'DLG_{R}')

    def set_new_clients(self, clientObj):
        for i in range(self.num_clients, self.num_clients + self.num_new_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=False, 
                            send_slow=False)
            self.new_clients.append(client)

    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_model)
            opt = torch.optim.SGD(client.model.parameters(), lr=self.learning_rate)
            CEloss = torch.nn.CrossEntropyLoss()
            trainloader = client.load_train_data()
            client.model.train()
            for e in range(self.fine_tuning_epoch_new):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(client.device)
                    else:
                        x = x.to(client.device)
                    y = y.to(client.device)
                    output = client.model(x)
                    loss = CEloss(output, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

    # evaluating on new clients
    def test_metrics_new_clients(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.new_clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.new_clients]

        return ids, num_samples, tot_correct, tot_auc
    
    def data_log(self, data):
        if not self.no_wandb:
            wandb.log(data)

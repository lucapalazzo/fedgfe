import copy
import wandb
from flcore.clients.clientRewind import clientRewind
from flcore.servers.serverbase import Server
from threading import Thread
import time
import numpy as np
from collections import defaultdict
import random
import itertools
from utils.data_utils import read_client_data
import concurrent.futures
import torch.futures as futures
import pandas as pd
from sklearn.metrics import confusion_matrix

import time
from itertools import cycle

from flcore.routing.scoredrouting import ScoredRouting
from flcore.routing.randomrouting import RandomRouting
from flcore.routing.cyclicstaticrouting import StaticCyclicRouting

class FedRewind(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.rewind_ratio = args.rewind_ratio
        self.rewind_epochs = args.rewind_epochs
        self.rewind_interval = args.rewind_interval
        self.rewind_rotate = args.rewind_rotate
        self.global_rounds = args.global_rounds
        self.rewind_random = args.rewind_random
        self.rewind_noise = args.rewind_noise
        self.rewind_donkey = args.rewind_donkey
        self.rewind_donkey_count = args.rewind_donkey_count
        self.rewind_learning_rate_decay = args.rewind_learning_rate_decay
        self.rewind_learning_rate_decay_ratio = args.rewind_learning_rate_decay_ratio
        self.rewind_learning_rate_keep = args.rewind_learning_rate_keep
        
        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientRewind)
        if self.no_wandb == False:
            self.define_metrics()

        for client in self.clients:
            client.federation_clients = self.clients

        # routes = self.get_routes()
        # self.distribute_routes(routes)
        
        # print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.num_classes = args.num_classes
        self.statistics_dataframe = None
        #self.global_logits = [None for _ in range(args.num_classes)]

    def statistics_init(self):
        # init pandas dataframe for nodes and model statistics per round
        self.statistics_dataframe = pd.DataFrame(columns=['round', 'node', 'model', 'train_loss', 'train_acc', 'test_acc', 'test_auc', 'rewind_loss', 'rewind_acc', 'rewind_auc'])

    def statistics_update(self, round, node, model, train_loss, train_acc, test_acc, test_auc, rewind_loss, rewind_acc, rewind_auc):
        # update pandas dataframe with statistics
        self.statistics_dataframe = self.statistics_dataframe.append({'round': round, 'node': node, 'model': model, 'train_loss': train_loss, 'train_acc': train_acc, 'test_acc': test_acc, 'test_auc': test_auc, 'rewind_loss': rewind_loss, 'rewind_acc': rewind_acc, 'rewind_auc': rewind_auc}, ignore_index=True)
        
    def train_thread(self, client, device=-1, future = None, previous_node = None):

        if (device != -1):
            client.device = device
        thread = Thread(target=self.client_thread, args=(client, device, future, previous_node))
        thread.start()
        
        return thread

    def client_thread(self, client, device=-1, future = None, previous_node = None):

        if (device != -1):
            client.device = device
        target=client.train( rewind_train_node = previous_node )
        if future != None:
            future.set_result(-1)

    def train(self):
       
        for i in range(self.global_rounds+1):
            s_t = time.time()
            # importante commentare questa riga per avere i client sempre ordinati
            #self.selected_clients = self.select_clients()
            self.selected_clients = self.clients

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
                self.evaluate()
            running_threads = { 0: None, 1: None }
            running_futures = { 0: None, 1: None }
            running_clients = { 0: None, 1: None }
            running_start_times = { 0: 0, 1: 0 }
            # with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:a
            client_count = len(self.clients)
            client_index = 0
            # availables_gpus = [ 0 , 1]
            availables_gpus = self.gpus
            while client_index < client_count:
                client = self.clients[client_index]
                client.round = i
                client.thread = None
                client.federation_size = len(self.clients)
            # while client.thread == None:
                for gpu in availables_gpus:
                    if running_threads[gpu] == None:
                        # print("Starting training of node %d on GPU %d" % (client.id, gpu))

                        device = "cuda:"+str(gpu)
                        # executor.map(client.train, device)
                        running_futures[gpu] = futures.Future()
                        future = running_futures[gpu]
                        node_previous_length = len(client.rewind_previous_node_id)
                        previous_node = None
                        if ( node_previous_length > 0 ):
                            previous_node_index = client.rewind_previous_node_id[node_previous_length-1]
                            for previous_client in self.clients:
                                if previous_client.id == previous_node_index:
                                    previous_node = previous_client
                                    break
                        running_threads[gpu] = self.train_thread(client, device, future, previous_node)
                        running_start_times[gpu] = time.time()
                        running_clients[gpu] = client
                        # running_threads[gpu] = self.train_thread (client, device)
                        client.thread = running_threads[gpu]
                        client_index += 1
                        break
                for gpu in availables_gpus:
                    if running_futures[gpu] != None:
                        # print(running_futures[0].done())
                        if running_futures[gpu].done():
                            elapsed = time.time() - running_start_times[gpu]
                            client_type = "standard"
                            running_client = running_clients[gpu]
                            running_client_id = running_client.id
                            # client_model_name = str(running_client.model).split( "(", 1)[0]
                            running_threads[gpu] = None
                            running_futures[gpu] = None
                            
                            # print ( "Calling ending hook from main")
                            self.client_round_ending_hook( running_client )
                time.sleep(0.1)
            
            
            while running_futures[0] != None or running_futures[1] != None:
                for gpu in availables_gpus:
                    if running_futures[gpu] != None:
                        running_client = running_clients[gpu]
                        # print(running_futures[0].done())
                        if running_futures[gpu].done():
                            elapsed = time.time() - running_start_times[gpu]
                            client_type = "standard"
                            running_client_id = running_client.id
                            # if self.clients[running_client_id].is_strong:
                            #     client_type = "strong"
                            client_model_name = str(running_client.model).split( "(", 1)[0]
                            running_client.model
                            running_threads[gpu] = None
                            running_futures[gpu] = None
                            # print ( "Calling ending hook from loop")
                            self.client_round_ending_hook( running_client )
                time.sleep(0.1)
            
            self.routes = self.get_routes()
            # self.routes = get_routes(self.num_clients, self.clients)
            self.distribute_routes(self.routes)
            self.dump_routes(self.routes)


            print(self.uploaded_ids)
            # self.send_logits()

            self.Budget.append(time.time() - s_t)
            print('-'*50 + "Round time: ", self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

            self.save_checkpoint()


        # print("Node routes\n")
        # for node in self.clients:
        #     print ( "Node %d -> %s <- %s" % (node.id, node.node_routes, node.rewind_previous_node) )
        # for client in self.clients:

        #     client.save_model()
        # print("\nBest accuracy.")
        
        for test_client in self.clients:
            if not self.no_wandb:
                wandb.define_metric(f"node_acc_{test_client.id}", step_metric="node")
            for dest_client in self.clients:
                if ( test_client.id != dest_client.id):
                    acc, test_num, auc, y_true, y_prob = test_client.test_metrics_other(dest_client)
                    round_acc = acc/test_num
                    self.data_log({f"node_acc_{test_client.id}": round_acc, "node": dest_client.id})
                    print("Accuracy of nodes %d model on node %d: %02f" % (test_client.id, dest_client.id, round_acc ))
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        self.data_log({"best_acc": max(self.rs_test_acc)})
        # wandb.log({"best_acc": max(self.rs_test_acc)})
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        wandb.finish()

    def get_routes ( self, clients = None):
        if clients is None:
            clients = self.clients

        routing_clients = [client for client in self.clients]
        check_clients = [client for client in self.clients]
        random.shuffle(check_clients)
        routes = np.array([c.id for c in clients])
        for client in check_clients:
            client_next_route = client.route( available_clients = routing_clients)
            if client_next_route == -1:
                print("Error: client_next_route is -1, keeping model on current node")
                client_next_route = client.id
            routes[client.id] = client_next_route
            routing_clients.remove(self.clients[client_next_route])

        return routes


    
    def dump_routes ( self, routes ):
        print ( "Routes: ", end="")
        for node in routes:
            next_node = routes[node]
            orig_node_train_model_id = self.clients[next_node].starting_model.id
            orig_node_train_model = self.clients[next_node].starting_model.inner_model
            orig_node_train_optimizer = self.clients[next_node].starting_model.optimizer
            next_node_train_model_id = self.clients[next_node].next_train_model_id
            next_node_train_model = self.clients[next_node].next_train_model.inner_model
            next_node_train_optimizer = self.clients[next_node].next_train_model.optimizer
            orig_model_id = self.clients[next_node_train_model_id].id
            orig_model = self.clients[next_node_train_model_id].starting_model.inner_model
            print ( "%d->%d " % (node, next_node), end="")
            if ( self.rewind_epochs or self.rewind_ratio) and len(self.clients[next_node].rewind_previous_node_id) > 0:
                print ( "(>%d) " % (self.clients[next_node].rewind_previous_node_id[-1]), end="")
            # print ( "Orig Node training model id %d %s original id %d %s optim %s" % ( node, hex(id(orig_node_train_model) ), orig_node_train_model_id, hex(id(orig_model)), hex(id(orig_node_train_optimizer)) ) )
            # print ( "Next node training model id %d %s original id %d %s optim %s" % ( next_node, hex(id(next_node_train_model) ), next_node_train_model_id, hex(id(orig_model)), hex(id(next_node_train_optimizer)) ) )
        print()
    def distribute_routes (self, routes ):
        if routes is None:
            routes = self.routes

        for node in range(len(routes)):
            next_node = routes[node]
            previous_node = node
            next_client = None
            for next_client in self.clients:
                if next_node == next_client.id:
                    break
            if next_client == None:
                print("Error: next client not found")
                continue
            next_client.rewind_previous_node_id.append(previous_node)
            for c in self.clients:
                if previous_node == c.id:
                    next_client.rewind_previous_node.append(c)
                    break
            # next_client.rewind_previous_node.append(self.clients[previous_node])
            for client in self.clients:
                if node == client.id:
                    client.node_routes.append(next_node)
                    break
            if self.rewind_rotate:
                next_client.next_train_model = client.train_model
                next_client.next_train_model_id = client.train_model_id
                # print ( "Node %d model will be sent to %d and will rewind back to node %d" % (node, next_node, previous_node ) )
        
        for client in self.clients:
            client.model = client.next_train_model
            
    def client_round_ending_hook(self, client):

        round_loss, previous_loss = self.round_train_metrics( client )
        round_accuracy = self.round_test_metrics( client )

        print ( "Node %d orig loss %02f accuracy %02f last lost %02f" % ( client.id, round_loss, round_accuracy, previous_loss ) )

    def set_clients(self, clientObj):
        # n_strong = 0
        # n_weak = 0
        gpus = cycle(self.gpus)

        # for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            
            file_prefix = ""
            # if i != 2 and i != 3:
            # if i != 2:
                # continue 
            # train_data = read_client_data(self.dataset, i, is_train=True, prefix=file_prefix)
            # test_data = read_client_data(self.dataset, i, is_train=False, prefix=file_prefix)
            train_data = None
            test_data = None
            train_data_len = -1
            test_data_len = -1

            client = clientObj(self.args, 
                            id=i, 
                            train_samples=train_data_len, 
                            test_samples=test_data_len, 
                            train_slow=train_slow, 
                            send_slow=send_slow,
                            rewind_epochs=self.rewind_epochs,
                            rewind_interval=self.rewind_interval,
                            rewind_ratio=self.rewind_ratio,
                            train_data=train_data,
                            test_data=test_data,
                            dataset_limit=self.dataset_limit)
            client.prefix=file_prefix
            client.device = "cuda:"+str(next(gpus))
            client.available_clients = np.arange(self.num_clients)
            # client.routing = RandomRouting(self.num_clients, id = i)

            if self.args.routing_scored:
                client.routing = ScoredRouting(self.num_clients, id = i, average=self.routing_scored_average)
            if self.args.routing_cyclic:
                client.routing = StaticCyclicRouting(self.num_clients, id = i)
            else:
                client.routing = RandomRouting(self.num_clients, id = i)
            # client.node_data.stats_wandb_define()
            if self.no_wandb == False:
                client.node_data.stats_wandb_log()
            
            # if is_strong:
            #     n_strong += 1
            # else:
            #     n_weak += 1
            self.clients.append(client)

    def define_metrics(self):
        super().define_metrics()
  
        for client in self.clients:
            wandb.define_metric(f"test/model_{client.id}/acc", step_metric="round")
            wandb.define_metric(f"test/model_{client.id}/bal", step_metric="round")
            wandb.define_metric(f"train/model_{client.id}/round_train_loss_{client.id}", step_metric="round")
            wandb.define_metric(f"test/model_{client.id}/round_test_acc_{client.id}", step_metric="round")
            wandb.define_metric(f'train/model_loss_{client.train_model_id}', step_metric="round")
            wandb.define_metric(f"test/model_{client.id}/test_std", step_metric="round")
            wandb.define_metric(f"test/model_{client.id}/test_std_on_train", step_metric="round")
            if self.rewind_ratio > 0 or self.rewind_epochs > 0:
                wandb.define_metric(f"rewind/rewind_phase_loss_{client.id}", step_metric="rewind_step")
                wandb.define_metric(f"train/model_{client.id}/pre_rewind_loss_on_local", step_metric="round")
                wandb.define_metric(f"train/model_{client.id}/pre_rewind_loss_on_previous", step_metric="round")
                wandb.define_metric(f"train/model_{client.id}/post_rewind_loss_on_local", step_metric="round")
                wandb.define_metric(f"train/model_{client.id}/post_rewind_loss_on_previous", step_metric="round")
                wandb.define_metric(f"train/model_{client.id}/atend_loss_on_previous", step_metric="round")
                wandb.define_metric(f"train/model_{client.id}/atend_loss_on_local", step_metric="round")
                wandb.define_metric(f"rewind/rewind_loss_{client.id}", step_metric="round")

            for test_client in self.clients:
                wandb.define_metric(f"train/model_{client.id}/round_train_loss_{client.id}_on_{test_client.id}", step_metric="round")
                wandb.define_metric(f"test/model_{client.id}/round_test_acc_{client.id}_on_{test_client.id}", step_metric="round")

        # super().define_metrics()
        # if not self.no_wandb:
        #     for client in self.clients:
        #         wandb.define_metric(f"train/client_{client.id}/round_train_loss_{client.id}", step_metric="round")
        #         wandb.define_metric(f"test/client_{client.id}/round_test_acc_{client.id}", step_metric="round")
        #         if self.rewind_ratio > 0 or self.rewind_epochs > 0:
        #             wandb.define_metric(f"rewind/rewind_phase_loss_{client.id}", step_metric="round")

        #         for test_client in self.clients:
        #             if ( test_client.id != client.id):
        #                 wandb.define_metric(f"train/client_{client.id}/round_train_loss_{client.id}_on_{test_client.id}", step_metric="round")
        #                 wandb.define_metric(f"test/client_{client.id}/round_test_acc_{client.id}_on_{test_client.id}", step_metric="round")


    def evaluate(self):
        super().evaluate()

        return
        # clients_test_stats, clients_train_stats = self.federation_metrics()


    def federation_metrics(self):
        num_samples = []
        tot_correct = []
        tot_auc = []

        y_t =[]
        y_p = []
        clients_test_stats = []
        clients_train_stats = []

        for c in self.clients:
            client_test_stats = []
            for t in self.clients:
                client_test_stat = []
                ct, ns, auc, y_true, y_prob = c.test_metrics(t)
                # ct, ns, auc = c.test_metrics()
                tot_correct.append(ct*1.0)
                tot_auc.append(auc*ns)
                num_samples.append(ns)
                y_t.append(y_true)
                y_p.append(y_prob)
                client_test_stat.append(ct*1.0/ns)
                client_test_stat.append(auc*ns)
                client_test_stat.append(ns)
                client_test_stat.append(y_true)
                client_test_stat.append(y_prob)
                client_test_stats.append(client_test_stat)
            clients_test_stats.insert(c.model.id, client_test_stats)
            # clients_stats.append(client_stats)

        for c in self.clients:
            client_train_stats = []
            for t in self.clients:
                client_train_stat = []
                ct, ns, auc, y_true, y_prob = c.test_metrics(t, on_train = True )
                # ct, ns, auc = c.test_metrics()
                tot_correct.append(ct*1.0)
                tot_auc.append(auc*ns)
                num_samples.append(ns)
                y_t.append(y_true)
                y_p.append(y_prob)
                client_train_stat.append(ct*1.0/ns)
                client_train_stat.append(auc*ns)
                client_train_stat.append(ns)
                client_train_stat.append(y_true)
                client_train_stat.append(y_prob)
                client_train_stats.append(client_train_stat)
            # train_clients_stats.append(client_stats)
            clients_train_stats.insert(c.model.id, client_train_stats)
        # test_accs = []
        # train_accs = []
        # train_losses = []
        # acc_std = []
        # acc_std_on_train = []
        # for client in self.clients:
        #     test_acc, test_num, auc, test_y_true, test_y_prob = client.test_metrics()
        #     test_accs.append(test_acc)
        #     train_loss, train_num = client.train_metrics()
        #     # train_accs.append(train_acc)
        #     train_losses.append(train_loss)
        #     acc_std.append(client.test_std)
        #     acc_std_on_train.append(client.test_std_on_train)

        acc_std_mean, acc_std_on_train_mean  = self.federation_metrics_std( clients_test_stats, clients_train_stats )
        self.data_log({"federation/acc_std": acc_std_mean, "round":self.round})
        self.data_log({"federation/acc_std_on_train": acc_std_on_train_mean, "round":self.round})
        print(f"Mean standard deviation of accuracies on test sets: {acc_std_mean} {acc_std_on_train_mean}")

        return clients_test_stats, clients_train_stats
    
    def federation_metrics_std(self, clients_test_stats = None, clients_train_stats = None ):
        test_accs = []
        train_accs = []
        train_losses = []
        acc_std = []
        acc_std_on_train = []
        if ( clients_test_stats == None ):
            for client in self.clients:
                test_acc, test_num, auc, test_y_true, test_y_prob = client.test_metrics()
                test_accs.append(test_acc)
                train_loss, train_num = client.train_metrics()
                # train_accs.append(train_acc)
                train_losses.append(train_loss)
                acc_std.append(client.test_std)
                acc_std_on_train.append(client.test_std_on_train)
        else:
            for client_stats in clients_test_stats:
                for test_client_stats in client_stats:
                    test_acc = test_client_stats[0]
                    test_accs.append(test_acc)
                    test_num = test_client_stats[2]
                    auc = test_client_stats[1]/test_num
                    # train_loss = test_client_stats[3]
                    # train_losses.append(train_loss)
                    y_true = test_client_stats[3]
                    y_prob = test_client_stats[4]
                    acc_std.append(self.round_test_metric_deviation( test_accs ))
            for client_stats in clients_train_stats:
                for test_client_stats in client_stats:
                    test_acc = test_client_stats[0]
                    test_accs.append(test_acc)
                    test_num = test_client_stats[2]
                    auc = test_client_stats[1]/test_num
                    # train_loss = test_client_stats[3]
                    # train_losses.append(train_loss)
                    y_true = test_client_stats[3]
                    y_prob = test_client_stats[4]
                    acc_std_on_train.append(self.round_test_metric_deviation( test_accs ))

        acc_std_mean = np.mean(acc_std)
        acc_std_on_train_mean = np.mean(acc_std_on_train)
        return acc_std_mean, acc_std_on_train_mean


    def round_rewind_train_metrics(self, client):
        previous_node = None
        previous_loss = -1
        previous_losses = []
        previous_losses_log = ""
        for rewind_node in client.rewind_previous_node:
            previous_loss, previous_train = rewind_node.train_metrics_other(client)
            previous_loss = previous_loss/previous_train
            previous_losses.append(previous_loss)
            previous_losses_log += f"{rewind_node.id}:{previous_loss:.2f} "
        return previous_losses, previous_losses_log

    def round_train_metrics(self, client):
        losses, train = client.train_metrics()

        # print ( "Getting train metrics on model %s loss %s " % ( hex(id(client.train_model)), hex(id(client.loss) ) ))
        previous_loss = -1
        round_loss = losses/train
        self.data_log({f'train/model_{client.id}/round_train_loss_{client.id}': round_loss, "round": self.round})
        # self.data_log({f'train/model_loss_{client.train_model_id}': round_loss, "round": self.round})
        loss_dict = {client.train_model_id: round_loss}
        client.node_data_losses.append(loss_dict)
        # node_data_loss_string = [ f"{k}:{v:.2f}" for k,v in [ node_data_loss for node_data_loss in client.node_data_losses ] ]
        node_data_loss_string = ""
        for node_data_loss in client.node_data_losses:
            k,v = [n for n in node_data_loss.items()][0]
            node_data_loss_string += f"{k}:{v:.2f} "
        
        print("** Round %d Trained node %d using model from %d on dataset %d loss %02f (%s)" % ( client.round, client.id, client.train_model_id, client.node_data.id, round_loss, node_data_loss_string ))
        if len(client.rewind_previous_node):
            previous_losses, previous_losses_log = self.round_rewind_train_metrics(client)
            previous_loss = previous_losses[-1]
            client.rewind_previous_node_loss.append(previous_loss)
            print("** Previous rewind nodes' loss %s" % ( previous_losses_log ))
            self.data_log({f"rewind/rewind_loss_{client.model.id}": previous_loss, "round": self.round})
        # else:
        #     client.rewind_previous_node_loss.append(round_loss)

        return round_loss, previous_loss

    def round_test_metric_deviation (self, accuracies):
        # Calcola la deviazione standard tra le loss dei nodi e dei modelli
        standard_deviation = np.std(accuracies)

        return standard_deviation
    def round_test_metrics(self, client):
        # if len(client.rewind_previous_node) > 0:
        #     previous_node = client.rewind_previous_node[-1]
        #     previous_accuracy, previous_test = client.test_metrics_other(previous_node)
        #     previous_accuracy = previous_accuracy/previous_test

        acc, test_num, auc, y_true, y_prob = client.test_metrics()
        client_round_acc = acc/test_num
        # if not self.no_wandb:
        #     wandb.log({f'test/model_{client.id}/round_test_acc_{client.id}': client_round_acc, "round": self.round})
        test_acc, test_num, auc, test_y_true, test_y_prob  = client.test_metrics()

        accuracy = test_acc/test_num
        accuracies = self.round_test_metrics_models(client, ignore_last=False)
        accuracies_list = [ acc['accuracy'] for acc in accuracies]
        acc_std = np.std(accuracies_list)
        client.test_std.append(acc_std)


        test_acc, test_num, auc, test_y_true, test_y_prob  = client.test_metrics( on_train = True)
        accuracy_on_train = test_acc/test_num
        accuracies_on_train = self.round_test_metrics_models(client, on_train = True, ignore_last=False)
        accuracies_list = [ acc['accuracy'] for acc in accuracies_on_train]

        acc_std_on_train = np.std(accuracies_list)
        client.test_std_on_train.append(acc_std_on_train)
       
        print("** Round %d Trained node %d model %d accuracy %02f other %s" % (self.round, client.id, client.model.id, client_round_acc, accuracies ))
        print("** Round %d Accuracies on test sets %.02f %s" % ( self.round, accuracy, accuracies ))
        print("** Round %d Accuracies on train sets %.02f %s" % ( self.round, accuracy_on_train, accuracies_on_train ))
        print("** Round %d std on test %.02f on train %.02f" % ( self.round, acc_std, acc_std_on_train ))
        if not self.no_wandb:    
            wandb.log({f'test/model_{client.id}/test_std': acc_std, "round": self.round})
            wandb.log({f'test/model_{client.id}/test_std_on_train': acc_std_on_train, "round": self.round})
        # standard_deviation = self.round_test_metric_deviation(client)
        # print(f"Standard deviation of accuracies for client {client.id}: {standard_deviation}")
        return client_round_acc
    
    def round_test_metrics_models (self, client, ignore_last = True, on_train = False):
        accuracies = []
        for test_client in self.clients:
            if ( test_client.node_data.id != client.node_data.id or ignore_last == False ):
                acc, test_num, auc, y_true, y_prob = client.test_metrics(test_client, on_train = on_train)
                if test_num == 0:
                    continue
                round_acc = acc/test_num
                other_accuracy = { 'node_dataset': test_client.node_data.id, 'accuracy': round_acc }
                accuracies.append(other_accuracy)
                # print("Node's model %d accuracy dataset %d: %02f" % (client.id, test_client.id, round_acc )) 
                if  not self.no_wandb:
                    wandb.log({f'test/model_{client.model.id}/round_test_acc_{client.model.id}_on_{test_client.node_data.id}': round_acc, 'round': self.round } )
                # if previous_node != None:
                #     client.rewind_previous_node_loss.append(previous_loss)
                #     print("Previous node %d loss %02f" % ( previous_node.id, previous_loss))
        return accuracies

            
# ---------------- From FedER -----------------------------
def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

def get_routes(n_nodes, clients = None):
    if clients is not None:
        idxs = [client.id for client in clients]
    else:
        idxs = [i for i in range(n_nodes)]
    random.shuffle(idxs)
    routes = {x[0]:x[1] for x in pairwise(idxs)}
    last_route = (idxs[-1],idxs[0])
    routes[last_route[0]] = last_route[1]
    return routes # k: sender, v: receiver

#---------------------------------------------





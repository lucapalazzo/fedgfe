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

import time


class FedRewind(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.rewind_ratio = args.rewind_ratio
        self.rewind_epochs = args.rewind_epochs
        self.rewind_interval = args.rewind_interval
        self.rewind_rotate = args.rewind_rotate
        self.global_rounds = args.global_rounds
        
        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientRewind)
        self.define_metrics()
       
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.num_classes = args.num_classes
        #self.global_logits = [None for _ in range(args.num_classes)]

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
            self.round = i
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
                client.thread = None
            # while client.thread == None:
                for gpu in availables_gpus:
                    if running_threads[gpu] == None:
                        # print("Starting training of node %d on GPU %d" % (client.id, gpu))

                        device = "cuda:"+str(gpu)
                        # executor.map(client.train, device)
                        running_futures[gpu] = futures.Future()
                        future = running_futures[gpu]
                        node_previous_length = len(client.rewind_previous_node)
                        previous_node = None
                        if ( node_previous_length > 0 ):
                            previous_node_index = client.rewind_previous_node[node_previous_length-1]
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
                            # if self.clients[running_client_id].is_strong:
                            #     client_type = "strong"
                            client_model_name = str(running_client.model).split( "(", 1)[0]
                            # running_client_id.model
                            running_threads[gpu] = None
                            running_futures[gpu] = None
                            self.round_train_metrics( running_client )
                            self.round_test_metrics( running_client )
                time.sleep(0.1)
                # client.train()
            
            
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
                            self.round_train_metrics( running_client )
                            self.round_test_metrics( running_client )
                            # acc, test_num, auc, y_true, y_prob = client.test_metrics()
                            # round_acc = acc/test_num
                            # round_loss = losses/train
                            # if not self.no_wandb:
                            #     wandb.log({f'train_round_loss_{running_client.id}': round_loss, "round": i})
                            #     wandb.log({f'test_round_acc_{running_client.id}': round_acc, "round": i})
                            # print("Trained %s node %d model %s on GPU %d in %d seconds loss %02f accuracy %02f test_num %d" % (client_type, running_clients[gpu], client_model_name, gpu, elapsed, round_loss, round_acc, test_num ))
                            # for test_client in self.clients:
                            #     if ( test_client.id != running_client.id):
                            #         acc, test_num, auc, y_true, y_prob = running_client.test_metrics_other(test_client)
                            #         round_acc = acc/test_num
                            #         print("Accuracy on node %d test set accuracy %02f" % (test_client.id, round_acc )) 
                time.sleep(0.1)
            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            #self.receive_logits()
            #self.global_logits = logit_aggregation(self.uploaded_logits)
            self.routes = get_routes(self.num_clients, self.clients)
            for node in self.routes:
                next_node = self.routes[node]
                previous_node = node
                for next_client in self.clients:
                    if next_node == next_client.id:
                        break

                next_client.rewind_previous_node.append(previous_node)
                for client in self.clients:
                    if node == client.id:
                        client.node_routes.append(next_node)
                if self.rewind_rotate:
                    next_client.train_model = client.train_model
                    next_client.train_model_id = next_client.id
                    print ( "Node %d model will be sent to %d and will rewind back to node %d" % (node, next_node, previous_node ) )

            print(self.uploaded_ids)
            # self.send_logits()

            self.Budget.append(time.time() - s_t)
            print('-'*50, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("Node routes\n")
        for node in self.clients:
            print ( "Node %d -> %s <- %s" % (node.id, node.node_routes, node.rewind_previous_node) )
        # for client in self.clients:

        #     client.save_model()
        print("\nBest accuracy.")
        # wandb.define_metric("node")
        for test_client in self.clients:
            wandb.define_metric(f"node_acc_{test_client.id}", step_metric="node")
            for dest_client in self.clients:
                if ( test_client.id != dest_client.id):
                    acc, test_num, auc, y_true, y_prob = test_client.test_metrics_other(dest_client)
                    round_acc = acc/test_num
                    wandb.log({f"node_acc_{test_client.id}": round_acc, "node": dest_client.id})
                    print("Accuracy of nodes %d model on node %d: %02f" % (test_client.id, dest_client.id, round_acc ))
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        self.data_log({"best_acc": max(self.rs_test_acc)})
        # wandb.log({"best_acc": max(self.rs_test_acc)})
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        wandb.finish()
        

    def send_logits(self):
        assert (len(self.clients) > 0)

        for snd_client in self.clients:
            start_time = time.time()
            rcv_client = self.clients[self.routes[snd_client.id]]
            assert snd_client.id != rcv_client.id
            assert rcv_client.id == self.routes[snd_client.id]
            # if rcv_client.is_strong:
            #     print ("Node is strong not KD-ing")
            #     rcv_client.set_logits(None)
            # else:
            #     rcv_client.set_logits(copy.deepcopy(snd_client.logits))

            print("send logits from {} to {}".format(snd_client.id, rcv_client.id))

            snd_client.send_time_cost['num_rounds'] += 1
            snd_client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    # def receive_logits(self):
    #     assert (len(self.selected_clients) > 0)

    #     self.uploaded_ids = []
    #     self.uploaded_logits = []
    #     for client in self.selected_clients:
    #         self.uploaded_ids.append(client.id)
    #         self.uploaded_logits.append(client.logits)

    def set_clients(self, clientObj):
        # n_strong = 0
        # n_weak = 0

        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            
            file_prefix = ""
            # if i != 2 and i != 3:
            # if i != 2:
            #     continue 
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
            
            # if is_strong:
            #     n_strong += 1
            # else:
            #     n_weak += 1
            self.clients.append(client)

    def define_metrics(self):
        if not self.no_wandb:
            for client in self.clients:
                wandb.define_metric(f"train/client_{client.id}/round_train_loss_{client.id}", step_metric="round")
                wandb.define_metric(f"test/client_{client.id}/round_test_acc_{client.id}", step_metric="round")
                if self.rewind_ratio > 0 or self.rewind_epochs > 0:
                    wandb.define_metric(f"rewind/rewind_phase_loss_{client.id}", step_metric="round")

                for test_client in self.clients:
                    if ( test_client.id != client.id):
                        wandb.define_metric(f"train/client_{client.id}/round_train_loss_{client.id}_on_{test_client.id}", step_metric="round")
                        wandb.define_metric(f"test/client_{client.id}/round_test_acc_{client.id}_on_{test_client.id}", step_metric="round")


    def federation_metrics(self):
        test_accs = []
        train_accs = []
        train_losses = []
        for client in self.clients:
            test_acc, test_num, _ = client.test_metrics()
            test_accs.append(test_acc)
            train_acc, train_loss, train_num = client.train_metrics()
            train_accs.append(train_acc)
            train_losses.append(train_loss)
        return test_accs, train_accs, train_losses

    def round_train_metrics(self, client):
        losses, train = client.train_metrics()
        
        # round_acc = acc/test_num
        round_loss = losses/train
        if not self.no_wandb:
            wandb.log({f'train/client_{client.id}/round_train_loss_{client.id}': round_loss, "round": self.round})
        print("Trained node %d loss %02f" % ( client.id, round_loss ))

    def round_test_metrics(self, client):
        acc, test_num, auc, y_true, y_prob = client.test_metrics()
        if not self.no_wandb:
            wandb.log({f'test/client_{client.id}/round_test_acc_{client.id}': round_acc, "round": self.round})
        print("Trained node %d accuracy %02f test_num %d" % (client.id, round_acc ))

        for test_client in self.clients:
            if ( test_client.id != client.id):
                acc, test_num, auc, y_true, y_prob = client.test_metrics_other(test_client)
                round_acc = acc/test_num
                print("Node's model %d accuracy dataset %d: %02f" % (client.id, test_client.id, round_acc )) 
                if  not self.no_wandb:
                    wandb.log({f'test/client_{client.id}/round_test_acc_{client.id}_on_{test_client.id}': round_acc, 'round': self.round } )
            
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





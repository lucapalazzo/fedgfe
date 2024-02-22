import copy
import wandb
from flcore.clients.clientZIO import clientZIO
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


class FedZio(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientZIO)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.num_classes = args.num_classes
        #self.global_logits = [None for _ in range(args.num_classes)]

    def train_thread(self, client, device=-1, future = None):

        if (device != -1):
            client.device = device
        thread = Thread(target=self.client_thread, args=(client, device, future))
        thread.start()
        
        return thread

    def client_thread(self, client, device=-1, future = None):

        if (device != -1):
            client.device = device
        target=client.train()
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
            availables_gpus = [ 0 , 1]
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
                        running_threads[gpu] = self.train_thread(client, device, future)
                        running_start_times[gpu] = time.time()
                        running_clients[gpu] = client_index
                        # running_threads[gpu] = self.train_thread (client, device)
                        client.thread = running_threads[gpu]
                        client_index+=1
                        break
                for gpu in availables_gpus:
                    if running_futures[gpu] != None:
                        # print(running_futures[0].done())
                        if running_futures[gpu].done():
                            elapsed = time.time() - running_start_times[gpu]
                            client_type = "weak"
                            running_client_id = running_clients[gpu]
                            if self.clients[running_client_id].is_strong:
                                client_type = "strong"
                            client_model_name = str(self.clients[running_client_id].model).split( "(", 1)[0]
                            self.clients[running_client_id].model
                            print("Trained %s node %d model %s on GPU %d in %d seconds" % (client_type, running_clients[gpu], client_model_name, gpu, elapsed ))
                            running_threads[gpu] = None
                            running_futures[gpu] = None
                time.sleep(0.1)
                # client.train()
            
            while running_futures[0] != None or running_futures[1] != None:
                for gpu in availables_gpus:
                    if running_futures[gpu] != None:
                        # print(running_futures[0].done())
                        if running_futures[gpu].done():
                            elapsed = time.time() - running_start_times[gpu]
                            client_type = "weak"
                            running_client_id = running_clients[gpu]
                            if self.clients[running_client_id].is_strong:
                                client_type = "strong"
                            client_model_name = str(self.clients[running_client_id].model).split( "(", 1)[0]
                            self.clients[running_client_id].model
                            print("Trained %s node %d model %s on GPU %d in %d seconds" % (client_type, running_clients[gpu], client_model_name, gpu, elapsed ))
                            running_threads[gpu] = None
                            running_futures[gpu] = None     
                time.sleep(0.1)
            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            #self.receive_logits()
            #self.global_logits = logit_aggregation(self.uploaded_logits)
            self.routes = get_routes(len(self.clients))
            print(self.uploaded_ids)
            self.send_logits()

            self.Budget.append(time.time() - s_t)
            print('-'*50, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        wandb.log({"best_acc": max(self.rs_test_acc)})
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        

    def send_logits(self):
        assert (len(self.clients) > 0)

        for snd_client in self.clients:
            start_time = time.time()
            rcv_client = self.clients[self.routes[snd_client.id]]
            assert snd_client.id != rcv_client.id
            assert rcv_client.id == self.routes[snd_client.id]
            if rcv_client.is_strong:
                print ("Node is strong not KD-ing")
                rcv_client.set_logits(None)
            else:
                rcv_client.set_logits(copy.deepcopy(snd_client.logits))

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
        n_strong = 0
        n_weak = 0

        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            
            file_prefix = ""
            is_strong = False
            if n_strong < self.args.num_clients_strong:
                is_strong = True
                file_prefix = "strong-"
                idx=n_strong
            else:
                file_prefix = "weak-"
                idx=n_weak
            
            train_data = read_client_data(self.dataset, idx, is_train=True, prefix=file_prefix)
            test_data = read_client_data(self.dataset, idx, is_train=False, prefix=file_prefix)

            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=train_slow, 
                            send_slow=send_slow,
                            is_strong=is_strong,
                            id_by_type=idx)
            client.prefix=file_prefix
            
            if is_strong:
                n_strong += 1
            else:
                n_weak += 1
            self.clients.append(client)

# ---------------- From FedER -----------------------------
def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

def get_routes(n_nodes):
    idxs = [i for i in range(n_nodes)]
    random.shuffle(idxs)
    routes = {x[0]:x[1] for x in pairwise(idxs)}
    last_route = (idxs[-1],idxs[0])
    routes[last_route[0]] = last_route[1]
    return routes # k: sender, v: receiver

#---------------------------------------------




# get best routing for client's model base on confugsion matrix and available clients

from flcore.routing.routingbase import FLRoutingBase
import numpy as np
import sklearn
import torch
from torch.nn import functional as F
import itertools
import random

class RandomRouting(FLRoutingBase):

    def __init__(self, clients_count = -1, federation_clients = None, id = -1, model = None):
        super(RandomRouting, self).__init__(clients_count, federation_clients, id = id, model = model)
        # self.route_pairs = None

    def __call__(self, *args, **kwargs):
        clients = kwargs.get('available_clients', None) 
        self.pairwise(clients)
        return self.route(*args, **kwargs)

    def route_pairs(self, available_clients = None):
        if available_clients is None:
            available_clients = self.federation_clients
        
        available_clients_ids = [client.id for client in available_clients]
        random.shuffle(available_clients_ids)
        route_pairs = {x[0]:x[1] for x in self.pairwise(available_clients_ids)}
        return route_pairs
    
    def route(self, available_clients = None, id = -1):
        """
        Route the request to the available clients.
        """
        super(RandomRouting, self).route(available_clients)
        # Get the best client based on the confusion matrix
        
        if available_clients is None:
            available_clients = self.federation_clients
        
        next_client_id = -1

        available_clients = self.get_available_clients(available_clients)
        # self.route_pairs = list(RandomRouting.pairwise(available_clients))

        # if self.id in self.route_pairs:
        #     next_client_id = self.route_pairs[self.id]
        next_client = np.random.choice(available_clients)
        next_client_id = next_client.id
        # print (f"Chosen client id: {next_client_id} ")
        return next_client_id

    def get_available_clients(self, available_clients, reduce_clients=False ):
        """
        Get the available clients.
        """
        if ( reduce_clients):
        # reduce the number of clients by choosing random clients from the available clients
            clients_count = np.random.randint(len(self.federation_clients))
            available_clients = np.sort(np.random.choice(self.federation_clients, clients_count, replace=False))

        if len(available_clients) > 1:
            available_clients = [client for client in available_clients if client.id != self.id]
        
        return available_clients
    
    def get(self, path):
        """
        Get the routing for the given path.
        """
        return self.routing.get(path)

    def add(self, path, routing):
        """
        Add the routing for the given path.
        """
        self.routing[path] = routing

    def remove(self, path):
        """
        Remove the routing for the given path.
        """
        if path in self.routing:
            del self.routing[path]

    def __iter__(self):
        return iter(self.routing)

    def __len__(self):
        return len(self.routing)

    def __str__(self):
        return str(self.routing)
    
    def pairwise(self, iterable):
        # pairwise('ABCDEFG') --> AB BC CD DE EF FG
        a, b = itertools.tee(iterable)
        next(b, None)
        return zip(a,b)
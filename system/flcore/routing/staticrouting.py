# get best routing for client's model base on confugsion matrix and available clients

from flcore.routing.routingbase import FLRoutingBase
import numpy as np
import sklearn
import torch
from torch.nn import functional as F
import itertools
import random

class StaticRouting(FLRoutingBase):

    def __init__(self, clients_count = -1, federation_clients = None, id = -1, model = None, random = False ):
        super(StaticRouting, self).__init__(clients_count, federation_clients, id = id, model = model)
        self.route_pairs = None
        self.running_round = -1
        self.routing_random = random

    def route(self, available_clients = None, round = -1, id = -1 ):
        """
        Route the request to the available clients.
        """
        super(StaticRouting, self).route(available_clients)
        # Get the best client based on the confusion matrix
        
        if available_clients is None:
            available_clients = self.federation_clients
        
        next_client_id = -1

        client_id = self.id if id == -1 else id

        if self.route_pairs is None or len(self.route_pairs) == 0:
            self.create_routes(available_clients)
            
        if client_id in self.route_pairs:
            next_client_id = self.route_pairs[client_id]

        return next_client_id
    def create_routes ( self, clients = None, shuffle = False):
        """
        Create the routes for the clients.
        """
        if self.route_pairs is None:
            pairs = list(StaticRouting.pairwise(self.federation_clients_ids, shuffle = self.routing_random))
            self.route_pairs = {x[0]:x[1] for x in pairs}
        
        return self.route_pairs

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
    
    def pairwise(iterable, shuffle = False):
        # pairwise('ABCDEFG') --> AB BC CD DE EF FG
        a, b = itertools.tee(iterable)

        if ( shuffle ):
            b = list(b)
            random.shuffle(b)
        else:
            b = list(b)
            b = b[1:] + b[:1]
        # next(b, 0)
        return zip(a, b)
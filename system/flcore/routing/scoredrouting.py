# get best routing for client's model base on confugsion matrix and available clients

from flcore.routing.routingbase import FLRoutingBase
import numpy as np
import sklearn
import torch
from torch.nn import functional as F

class ScoredRouting(FLRoutingBase):

    def __init__(self, clients_count = -1, federation_clients = None, id = -1, model = None, average = 'macro'):

        super(ScoredRouting, self).__init__(clients_count, federation_clients, id = id, model=model)
        self.average = average # average method for the f1_score matrix
    def route(self, available_clients = None):
        """
        Route the request to the available clients.
        """
        super(ScoredRouting, self).route(available_clients)
        # Get the best client based on the confusion matrix
        if available_clients is None:
            available_clients = self.federation_clients
        
        available_clients = self.get_available_clients(available_clients)
        # selected_clients = available_clients.copy()
        # np.random.shuffle(selected_clients)
        best_client_id, scores = self.get_best_client(available_clients)
        best_client = available_clients[best_client_id]
        print (f"Node {self.id} route best client id: {best_client.id} {scores}")
        return best_client.id

    def get_best_client(self, available_clients):
        """
        Get the best client based on the confusion matrix.
        """
        best_client = None
        best_score = 0
        scores = []
        for client in available_clients:
            test_acc, test_num, auc, y_true, y_pred = client.get_scores_data(model=self.model, on_train=True)
            y_pred = F.softmax(torch.tensor(y_pred), dim=1).numpy()
            y_pred = np.argmax(y_pred, axis=1)
            f1_score = sklearn.metrics.f1_score(y_true, y_pred, average=self.average)
            scores.append(f1_score)
        best_client_id = np.argmin(scores)
        return best_client_id, scores
    
    def get_available_clients(self, available_clients, reduce_clients=False ):
        """
        Get the available clients.
        """
        if ( reduce_clients):
        # reduce the number of clients by choosing random clients from the available clients
            clients_count = np.random.randint(len(self.federation_clients))
            available_clients = np.sort(np.random.choice(self.federation_clients, clients_count, replace=False))   
        
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
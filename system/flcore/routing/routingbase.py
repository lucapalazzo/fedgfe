

class FLRoutingBase(object):
    """
    Base class for routing.
    """


    def __init__(self, clients_count = -1, federation_clients = None, id = -1, model = None):
        self.model = model
        self.id = id
        self.previous = []
        self.federation_clients = []
        self.federation_clients = federation_clients
        self.federation_clients_count = clients_count
        
        if ( federation_clients is not None ):
            self.federation_clients_ids = federation_clients
        else:
            self.federation_clients_ids = [ client for client in range(clients_count)]

    def route(self, available_clients):
        """
        Route the request to the available clients.
        """

    def get(self, path):
        """
        Get the routing for the given path.
        """
        raise NotImplementedError()

    def add(self, path, routing):
        """
        Add the routing for the given path.
        """
        raise NotImplementedError()

    def remove(self, path):
        """
        Remove the routing for the given path.
        """
        raise NotImplementedError()

    def __getitem__(self, path):
        return self.get(path)

    def __setitem__(self, path, routing):
        self.add(path, routing)

    def __delitem__(self, path):
        self.remove(path)

    def __contains__(self, path):
        return self.get(path) is not None

    def __iter__(self):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def __str__(self):
        raise NotImplementedError()

    def __repr__(self):
        return str(self)
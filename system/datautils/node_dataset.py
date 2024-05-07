from utils.data_utils import read_client_data
from torch.utils.data import DataLoader
class NodeData:
    def __init__(self, args, id = -1, **kwargs):
        self.id = id
        self.args = args
        self.kwargs = kwargs   
        self.train_data = None
        self.train_samples = 0
        self.test_data = None
        self.test_samples = 0
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.labels = None
        self.labels_count = None
        self.labels_percent = None

    def load_train_data(self, batch_size, dataset_limit=0):
        if self.train_data == None:
            print("Loading train data for client %d" % self.id)
            self.train_data = read_client_data(self.dataset, self.id, is_train=True,dataset_limit=dataset_limit)
            self.train_samples = len(self.train_data)
        return DataLoader(self.train_data, batch_size, drop_last=True, shuffle=True)
   
    def load_test_data(self, batch_size, dataset_limit=0):
        if self.test_data == None:
            print("Loading test data for client %d" % self.id)
            self.test_data = read_client_data(self.dataset, self.id, is_train=False,dataset_limit=dataset_limit)
            self.test_samples = len(self.test_data)
        return DataLoader(self.test_data, batch_size, drop_last=False, shuffle=True)
    
    def stats_get(self):
        # labels = self.labels_get()
        labels = list(range(self.num_classes))
        if self.labels_count == None or self.labels_percent == None:
            self.labels_count = dict(zip(labels, [0]*len(labels)))
            for _,l in self.train_data:
                self.labels_count[l.item()] += 1
            self.labels_percent = {k: v*100/self.train_samples for k,v in self.labels_count.items()}

        return self.labels_count, self.labels_percent

    def stats_dump(self):
        labels_count, labels_percent = self.stats_get()
        print("Dataset %d stats: %s" % (self.id,labels_count))
        # print("Labels percent: %s" % labels_percent)

    def labels_get(self):
        if self.labels == None:
            self.labels = set([l.item() for t,l in self.train_data])
        return self.labels
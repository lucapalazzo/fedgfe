from torch.utils.data import DataLoader
import pandas as pd

class ChestDataload(DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers, dataset_path = "dataset", datafile = "Data Chest X-Ray RSUA (Validated)/Split_Data_RSUA_Paths_k3.xlsx"):
        super(ChestDataload, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers


    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __iter__(self):
        return self

    def __next__(self):
        return self

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __del__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def __enter__(self):
        pass

    def __delitem__(self, key):
        pass

    def __setitem__(self, key, value):
        pass

    def __getitem__(self, key):
        pass

    def __contains__(self, item):
        pass

    def __iter__(self):
        pass

    def __reversed__(self):
        pass
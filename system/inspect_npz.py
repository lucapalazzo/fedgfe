import numpy as np
import torch
import time
filename = "dataset/CIFAR-10-nIID-10C/train/0.npz"
data = np.load(filename, allow_pickle=True)['data'].tolist()
start_time = time.time()

X_train = torch.Tensor(data['x']).type(torch.float32)
y_train = torch.Tensor(data['y']).type(torch.int64)

train_data = [(x, y) for x, y in zip(X_train, y_train)]
lbls = set([l.item() for t,l in train_data])
stats = [{l:0} for l in lbls]

for y in y_train:
    stats[y.item()] += 1
    # print ( stats[y.item()] )
    print(y)

print(lbls)
# for item in lst:
#     print(item)
#     tensor = data[item]
#     # print(tensor)

print("--- %s seconds ---" % (time.time() - start_time))
import numpy as np
import time
filename = "dataset/ISIC_mel/train/0.npz"
data = np.load(filename, allow_pickle=True)
lst = data.files
start_time = time.time()
for item in lst:
    print(item)
    tensor = data[item]
    # print(tensor)

print("--- %s seconds ---" % (time.time() - start_time))
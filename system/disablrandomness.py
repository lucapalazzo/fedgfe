import random


def set_seed(seed):
    # Python
	random.seed(seed)
	
    # PyTorch
	try:
		import torch
		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)  # se si utilizza il multi-GPU
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
	except ImportError:
		pass

    # NumPy
	try:
		import numpy as np
		np.random.seed(seed)
	except ImportError:
		pass

	# scikit-learn USELESS IT USES NUMPY
	# try:
	# 	import sklearn
	# 	sklearn.set_config(random_state=seed)
	# except ImportError:
	# 	pass

	# TensorFlow
	try:
		import tensorflow as tf
		tf.random.set_seed(seed)
	except ImportError:
		pass

	# CuPy
	try:
		import cupy as cp
		cp.random.seed(seed)
	except ImportError:
		pass

	# SciPy USELESS IT USES NUMPY
	# try:
	# 	import scipy
	# 	scipy.random.seed(seed)
	# 	scipy.ran
	# except ImportError:
	# 	pass

	# pandas
	try:
		import pandas as pd
		pd.set_option('mode.chained_assignment', None)
	except ImportError:
		pass

	# Matplotlib matplotlib USELESS IF IT USES NUMPY
	# try:
	# 	import matplotlib.pyplot as plt
	# 	plt.seed(seed)
	# except ImportError:
	# 	pass

	# Seaborn
	# try:
	# 	import seaborn as sns
	# 	sns.set(seed)
	# except ImportError:
	# 	pass

	# OpenCV
	try:
		import cv2
		cv2.setRNGSeed(seed)
	except ImportError:
		pass
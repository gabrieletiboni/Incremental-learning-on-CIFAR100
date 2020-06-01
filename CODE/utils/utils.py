import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys


def get_indexes_from_labels(dataset, labels):

	targets = dataset.targets
	indexes = []

	for i,target in enumerate(targets):
		if target in labels:
			indexes.append(i)

	return indexes


def train_validation_split(dataset, indexes, train_size=0.9, random_state=None):

	targets = dataset.targets

	train_indexes, val_indexes, _, _ = train_test_split(list(range(0, len(indexes))), list(range(0, len(indexes))), train_size=train_size, stratify=[targets[i] for i in indexes], random_state=random_state)

	np_indexes = np.array(indexes)

	return list(np_indexes[train_indexes]), list(np_indexes[val_indexes])

# SHOW SOME RANDOM IMAGES
def show_random_images(dataset, n=5, mean=None, std=None):
	for i in range(n):
	    j = np.random.randint(0, len(dataset))
	    print('Label:',dataset[j][1])
	    imgshow(dataset[j][0], mean=mean, std=std)

	return

# --- vecchia versione di imgshow
# def imgshow(inp, title=None):
#     """Imshow for Tensor."""
#     inp = inp.numpy().transpose((1, 2, 0))
#     mean = np.array([0.5071, 0.4865, 0.4409])
#     std = np.array([0.2673, 0.2564, 0.2762])
#     # mean = np.array([0.5071, 0.4865, 0.4409])
#     # std = np.array([0.5071, 0.4865, 0.4409])
#     inp = std * inp + mean
#     inp = np.clip(inp, 0, 1)
#     plt.imshow(inp)
#     if title is not None:
#         plt.title(title)
#     plt.pause(0.001)  # pause a bit so that plots are updated

def imgshow(img, mean=None, std=None):
	if mean == None or std == None:
		print('FATAL ERROR - Dare mean e std alla funzione imgshow')
		sys.exit()

	mean = np.array(mean)
	std = np.array(std)
	for i in range(3):
		img[i] = img[i]*std[i] + mean[i] # unnormalize

	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.show()

	return
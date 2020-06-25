"""

1) NB come classifier

2) Logistic Regression come classifier


2) PCA come dimensionality reduction


3) Scheduling of K with different functions + LR variabile



Cose aggiuntive da provare:
- freezare gli output successivi
- freezare gli output layer già imparati




"""



import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.naive_bayes import GaussianNB

import torch

def eval_model_NB(net,
				  training_dataset,
				  test_dataloader,
				  dataset_length=None,
				  display=True, suffix=' (group)', icarl=None,
				  normalize=True):

	if icarl == None:
		raise RuntimeError('Errore icarl non presente')

	training_length = len(training_dataset)

	print('Training set lenght in NB:', training_length)

	print('test un sample di prova:', training_dataset[i][0], training_dataset[i][1])


	X = torch.zeros( (training_length, 64), dtype=torch.float32)
	y = torch.zeros( (training_length))

	for i in range(training_length):
		X[i,:] = net.feature_map(training_dataset[i][0])
		y[i] = training_dataset[i][1]

	X = X.to('cuda')
	y = y.to('cuda')

	gnb = GaussianNB()
	gnb.fit(X, y)

	


	return
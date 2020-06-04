import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import datetime
import sys
import pandas as pd

# from google.colab import auth
import gspread
# from oauth2client.client import GoogleCredentials
from oauth2client.service_account import ServiceAccountCredentials
from google.colab import output

import torch
import os


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

def compute_mean_and_std():
	"""**Compute means and stds to normalize**"""
	means_1 = []
	means_2 = []
	means_3 = []
	stds_1 = []
	stds_2 = []
	stds_3 = []

	for img, lab in train_dataset:      # For test dataset Normalization iter on test_dataset
	    means_1.append(torch.mean(img[0]))
	    means_2.append(torch.mean(img[1]))
	    means_3.append(torch.mean(img[2]))
	    stds_1.append(img[0])
	    stds_2.append(img[1])
	    stds_3.append(img[2])

	stds_1 = torch.cat((stds_1), 0)
	stds_2 = torch.cat((stds_2), 0)
	stds_3 = torch.cat((stds_3), 0)
	mean_1 = torch.mean(torch.tensor(means_1))
	mean_2 = torch.mean(torch.tensor(means_2))
	mean_3 = torch.mean(torch.tensor(means_3))
	std_1 = torch.std(stds_1)
	std_2 = torch.std(stds_2)
	std_3 = torch.std(stds_3)

	print("Means = [{:.4f}, {:.4f}, {:.4f}]".format(mean_1.item(), mean_2.item(), mean_3.item()))
	print("Stds = [{:.4f}, {:.4f}, {:.4f}]".format(std_1.item(), std_2.item(), std_3.item()))

def draw_graphs(losses_train, losses_eval, accuracies_train, accuracies_eval, num_epochs, use_validation=True, print_img=False, save=False, path=None, group_number=None):
	if use_validation:
		text1 = 'Validation loss'
		text2 = 'Validation accuracy'
	else:
		text1 = 'Test loss'
		text2 = 'Test accuracy'

	fig1, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,5))

	epochs = [i for i in range(1, num_epochs+1)]
	ax.plot(epochs, losses_train, linestyle='-', marker='o', label='Training loss')
	ax.plot(epochs, losses_eval, linestyle='-', marker='o', label=text1)

	ax.set_xlabel('Epochs', labelpad=12, fontweight='bold')
	ax.set_ylabel('Loss', labelpad=12, rotation=90, fontweight='bold')

	ax.set_title('Loss during gradient descent', pad=20, fontweight='bold')

	ax.legend()
	plt.grid(alpha=0.3)
	if print_img:
		plt.show()
	plt.close();

	# Plot accuracies
	fig2, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,5))

	epochs = [i for i in range(1, num_epochs+1)]

	if len(accuracies_train) < num_epochs:
		ax.plot(epochs, accuracies_eval, color='#FFC107', linestyle='-', marker='o', label=text2)
	else:
		ax.plot(epochs, accuracies_train, color='#7B1FA2', linestyle='-', marker='o', label='Training accuracy')
		ax.plot(epochs, accuracies_eval, color='#FFC107', linestyle='-', marker='o', label=text2)

	ax.set_xlabel('Epochs', labelpad=12, fontweight='bold')
	ax.set_ylabel('Accuracy', labelpad=12, rotation=90, fontweight='bold')

	ax.set_title('Accuracy during gradient descent', pad=20, fontweight='bold')

	ax.legend()
	plt.grid(alpha=0.3)
	if print_img:
		plt.show()
	plt.close();

	# Save figures
	if save:
		if path == None:
			print('FATAL ERROR - Dare un path come parametro al draw_graphs')
			sys.exit()

		fig1.savefig(path+'/group_'+str(group_number)+'/loss.png')
		fig2.savefig(path+'/group_'+str(group_number)+'/accuracy.png')

def draw_final_graphs(group_losses_train, group_losses_eval, group_accuracies_train, group_accuracies_eval, use_validation=True, print_img=False, save=True, path=None):
	if use_validation:
		text1 = 'Validation loss'
		text2 = 'Validation accuracy'
	else:
		text1 = 'Test loss'
		text2 = 'Test accuracy'

	n_groups = len(group_losses_eval)
	group_list = [(i+1)*10 for i in range(n_groups)]

	fig1, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,5))

	ax.plot(group_list, group_losses_train, linestyle='-', marker='o', label='Training loss')
	ax.plot(group_list, group_losses_eval, linestyle='-', marker='o', label=text1)

	ax.set_xlabel('Number of classes', labelpad=12, fontweight='bold')
	ax.set_ylabel('Loss', labelpad=12, rotation=90, fontweight='bold')

	ax.set_xticks(group_list)

	# ax.set_title('Incremental', pad=20, fontweight='bold')

	ax.legend()
	plt.grid(alpha=0.3)
	if print_img:
		plt.show()
	plt.close();

	# Plot accuracies
	fig2, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,5))

	ax.plot(group_list, group_accuracies_train, color='#7B1FA2', linestyle='-', marker='o', label='Training accuracy')
	ax.plot(group_list, group_accuracies_eval, color='#FFC107', linestyle='-', marker='o', label=text2)

	ax.set_xlabel('Number of classes', labelpad=12, fontweight='bold')
	ax.set_ylabel('Accuracy', labelpad=12, rotation=90, fontweight='bold')

	# ax.set_title('Accuracy', pad=20, fontweight='bold')

	ax.legend()
	plt.grid(alpha=0.3)
	if print_img:
		plt.show()
	plt.close();

	# Save figures
	if save:
		if path == None:
			print('FATAL ERROR - Dare un path come parametro al draw_final_graphs')
			sys.exit()

		fig1.savefig(path+'/group_loss.png')
		fig2.savefig(path+'/group_accuracy.png')

	return


def create_dir_for_current_group(group_number, path=None):
	if path == None:
		print('FATAL ERROR - Dare un path come parametro al create_dir_for_current_group')
		sys.exit()

	try:
	    os.makedirs(path+'/group_'+str(group_number))
	except OSError:
	    print ("FATAL ERROR - Creation of the directory of the current group failed")
	    sys.exit()

def dump_to_csv(losses_train, losses_eval, accuracies_train, accuracies_eval, group_number=-1, path=None):
	if path == None:
		print('FATAL ERROR - Dare un path come parametro al dump_to_csv')
		sys.exit()

	if len(accuracies_train) < len(accuracies_eval):
		df = pd.DataFrame({'losses_train': losses_train, 'losses_eval': losses_eval, 'accuracies_eval': accuracies_eval})
	else:
		df = pd.DataFrame({'losses_train': losses_train, 'losses_eval': losses_eval, 'accuracies_eval': accuracies_eval, 'accuracies_train': accuracies_train})

	df.to_csv(path+'/group_'+str(group_number)+'/values.csv', encoding='utf-8', index=False)


def dump_hyperparameters(path, lr, weight_decay, num_epochs, method, batch_size):

	df = pd.DataFrame({'Method': [method], 'LR': [lr], 'num_epochs': [num_epochs], 'batch_size': [batch_size], 'weight_decay': [weight_decay]})

	df.to_csv(path+'/hyperparameters.csv', encoding='utf-8', index=False)

def get_hyperparameter_string(lr, weight_decay, num_epochs, batch_size, multilrstep, gamma):
	return 'LR='+str(lr)+', weight_decay='+str(weight_decay)+', num_epochs='+str(num_epochs)+', batch_size='+str(batch_size)+', multilrstep='+str(multilrstep)+', gamma='+str(gamma)

def dump_final_values(losses_train, losses_eval, accuracies_train, accuracies_eval, path=None):
	if path == None:
		print('FATAL ERROR - Dare un path come parametro al dump_final_values')
		sys.exit()

	df = pd.DataFrame({'losses_train': losses_train, 'losses_eval': losses_eval, 'accuracies_eval': accuracies_eval, 'accuracies_train': accuracies_train})

	df.to_csv(path+'/final_values_for_each_group.csv', encoding='utf-8', index=False)

def eval_model(net, eval_dataloader, criterion, dataset_length, device, display=True, suffix=''):
	net.train(False)

	running_corrects_eval = 0
	cum_loss_eval = 0

	for images_eval, labels_eval in eval_dataloader:
	    images_eval = images_eval.to(device)
	    labels_eval = labels_eval.to(device)

	    # Forward Pass
	    outputs_eval = net(images_eval)

	    cum_loss_eval += criterion(outputs_eval, labels_eval).item()

	    # Get predictions
	    _, preds = torch.max(outputs_eval.data, 1)

	    # Update Corrects
	    running_corrects_eval += torch.sum(preds == labels_eval.data).data.item()

	# Calculate Accuracy
	accuracy_eval = running_corrects_eval / float(dataset_length)
	loss_eval = cum_loss_eval / float(dataset_length)
	
	if display:
		print('Loss on eval'+str(suffix)+':', loss_eval)
		print('Accuracy on eval'+str(suffix)+':', accuracy_eval)

	return loss_eval, accuracy_eval

def eval_model_accuracy(net, dataloader, dataset_length, device, display=True, suffix=''):
	net.train(False)

	running_corrects_train = 0

	for images_train, labels_train in dataloader:
	    images_train = images_train.to(device)
	    labels_train = labels_train.to(device)

	    # Forward Pass
	    outputs_train = net(images_train)

	    # Get predictions
	    _, preds = torch.max(outputs_train.data, 1)

	    # Update Corrects
	    running_corrects_train += torch.sum(preds == labels_train.data).data.item()

	# Calculate Accuracy
	accuracy_train = running_corrects_train / float(dataset_length)

	if display:
		print('Accuracy on train'+str(suffix)+':', accuracy_train)

	return accuracy_train

# def dump_on_gspreadsheet(path, link, method, losses_train, losses_eval, accuracies_train, accuracies_eval, use_validation, hyperparameters=None):
	
# 	auth.authenticate_user()
# 	gc = gspread.authorize(GoogleCredentials.get_application_default())
	
# 	# Open
# 	sheet = gc.open_by_url('https://docs.google.com/spreadsheets/d/1lxrz5nrHcYjzODCsvCoGal30N-beyxo3r65X9YPig6E/edit?usp=sharing')

# 	# select worksheet
# 	worksheet = sheet.worksheet('Foglio1')

# 	losses_train = '[' + ', '.join([str(elem) for elem in losses_train]) + "]" 
# 	losses_eval = '[' + ', '.join([str(elem) for elem in losses_eval]) + "]"
# 	accuracies_train = '[' + ', '.join([str(elem) for elem in accuracies_train]) + "]" 
# 	accuracies_eval = '[' + ', '.join([str(elem) for elem in accuracies_eval]) + "]" 
# 	values = [path, link, method, losses_train, losses_eval, accuracies_train, accuracies_eval, use_validation, hyperparameters]

# 	# Update with new values
# 	worksheet.append_row(values, value_input_option='USER_ENTERED')

# 	return

def dump_on_gspreadsheet(path, link, method, losses_train, losses_eval, accuracies_train, accuracies_eval, duration, use_validation, hyperparameters=None) :
	scope = ['https://www.googleapis.com/auth/spreadsheets']
	credentials = ServiceAccountCredentials.from_json_keyfile_name('/content/Incremental-learning-on-image-recognition/config/credentials.json', scope)

	gc = gspread.authorize(credentials)

	# Open
	sheet = gc.open_by_url('https://docs.google.com/spreadsheets/d/1lxrz5nrHcYjzODCsvCoGal30N-beyxo3r65X9YPig6E/edit?usp=sharing')

	# select worksheet
	worksheet = sheet.worksheet('Foglio1')

	losses_train = '[' + ', '.join([str(elem) for elem in losses_train]) + "]" 
	losses_eval = '[' + ', '.join([str(elem) for elem in losses_eval]) + "]"
	accuracies_train = '[' + ', '.join([str(elem) for elem in accuracies_train]) + "]" 
	accuracies_eval = '[' + ', '.join([str(elem) for elem in accuracies_eval]) + "]" 
	values = [path, link, method, str(duration), losses_train, losses_eval, accuracies_train, accuracies_eval, use_validation, str(hyperparameters)]

	# Update with new values
	worksheet.append_row(values, value_input_option='USER_ENTERED')

	return

def beep():
	# Play an audio beep. Any audio URL will do.
	output.eval_js('new Audio("https://upload.wikimedia.org/wikipedia/commons/0/05/Beep-09.ogg").play()')

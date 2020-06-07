import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
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
	# split indexes 
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
		raise RuntimeError("Dare mean e std alla funzione imgshow")

	mean = np.array(mean)
	std = np.array(std)
	for i in range(3):
		img[i] = img[i]*std[i] + mean[i] # unnormalize

	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.show()

	return

def compute_mean_and_std(dataset):
	"""**Compute means and stds to normalize**"""
	means_1 = []
	means_2 = []
	means_3 = []
	stds_1 = []
	stds_2 = []
	stds_3 = []

	for img, lab in dataset:      # For test dataset Normalization iter on test_dataset
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
	means = (mean_1.item(), mean_2.item(), mean_3.item())
	stds = (std_1.item(), std_2.item(), std_3.item())

	return means, stds

def compute_mean_and_std_res(means, stds, n):
	m0, m1, m2 = 0, 0, 0
	s0, s1, s2 = 0, 0, 0

	for m in means:
		m0 += m[0]
		m1 += m[1]
		m2 += m[2]

	for s in stds:
		s0 += s[0]
		s1 += s[1]
		s2 += s[2]

	means_res = (m0/n, m1/n, m2/n)
	stds_res = (s0/n, s1/n, s2/n)
	return means_res, stds_res

def draw_graphs(losses_train, losses_eval, accuracies_train, accuracies_eval, num_epochs, use_validation=True, print_img=False, save=False, path=None, group_number=None):
	if len(losses_eval) < num_epochs:
		return

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
			raise RuntimeError("Dare un path come parametro al draw_graphs")

		fig1.savefig(path+'/group_'+str(group_number)+'/loss.png')
		fig2.savefig(path+'/group_'+str(group_number)+'/accuracy.png')

def draw_final_graphs(group_losses_train, group_losses_eval, group_accuracies_eval_curr, group_accuracies_eval, is_joint_training=False, use_validation=True, print_img=False, save=True, path=None):
	if use_validation:
		text1 = 'Validation loss'
		text2 = 'Validation accuracy on all classes'
	else:
		text1 = 'Test loss'
		text2 = 'Test accuracy on all classes'

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

	if is_joint_training :
		ax.plot(group_list, group_accuracies_eval, color='#FFC107', linestyle='-', marker='o', label=text2)
	else:
		ax.plot(group_list, group_accuracies_eval_curr, color='#7B1FA2', linestyle='-', marker='o', label='Test accuracy on novel classes')
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
			raise RuntimeError("Dare un path come parametro al draw_final_graphs")

		fig1.savefig(path+'/group_loss.png')
		fig2.savefig(path+'/group_accuracy.png')

	return

def draw_final_graphs_nme(group_losses_train, group_accuracies_eval_nme, group_accuracies_eval, use_validation=False, print_img=False, save=True, path=None):
	if use_validation:
		text2 = 'Validation accuracy (hybrid 1)'
	else:
		text2 = 'Test accuracy (hybrid 1)'

	n_groups = len(group_losses_eval)
	group_list = [(i+1)*10 for i in range(n_groups)]

	fig1, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,5))

	ax.plot(group_list, group_losses_train, linestyle='-', marker='o', label='Training loss')

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

	ax.plot(group_list, group_accuracies_eval_nme, color='#7B1FA2', linestyle='-', marker='o', label='Test accuracy')
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
			raise RuntimeError("Dare un path come parametro al draw_final_graphs_nme")

		fig1.savefig(path+'/group_loss.png')
		fig2.savefig(path+'/group_accuracy.png')

	return

def create_dir_for_current_group(group_number, path=None):
	if path == None:
		raise RuntimeError("Dare un path come parametro al create_dir_for_current_group")

	try:
	    os.makedirs(path+'/group_'+str(group_number))
	except OSError:
		raise RuntimeError("Creation of the directory of the current group failed")

def dump_to_csv(losses_train, losses_eval, accuracies_train, accuracies_eval, group_number=-1, path=None):
	if path == None:
		raise RuntimeError("Dare un path come parametro al dump_to_csv")


	if len(losses_eval) < len(losses_train):
		return

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

def dump_final_values(losses_train, losses_eval, accuracies_train, accuracies_eval, accuracies_eval_curr, path=None):
	if path == None:
		raise RuntimeError("Dare un path come parametro al dump_final_values")

	if len(accuracies_eval_curr) < len(accuracies_eval):
		# JOINT TRAINING
		df = pd.DataFrame({'losses_train': losses_train, 'losses_eval': losses_eval, 'accuracies_eval': accuracies_eval, 'accuracies_train': accuracies_train})
	else:
		# NO JOINT TRAINING
		df = pd.DataFrame({'losses_train': losses_train, 'losses_eval': losses_eval, 'accuracies_eval': accuracies_eval, 'accuracies_eval_curr': accuracies_eval_curr, 'accuracies_train': accuracies_train})

	df.to_csv(path+'/final_values_for_each_group.csv', encoding='utf-8', index=False)

def dump_final_values_nme(losses_train, accuracies_train, accuracies_eval_nme, accuracies_eval, accuracies_eval_curr, path=None):
	if path == None:
		raise RuntimeError("Dare un path come parametro al dump_final_values")

	df = pd.DataFrame({'losses_train': losses_train, 'accuracies_eval_nme' : accuracies_eval_nme, 'accuracies_eval': accuracies_eval, 'accuracies_eval_curr': accuracies_eval_curr, 'accuracies_train': accuracies_train})

	df.to_csv(path+'/final_values_for_each_group.csv', encoding='utf-8', index=False)

def eval_model(net, eval_dataloader, criterion, dataset_length, use_bce_loss, ending_label, loss=True, device=None, display=True, suffix=''):
	net.train(False)

	running_corrects_eval = 0
	cum_loss_eval = 0

	for images_eval, labels_eval in eval_dataloader:
		images_eval = images_eval.to(device)
		labels_eval = labels_eval.to(device)

		# Forward Pass
		outputs_eval = net(images_eval)

		if loss : 
			batch_size = len(outputs_eval)

			if use_bce_loss:
				targets_bce = torch.zeros([batch_size, ending_label], dtype=torch.float32)
				for i in range(batch_size):
					targets_bce[i][labels_eval[i]] = 1

				targets_bce = targets_bce.to(device)

				cum_loss_eval += criterion(outputs_eval[:, 0:ending_label], targets_bce).item()
			else:
				# cum_loss_eval += criterion(outputs_eval, labels_eval).item()
				cum_loss_eval += criterion(outputs_eval[:, 0:ending_label], labels_eval).item()

		# Get predictions
		_, preds = torch.max(outputs_eval[:, 0:ending_label].data, 1)

		# Update Corrects
		running_corrects_eval += torch.sum(preds == labels_eval.data).data.item()

	# Calculate Accuracy
	accuracy_eval = running_corrects_eval / float(dataset_length)
	loss_eval = cum_loss_eval / float(dataset_length)
	
	if display:
		print('Loss on eval'+str(suffix)+':', loss_eval)
		print('Accuracy on eval'+str(suffix)+':', accuracy_eval)

	return loss_eval, accuracy_eval

def eval_model_accuracy(net, dataloader, dataset_length, starting_label, ending_label, device, display=True, suffix=''):
	net.train(False)

	running_corrects = 0

	for images, labels in dataloader:
		images = images.to(device)
		labels = labels.to(device)

		# Forward Pass
		outputs = net(images)

		# Get predictions
		_, preds = torch.max(outputs[:,starting_label:ending_label].data, 1)

		labels = labels - starting_label

		# Update Corrects
		running_corrects += torch.sum(preds == labels.data).data.item()

	# Calculate Accuracy
	accuracy = running_corrects / float(dataset_length)

	if display:
		print('Accuracy on '+str(suffix)+':', accuracy)

	return accuracy


def display_conf_matrix(conf_mat,display=False,save=False,path=None):
	n_classes = len(conf_mat)

	ticks = [i-0.5 for i in range(n_classes)]

	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))

	im = ax.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.jet)

	tick_strings = []
	for i in range(n_classes):
		if (i+1)%10 == 0:
			tick_strings.append(str(i+1))
		else:
			tick_strings.append('')

	ax.set(yticks=ticks, 
	       xticks=ticks,
	       yticklabels=tick_strings, 
			xticklabels=tick_strings)

	ax.yaxis.set_major_locator(ticker.IndexLocator(base=1, offset=0.5))
	ax.xaxis.set_major_locator(ticker.IndexLocator(base=1, offset=0.5))

	ax.tick_params(length=0, labelsize='14')

	ax.set_xlabel('Predicted class', labelpad=14, fontsize='16')
	ax.set_ylabel('True class', labelpad=14, rotation=90, fontsize='16')

	if display: 
		plt.show()
	plt.close();

	# Save figures
	if save:
		if path == None:
			raise RuntimeError("Devi passare il path alla funzione display_conf_matrix")

		fig.savefig(path+'/conf_matrix.png')
	
	return 

def get_conf_matrix(net, eval_dataloader, ending_label, device):
	net.train(False)
	# flag 
	FIRST = True 

	y_pred = None
	y_test = None

	for images_eval, labels_eval in eval_dataloader:
		images_eval = images_eval.to(device)
		labels_eval = labels_eval.to(device)

		# Forward Pass
		outputs_eval = net(images_eval)
		outputs_eval = outputs_eval[:,:ending_label]

		# Get predictions
		_, preds = torch.max(outputs_eval.data, 1)

		# concatenate predictions and labels
		if FIRST : 
			y_pred = preds.detach().cpu().clone()
			y_test = labels_eval.detach().cpu().clone()
			FIRST=False 
		else: 
			y_pred = torch.cat( (y_pred,preds.cpu()))
			y_test = torch.cat( (y_test,labels_eval.cpu()))
	
	return confusion_matrix(y_test, y_pred)

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

def dump_on_gspreadsheet(path, user, link, method, losses_train, losses_eval, accuracies_train, accuracies_eval, accuracies_eval_curr, duration, use_validation, hyperparameters=None) :
	scope = ['https://www.googleapis.com/auth/spreadsheets']
	credentials = ServiceAccountCredentials.from_json_keyfile_name('/content/Incremental-learning-on-image-recognition/config/credentials.json', scope)

	gc = gspread.authorize(credentials)

	# Open
	sheet = gc.open_by_url('https://docs.google.com/spreadsheets/d/1lxrz5nrHcYjzODCsvCoGal30N-beyxo3r65X9YPig6E/edit?usp=sharing')

	# select worksheet
	worksheet = sheet.worksheet('Foglio1')

	if user == 0:
		user_name = 'Roberto'
	elif user == 1:
		user_name = 'Alessandro'
	elif user == 2:
		user_name = 'Gabriele'
	else:
		raise RuntimeError('Dare uno user a dump_on_gspreadsheet che sia valido')

	losses_train = '[' + ', '.join([str(elem) for elem in losses_train]) + "]" 
	losses_eval = '[' + ', '.join([str(elem) for elem in losses_eval]) + "]"
	accuracies_train = '[' + ', '.join([str(elem) for elem in accuracies_train]) + "]" 
	accuracies_eval = '[' + ', '.join([str(elem) for elem in accuracies_eval]) + "]" 
	accuracies_eval_curr = '[' + ', '.join([str(elem) for elem in accuracies_eval_curr]) + "]" 
	values = [path, link, user_name, method, str(duration), losses_train, losses_eval, accuracies_train, accuracies_eval, accuracies_eval_curr, use_validation, str(hyperparameters)]

	# Update with new values
	worksheet.append_row(values, value_input_option='USER_ENTERED')

	return


def dump_on_gspreadsheet_nme(path, user, link, method, losses_train, accuracies_train, accuracies_eval_nme, accuracies_eval, accuracies_eval_curr, duration, use_validation, hyperparameters=None) :
	scope = ['https://www.googleapis.com/auth/spreadsheets']
	credentials = ServiceAccountCredentials.from_json_keyfile_name('/content/Incremental-learning-on-image-recognition/config/credentials.json', scope)

	gc = gspread.authorize(credentials)

	# Open
	sheet = gc.open_by_url('https://docs.google.com/spreadsheets/d/1lxrz5nrHcYjzODCsvCoGal30N-beyxo3r65X9YPig6E/edit?usp=sharing')

	# select worksheet
	worksheet = sheet.worksheet('icarl')

	if user == 0:
		user_name = 'Roberto'
	elif user == 1:
		user_name = 'Alessandro'
	elif user == 2:
		user_name = 'Gabriele'
	else:
		raise RuntimeError('Dare uno user a dump_on_gspreadsheet che sia valido')

	losses_train = '[' + ', '.join([str(elem) for elem in losses_train]) + "]" 
	accuracies_train = '[' + ', '.join([str(elem) for elem in accuracies_train]) + "]" 
	avg_incremental_accuracy = np.mean(accuracies_eval_nme)
	accuracies_eval_nme = '[' + ', '.join([str(elem) for elem in accuracies_eval_nme]) + "]" 
	accuracies_eval = '[' + ', '.join([str(elem) for elem in accuracies_eval]) + "]" 
	accuracies_eval_curr = '[' + ', '.join([str(elem) for elem in accuracies_eval_curr]) + "]" 
	values = [path, link, user_name, method, str(duration), losses_train, accuracies_train, avg_incremental_accuracy, accuracies_eval_nme, accuracies_eval, accuracies_eval_curr, use_validation, str(hyperparameters)]

	# Update with new values
	worksheet.append_row(values, value_input_option='USER_ENTERED')

	return

def beep():
	# Play an audio beep. Any audio URL will do.
	output.eval_js('new Audio("https://upload.wikimedia.org/wikipedia/commons/0/05/Beep-09.ogg").play()')

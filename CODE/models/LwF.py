import numpy as np
import matplotlib.pyplot as plt
import sys
import torch
import torchvision
import math
import random
from torch.utils.data import Subset


class LwF() :
    '''Learning without Forgetting (LwF) class implemented as described in iCaRL paper'''
    def __init__(self, dataset, batch_size=0, K=2000, device='cuda') :

        self.device = device
        self.batch_size = batch_size
        self.K = K       # max number of exemplars
        self.dataset = dataset

    def get_indexes_from_label(self, label):
        targets = self.dataset.targets
        indexes = []

        for i,target in enumerate(targets):
            if target == label:
                indexes.append(i)

        return indexes

    def L2_norm(self, features): 
        # L2-norm on rows

        #return [feature/torch.sqrt(torch.sum(torch.square(feature))).item() for feature in features]
        features_norm = torch.zeros((features.size(0),features.size(1)), dtype=torch.float64).to(self.device)

        for i,feature in enumerate(features):
            square = torch.square(feature)
            somma = torch.sum(square)
            sqrt = torch.sqrt(somma).item()
            features_norm[i] += feature/sqrt
            #print(feature/sqrt)

        return features_norm

    def bce_loss_with_logits(self, net, net_old, criterion, images, labels, current_classes, starting_label, ending_label) :

        # Forward pass to the network
        outputs = net(images)

        if starting_label == 0:
            targets_bce = torch.zeros([self.batch_size, ending_label], dtype=torch.float32)
            for i in range(self.batch_size):
                targets_bce[i][labels[i]] = 1

            targets_bce = targets_bce.to(self.device)

            loss = criterion(outputs[:, 0:ending_label], targets_bce)
        else:
            with torch.no_grad():
                outputs_old = net_old(images)
                sigmoids_old = torch.sigmoid(outputs_old[:,0:starting_label])

            targets_bce = torch.zeros([self.batch_size, ending_label], dtype=torch.float32)
            for i in range(self.batch_size):
                if labels[i] in current_classes:
                    # nuovo
                    targets_bce[i][labels[i]] = 1.

                targets_bce[i,0:starting_label] = sigmoids_old[i]

            targets_bce = targets_bce.to(self.device)

            loss = criterion(outputs[:, 0:ending_label], targets_bce)

        return loss
    
    def update_representation(self, net, net_old, train_dataloader_cum_exemplars, criterion, optimizer, current_classes, starting_label, ending_label, current_step, bce_var=1) :
        FIRST = True
        ###net.train() # Sets module in training mode (lo facciamo già nel main di iCaRL)

        # Iterate over the dataset
        for images, labels in train_dataloader_cum_exemplars :
            # Bring data over the device of choice
            images = images.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad() # Zero-ing the gradients
            
            loss = self.bce_loss_with_logits(net, net_old, criterion, images, labels, current_classes, starting_label, ending_label, bce_var=bce_var)            

            if current_step == 0 and FIRST:
                print('--- Initial loss on train: {}'.format(loss.item()))
                FIRST = False 

            # Compute gradients for each layer and update weights
            loss.backward()  # backward pass: computes gradients
            optimizer.step() # update weights based on accumulated gradients

        return loss


    ## FINO A QUA
    def eval_model_nme(self, net, test_dataloader, dataset_length, display=True, suffix=''):

        running_corrects = 0
        for images,labels in test_dataloader:
            # Bring data over the device of choice
            images = images.to(self.device)
            labels = labels.to(self.device)

            net.train(False)

            # feature map (custom)
            features = net.feature_map(images)

            # normalization
            features = self.L2_norm(features)

            for i,sample in enumerate(features):
                dots = torch.tensor([torch.dot(mean, sample).data for mean in self.means_of_each_class])
                y_pred = torch.argmax(dots).item()
                if y_pred == labels[i] : 
                    running_corrects+=1

        accuracy_eval = running_corrects / float(dataset_length)

        if display :    
            print('Accuracy on eval NME'+str(suffix)+':', accuracy_eval)

        return accuracy_eval

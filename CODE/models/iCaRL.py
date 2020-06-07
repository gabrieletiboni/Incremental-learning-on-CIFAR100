import numpy as np
import matplotlib.pyplot as plt
import sys
import torch
import torchvision
import math
import random
from torch.utils.data import Subset


class iCaRL() :

    def __init__(self, dataset, batch_size=0, K=2000, device='cuda') :
        self.device = device
        self.batch_size = batch_size
        self.K = K
        self.exemplars = [list() for i in range(100)]
        self.means_of_each_class = None
        self.dataset = dataset

    def flattened_exemplars(self) :
        flat_list = []
        for sublist in self.exemplars:
            for item in sublist:
                flat_list.append(item)
        return flat_list

    def get_indexes_from_label(self, label):
        targets = self.dataset.targets
        indexes = []

        for i,target in enumerate(targets):
            if target == label:
                indexes.append(i)

        return indexes

    def construct_exemplars(self, net, s, t, herding=False):
        # dataloader: contins only current classes
        # s = startng labels 
        # t = ending label
        m = math.floor(self.K / t)
        count_per_class = []

        # iteriamo sulle nuove classi
        for c in range(s, t) :
            indexes = self.get_indexes_from_label(c)
            samples_of_this_class = Subset(self.dataset, indexes)

            samples_of_this_class_python = [(image, indexes[index]) for index,image in enumerate(samples_of_this_class)]
            random.shuffle(samples_of_this_class_python)

            if not herding :
                for i in range(m) :
                    self.exemplars[c] = samples_of_this_class_python[i][1]
            else :
                # compute features
                pass

        return

    def reduce_exemplars(self,s,t):
        # m = target number of exemplars
        m = math.floor(self.K / t)

        for i in range(s) : 
            self.exemplars[i] = self.exemplars[i][:m]

        return

    def L2_norm(self, features): 
        # L2-norm on rows
        return [feature/torch.sqrt(torch.sum(torch.square(feature)).data) for feature in features]

    def compute_means(self, net, dataloader, ending_label):
        sums = torch.zeros((ending_label,64), dtype=torch.float64).to(self.device)
        counts = torch.zeros(ending_label, dtype=torch.int32).to(self.device)

        with torch.no_grad() : 
            for images,labels in dataloader :
                # Bring data over the device of choice
                images = images.to(self.device)
                labels = labels.to(self.device)

                net.train(False)
                # feature map (custom)
                features = net.feature_map(images)
                # print(features.size()) #should be BATCH_SIZE x 64

                # normalization
                features = self.L2_norm(features)

                for i,sample in enumerate(features) :
                    sums[labels[i]] += sample 
                    counts[labels[i]] += 1

            means_of_each_class = [sums[i]/float(count) for i,count in enumerate(counts)]
            
            print(means_of_each_class)

            self.means_of_each_class = torch.tensor(self.L2_norm(means_of_each_class)).to(self.device)
            print("----->>>>>  Check the norm  <<<<<-----")

        return

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
                sigmoids_old = torch.sigmoid(outputs_old)

            targets_bce = torch.zeros([self.batch_size, ending_label], dtype=torch.float32)
            for i in range(self.batch_size):
                if labels[i] in current_classes:
                    # nuovo
                    targets_bce[i][labels[i]] = 1.

                targets_bce[i,0:starting_label] = sigmoids_old[i]

            targets_bce = targets_bce.to(self.device)

            loss = criterion(outputs[:, 0:ending_label], targets_bce)

        return loss
    
    def eval_model_nme(self, net, test_dataloader, dataset_length, display=True, suffix='') :
        
        for images,labels in test_dataloader :
            # Bring data over the device of choice
            images = images.to(self.device)
            labels = labels.to(self.device)

            net.train(False)

            # feature map (custom)
            features = net.feature_map(images)

            # normalization
            features = self.L2_norm(features)

            running_corrects = 0
            for i,sample in enumerate(features):
                dots = torch.tensor([torch.dot(mean, sample).data for mean in self.means_of_each_class])
                y_pred = torch.argmax(dots).item()
                if y_pred == labels[i] : 
                    running_corrects+=1

        accuracy_eval = running_corrects / float(dataset_length)

        if display :    
            print('Accuracy on eval NME'+str(suffix)+':', accuracy_eval)

        return accuracy_eval

    def update_representation(self, net, net_old, train_dataloader_cum_exemplars, criterion, optimizer, current_classes, starting_label, ending_label, current_step) :
        FIRST = True
        # Iterate over the dataset
        for images, labels in train_dataloader_cum_exemplars :
            # Bring data over the device of choice
            images = images.to(self.device)
            labels = labels.to(self.device)

            net.train() # Sets module in training mode

            optimizer.zero_grad() # Zero-ing the gradients
            
            loss = self.bce_loss_with_logits(net, net_old, criterion, images, labels, current_classes, starting_label, ending_label)			

            if current_step == 0 and FIRST:
                print('--- Initial loss on train: {}'.format(loss.item()))
                FIRST = False 

            # Compute gradients for each layer and update weights
            loss.backward()  # backward pass: computes gradients
            optimizer.step() # update weights based on accumulated gradients

        return loss



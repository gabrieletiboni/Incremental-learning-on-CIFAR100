import torch
import torch.nn as nn

#
# L2 Loss
# L2Loss(outputs, targets)
# outputs -> shape BATCH_SIZE x NUM_CLASSES
# targets -> shape BATCH_SIZE x NUM_CLASSES
#
class L2Loss():
    
    # Constructor
    def __init__(self, reduction=None, alpha=1.0):
        default_reduction = 'mean'

        if reduction == None:
            self.reduction = default_reduction
        elif reduction == 'mean': # Mean on batch_size
            self.reduction = 'mean'
        elif reduction == 'hardmean': # Mean on batch_size and also number_of_classes
            self.reduction = 'hardmean'
        elif reduction == 'sum':
            self.reduction = 'sum'
        else:
            self.reduction = default_reduction

        self.alpha = alpha

        return

    # Methods
    def __call__(self, outputs, targets):

        batch_size = outputs.shape[0]
        num_classes = outputs.shape[1]

        losses = torch.zeros((batch_size), dtype=torch.float32).to('cuda')

        for i, (output, target) in enumerate(zip(outputs, targets)):
            losses[i] = torch.sum(self.alpha*torch.square(output-target))
            # print(losses[i], len(output), len(target), output, target)

        if self.reduction == 'mean':
            losses = torch.sum(losses)/batch_size
        elif self.reduction == 'hardmean':
            losses = torch.sum(losses)/batch_size/num_classes
        elif self.reduction == 'sum':
            losses = torch.sum(losses)

        return losses
	

def CE_L2_loss(net, net_old, criterion, images, labels, current_classes, starting_label, ending_label, distillation_weight=1, outputs_normalization='sigmoid', alpha=100):

    # Classification loss -> CE
    # Distillation loss -> L2

    CE_criterion = nn.CrossEntropyLoss(reduction='sum')
    L2_criterion = L2Loss(reduction='sum', alpha=alpha)
    softmax = torch.nn.Softmax(dim=-1)

    outputs = net(images)

    batch_size = outputs.shape[0]
    
    if outputs_normalization == 'softmax':
        outputs_normalized = softmax(outputs)
    elif outputs_normalization == 'sigmoid':
        outputs_normalized = torch.sigmoid(outputs)
    else:
        raise RuntimeError('Errore nella scelta outputs_normalization in CE_L2')

    if starting_label == 0:
        loss = CE_criterion(outputs, labels)/batch_size
    else:
        with torch.no_grad():
            net_old.train(False)
            outputs_old = net_old(images)
            # sigmoids_old = torch.sigmoid(outputs_old[:,0:starting_label])
            if outputs_normalization == 'softmax':
                probabilities_old = softmax(outputs_old)
            elif outputs_normalization == 'sigmoid':
                probabilities_old = torch.sigmoid(outputs_old)

        ce_loss = CE_criterion(outputs, labels)#/batch_size

        targets = probabilities_old[:, :starting_label].to('cuda')
        dist_loss = L2_criterion(outputs_normalized[:, :starting_label], targets)#/batch_size

        # print(f"[CE loss: {ce_loss.item()} | Dist loss: {dist_loss.item()}")

        loss = (ce_loss + (distillation_weight*dist_loss))/batch_size

    return loss

def L2_L2_loss(net, net_old, criterion, images, labels, current_classes, starting_label, ending_label, distillation_weight=1, outputs_normalization='sigmoid', alpha=100, bce_var=2):

	# Classification loss -> L2
	# Distillation loss -> L2

	L2_criterion = L2Loss(reduction='hardmean', alpha=alpha)
	softmax = torch.nn.Softmax(dim=-1)

	outputs = net(images)

	batch_size = outputs.shape[0]

	if outputs_normalization == 'softmax':
	    outputs_normalized = softmax(outputs)
	elif outputs_normalization == 'sigmoid':
	    outputs_normalized = torch.sigmoid(outputs)
	else:
	    raise RuntimeError('Errore nella scelta outputs_normalization in L2_L2')

	if bce_var == 2:
		ending_label = 100

	if starting_label == 0:
		one_hot_targets = torch.zeros([batch_size, ending_label], dtype=torch.float32)
		# one hot encoding
		for i in range(batch_size):
		    one_hot_targets[i][labels[i]] = 1
		
		one_hot_targets = one_hot_targets.to('cuda')
		loss = L2_criterion(outputs_normalized, one_hot_targets)
	else:
		with torch.no_grad():
			net_old.train(False)
			outputs_old = net_old(images)
			# sigmoids_old = torch.sigmoid(outputs_old[:,0:starting_label])
			if outputs_normalization == 'softmax':
				probabilities_old = softmax(outputs_old)
			elif outputs_normalization == 'sigmoid':
				probabilities_old = torch.sigmoid(outputs_old)

		one_hot_targets = torch.zeros([batch_size, ending_label], dtype=torch.float32)
		# one hot encoding
		for i in range(batch_size):
			one_hot_targets[i,0:starting_label] = probabilities_old[i, :starting_label]

			if labels[i] in current_classes:
				one_hot_targets[i][labels[i]] = 1

		
		one_hot_targets = one_hot_targets.to('cuda')

	    # clf_loss = L2_criterion(outputs_normalized, one_hot_targets)/batch_size

	    # clf_loss = L2_criterion(outputs, labels)#/batch_size

	    # targets = probabilities_old[:, :starting_label].to('cuda')
	    # dist_loss = L2_criterion(outputs_normalized[:, :starting_label], targets)#/batch_size

	    # print(f"[CE loss: {ce_loss.item()} | Dist loss: {dist_loss.item()}")

	    # loss = (clf_loss + (distillation_weight*dist_loss))/batch_size

		loss = L2_criterion(outputs_normalized, one_hot_targets)

	return loss

def BCE_L2_loss(net, net_old, criterion, images, labels, current_classes, starting_label, ending_label, distillation_weight=1, outputs_normalization='sigmoid', alpha=100):

    # Binary Classification loss -> BCE (new)
    # Distillation loss -> L2 (old net)

    BCE_criterion = nn.BCEWithLogitsLoss(reduction='sum')
    L2_criterion = L2Loss(reduction='sum', alpha=alpha)

    softmax = torch.nn.Softmax(dim=-1)

    outputs = net(images)

    batch_size = outputs.shape[0] 
    
    if outputs_normalization == 'softmax':
        outputs_normalized = softmax(outputs)
    elif outputs_normalization == 'sigmoid':
        outputs_normalized = torch.sigmoid(outputs)
    else:
        raise RuntimeError('Errore nella scelta outputs_normalization in BCE_L2')
    
    # BCE_VAR=2
    ending_label = 100

    if starting_label == 0:
        # BCE computed da starting label fino a 100
        
        # first group of classes -> just BCE (no L2 distillation)
        one_hot_targets = torch.zeros([batch_size, ending_label], dtype=torch.float32)
        # one hot encoding
        for i in range(batch_size):
            one_hot_targets[i][labels[i]] = 1

        one_hot_targets = one_hot_targets.to('cuda')
        ## ONE-HOT
        # print(one_hot_targets.size()) 
        # print(outputs.size()) # torch.Size([128, 100])
        # print(outputs[:,0:ending_label].size())
        loss = BCE_criterion(outputs, one_hot_targets)/batch_size/100
    else:
        # BCE

        # compute old output
        with torch.no_grad():
            net_old.train(False)
            outputs_old = net_old(images)
            # sigmoids_old = torch.sigmoid(outputs_old[:,0:starting_label])
            if outputs_normalization == 'softmax':
                probabilities_old = softmax(outputs_old)
            elif outputs_normalization == 'sigmoid':
                probabilities_old = torch.sigmoid(outputs_old)
        
        one_hot_targets = torch.zeros([batch_size, ending_label], dtype=torch.float32)
        # one-hot encoding
        for i in range(batch_size):
            # old labels
            one_hot_targets[i,0:starting_label] = probabilities_old[i, :starting_label]

            # new labels (current group of classes)
            if labels[i] in current_classes:
                one_hot_targets[i][labels[i]] = 1
        
        # one_hot_targets contains one-hot encoding of all classes seen until now (cumulative)
        one_hot_targets = one_hot_targets.to('cuda')

        bce_loss = BCE_criterion(outputs[:,starting_label:ending_label], one_hot_targets[:,starting_label:ending_label]) #/batch_size 

        # test_sigmoid_outputs = softmax(outputs)
        # print('Some initial outputs:', test_sigmoid_outputs[0, labels[0]], test_sigmoid_outputs[1, labels[1]], test_sigmoid_outputs[2, labels[2]])
        # for i in range(len(outputs)):
        #     print('i',i,'- ', test_sigmoid_outputs[i, labels[i]].item())

        # MSE - distillation loss on old outputs (old net)
        targets = probabilities_old[:, :starting_label].to('cuda')
        dist_loss = L2_criterion(outputs_normalized[:, :starting_label], targets) #/(batch_size*)

        #print(f"[BCE loss: {bce_loss.item()} | L2/MSE loss: {dist_loss.item()}")

        loss = (bce_loss + dist_loss)/batch_size/100 #/batch_size
        #print(loss.item())

    return loss
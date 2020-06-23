import torch
import torch.nn as nn

# L2 Loss
# L2Loss(outputs, targets)
# outputs -> shape BATCH_SIZE x NUM_CLASSES
# targets -> shape BATCH_SIZE x NUM_CLASSES

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

def BCE_L2_loss(net, net_old, criterion, images, labels, current_classes, starting_label, ending_label, distillation_weight=1, outputs_normalization='sigmoid', alpha=100):

    # Binary Classification loss -> BCE (new)
    # Distillation loss -> L2 (old net)

    BCE_criterion = nn.BCEWithLogitsLoss(reduction='mean') 
    L2_criterion = L2Loss(reduction='sum', alpha=100)
    softmax = torch.nn.Softmax(dim=-1)

    outputs = net(images)

    batch_size = outputs.shape[0] 
    
    if outputs_normalization == 'softmax':
        outputs_normalized = softmax(outputs)
    elif outputs_normalization == 'sigmoid':
        outputs_normalized = torch.sigmoid(outputs)
    else:
        raise RuntimeError('Errore nella scelta outputs_normalization in BCE_L2')

    if starting_label == 0:

        ## ONE HOT
        loss = BCE_criterion(outputs, labels)/batch_size
    else:
        with torch.no_grad():
            net_old.train(False)
            outputs_old = net_old(images)
            # sigmoids_old = torch.sigmoid(outputs_old[:,0:starting_label])
            if outputs_normalization == 'softmax':
                probabilities_old = softmax(outputs_old)
            elif outputs_normalization == 'sigmoid':
                probabilities_old = torch.sigmoid(outputs_old)

        ce_loss = BCE_criterion(outputs, labels) #/batch_size
        
        test_sigmoid_outputs = softmax(outputs)
        print('Some initial outputs:', test_sigmoid_outputs[0, labels[0]], test_sigmoid_outputs[1, labels[1]], test_sigmoid_outputs[2, labels[2]])
        for i in range(len(outputs)):
            print('i',i,'- ', test_sigmoid_outputs[i, labels[i]].item())

        targets = probabilities_old[:, :starting_label].to('cuda')
        dist_loss = L2_criterion(outputs_normalized[:, :starting_label], targets) #/batch_size

        print(f"[CE loss: {ce_loss.item()} | Dist loss: {dist_loss.item()}")

        loss = (ce_loss + (distillation_weight*dist_loss))/batch_size

    return loss
# MAIN
import torch

numclass=10
feature_extractor=resnet32_cbam()
img_size=32
batch_size=128
task_size=10
memory_size=2000
epochs=70
learning_rate=2.0

model=iCaRLmodel(numclass,feature_extractor,batch_size,task_size,memory_size,epochs,learning_rate)

for i in range(10):
    model.beforeTrain()
    accuracy=model.train()
    model.afterTrain(accuracy)
    #batch_test_accuracy.append(accuracy)
    print('accuracy:%.3f' % accuracy)
   # batch_test_accuracy.append((j, i) for j in accuracy)
   # batch_val_accuracy.append((j, i) for j in val_accuracy)

# CLASSE
class iCaRLmodel(LWF):
 
    def _init_(self,numclass,feature_extractor,batch_size,task_size,memory_size,epochs,learning_rate):
          super()._init_(numclass,feature_extractor,batch_size,task_size,memory_size,epochs,learning_rate)
          self.exemplar_set = []
          self.class_mean_set = []
          self.index = 0

    def beforeTrain(self):
        self.net.eval()
        classes=[self.numclass-self.task_size,self.numclass]
        self.train_loader,self.test_loader=self._get_train_and_test_dataloader(classes)
        #from second iter
        if self.numclass>self.task_size:
            #modify model according to new classes
            self.net.Incremental_learning(self.numclass)
        
        self.net.train()
        self.net.to(device)

    def get_one_hot(self,target,num_class):
        one_hot=torch.zeros(target.shape[0],num_class).to(device)
        one_hot=one_hot.scatter(dim=1,index=target.long().view(-1,1),value=1.)
        return one_hot

    def _compute_loss(self, indexs, imgs, target,loss):
        output = self.net(imgs)
        target = self.get_one_hot(target, 100)
        output, target = output.to(device), target.to(device)
        if loss == "BCE_Logits": criterion= BCEWithLogitsLoss()
        if loss == "hinge" : criterion= torch.nn.HingeEmbeddingLoss(margin=1.0, 
                                                 size_average=None, reduce=None, reduction='mean')
        if loss == "MSE" : criterion = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
        if loss == "Cross" : criterion = torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
        if self.old_model == None:
            return criterion(output, target)
        else:
            old_target=torch.sigmoid(self.old_model(imgs)) # output immagini nuove predette su rete vecchia --> distribuzione di probabilità
            old_task_size = old_target.shape[1]
            target[..., :old_task_size] = old_target
            return criterion(output, target)

    def train(self):
        accuracy = 0
        opt = optim.SGD(self.net.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=0.00001)
        for epoch in range(self.epochs):
            if epoch == 48:
                if self.numclass==self.task_size:
                     print(1)
                     opt = optim.SGD(self.net.parameters(), lr=2/5, weight_decay=0.00001,)
                else:
                     for p in opt.param_groups:
                         p['lr'] = self.learning_rate/ 5
                     #opt = optim.SGD(self.model.parameters(), lr=self.learning_rate/ 5,weight_decay=0.00001,momentum=0.9,nesterov=True,)
                print("change learning rate:%.3f" % (self.learning_rate / 5))
            elif epoch == 62:
                if self.numclass>self.task_size:
                     for p in opt.param_groups:
                         p['lr'] =self.learning_rate/ 25
                     #opt = optim.SGD(self.model.parameters(), lr=self.learning_rate/ 25,weight_decay=0.00001,momentum=0.9,nesterov=True,)
                else:
                     opt = optim.SGD(self.net.parameters(), lr=2/25, weight_decay=0.00001)
                print("change learning rate:%.3f" % (self.learning_rate / 25))
            elif epoch == 80:
                  if self.numclass==self.task_size:
                     opt = optim.SGD(self.net.parameters(), lr=1.0 / 125,weight_decay=0.00001)
                  else:
                     for p in opt.param_groups:
                         p['lr'] =self.learning_rate/ 125
                     #opt = optim.SGD(self.model.parameters(), lr=self.learning_rate / 125,weight_decay=0.00001,momentum=0.9,nesterov=True,)
                  print("change learning rate:%.3f" % (self.learning_rate / 100))
            for step, (indexs, images, target) in enumerate(self.train_loader):
                # load train images and target into model
                images, target = images.to(device), target.to(device)
                #compute bce loss
                loss_value = self._compute_loss(indexs, images, target,'BCE_Logits') #passa parametro per cambiare il loss
                #zeroes grad structures 
                opt.zero_grad()
                #calculate 
                loss_value.backward()
                opt.step()
                if step%5 ==0: print('epoch:%d,step:%d,loss:%.3f' % (epoch, step, loss_value.item()))
            # calculate accuracy on validation set
        accuracy = self._test(self.test_loader, 1)
        print('epoch:%d,accuracy:%.3f' % (epoch, accuracy))
        return accuracy


    def afterTrain(self,accuracy):
        self.net.eval()
        m=int(self.memory_size/self.numclass)
        self._reduce_exemplar_sets(m)
        for i in range(self.numclass-self.task_size,self.numclass):
            print('construct class %s examplar:'%(i),end='')
            images=self.train_dataset.get_image_class(i)
            self._construct_exemplar_set(images,m)
        self.numclass+=self.task_size
        self.compute_exemplar_class_mean()
        self.net.train()
        KNN_accuracy=self._test(self.test_loader,0)
        print("NMS accuracy："+str(KNN_accuracy.item()))
        filename='accuracy:%.3f_KNN_accuracy:%.3f_increment:%d_net.pkl' % (accuracy, KNN_accuracy, i + 10)
        torch.save(self.net,filename)
        self.old_model=torch.load(filename)
        self.old_model.to(device)
        self.old_model.eval()


    def _construct_exemplar_set(self, images, m):
        class_mean, feature_extractor_output = self.compute_class_mean(images, self.transform)
        #class_var, feature_extractor_output_var = self.compute_class_variance(images, self.transform)
        #print(str(class_mean)+' '+str(class_var))
        exemplar = []
        now_class_mean = np.zeros((1, 64))
        #now_class_variance = np.zeros((1, 64))   
        
        for i in range(m):
  
            # shape：batch_size*64
            x = class_mean - (now_class_mean + feature_extractor_output) / (i + 1)
            # shape：batch_size
            x = np.linalg.norm(x, axis=1)
            index = np.argmin(x)   
            now_class_mean += feature_extractor_output[index]
            exemplar.append(images[index])
            
            
            

        print("the size of exemplar :%s" % (str(len(exemplar))))
        self.exemplar_set.append(exemplar)
        #self.exemplar_set.append(images)

    def _reduce_exemplar_sets(self, m):
        for index in range(len(self.exemplar_set)):
            self.exemplar_set[index] = self.exemplar_set[index][:m]
            print('Size of class %d examplar: %s' % (index, str(len(self.exemplar_set[index]))))

    def compute_class_mean(self, images, transform):
        x = self.Image_transform(images, transform).to(device)
        feature_extractor_output = F.normalize(self.net.feature_extractor(x).detach()).cpu().numpy()
        #feature_extractor_output = self.model.feature_extractor(x).detach().cpu().numpy()
        class_mean = np.mean(feature_extractor_output, axis=0)
        return class_mean, feature_extractor_output
    
    def compute_class_variance(self, images, transform):
        x = self.Image_transform(images, transform).to(device)
        feature_extractor_output = F.normalize(self.net.feature_extractor(x).detach()).cpu().numpy()
        #feature_extractor_output = self.model.feature_extractor(x).detach().cpu().numpy()
        class_mean = np.var(feature_extractor_output, axis=0)
        return class_mean, feature_extractor_output
   
    def Image_transform(self, images, transform):
        data = transform(Image.fromarray(images[0])).unsqueeze(0)
        for index in range(1, len(images)):
            data = torch.cat((data, self.transform(Image.fromarray(images[index])).unsqueeze(0)), dim=0)
        return data

    def compute_exemplar_class_mean(self):
        self.class_mean_set = []
        for index in range(len(self.exemplar_set)):
            print("compute the class mean of %s"%(str(index)))
            exemplar=self.exemplar_set[index]
            class_mean, _ = self.compute_class_mean(exemplar, self.transform)
            class_mean_,_=self.compute_class_mean(exemplar,self.classify_transform)
            class_mean=(class_mean/np.linalg.norm(class_mean)+class_mean_/np.linalg.norm(class_mean_))/2
            self.class_mean_set.append(class_mean)
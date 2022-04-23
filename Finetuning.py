from backbones.ResNet import resnet50_cbam
from torch.nn import functional as F
import torch.optim as optim
from datasets.dataset_maker import dataset_maker
from myNetwork import network
from thop.profile import profile
from torch.utils.data import DataLoader
from utils.configs import *


def get_one_hot(target, num_class):
    one_hot = torch.zeros(target.shape[0], num_class).to(device)
    one_hot = one_hot.scatter(dim=1, index=target.long().view(-1, 1), value=1.)
    return one_hot


class fine_model:

    def __init__(self, numclass, batch_size, epochs, learning_rate, task_size, dataset='UCMerced', img_size=64):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.exemplar_set = []
        self.class_mean_set = []
        self.feature_extractor = resnet50_cbam()

        # set datasets
        self.train_dataset, self.test_dataset = dataset_maker(dataset, img_size)
        self.numclass = numclass
        self.task_size = task_size
        self.batchsize = batch_size

        self.train_loader = None
        self.test_loader = None

    # get incremental train data
    # incremental
    def beforeTrain(self):
        classes = [self.numclass - self.task_size, self.numclass]
        self.train_loader, self.test_loader = self._get_train_and_test_dataloader(classes)
        self.model = network(self.task_size, self.feature_extractor)
        self.model.eval()
        self.model.train()
        self.model.to(device)

    def _get_train_and_test_dataloader(self, classes):
        self.train_dataset.getTrainData(classes, self.exemplar_set)
        self.test_dataset.getTestData(classes)
        train_loader = DataLoader(dataset=self.train_dataset,
                                  shuffle=True,
                                  batch_size=self.batchsize)

        test_loader = DataLoader(dataset=self.test_dataset,
                                 shuffle=True,
                                 batch_size=self.batchsize)

        return train_loader, test_loader

    # train model
    # compute loss
    # evaluate model
    def train(self):
        loss_list = []
        acc_list = []
        opt = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9, nesterov=True,
                        weight_decay=0.00001)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[48, 68, 85])
        for epoch in range(self.epochs):
            for step, (indexs, images, target) in enumerate(self.train_loader):
                images, target = images.to(device), target.to(device)
                opt.zero_grad()
                loss = self._compute_loss(images, target)
                opt.zero_grad()
                loss.backward()
                opt.step()
                print('epoch:%d,step:%d,loss:%.3f' % (epoch, step, loss.item()))
                loss_list.append(loss.item())

            accuracy = self._test(self.test_loader)
            acc_list.append(accuracy)

            print('epoch:%d,accuracy:%.5f' % (epoch, accuracy))
            # save acc&loss
            file_path = 'result/loss_fine_%d.txt' % (self.numclass)
            filename = open(file_path, 'w')
            for value in loss_list:
                filename.write(str(value))
            filename.close()

            file_path = 'result/acc_fine_%d.txt' % (self.numclass)
            filename = open(file_path, 'w')
            for value in acc_list:
                filename.write(str(value))
            filename.close()
        scheduler.step()
        self._test_all_tasks()
        return accuracy

    def _test(self, testloader, i=None):
        self.model.eval()
        correct, total = 0.0, 0.0
        for setp, (indexs, imgs, labels) in enumerate(testloader):
            if i == None:
                labels = labels - self.numclass + self.task_size
            else:
                labels = labels - self.task_size * i
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                outputs = self.model(imgs)
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)
        accuracy = correct.item() / total
        self.model.train()
        return accuracy

    def _test_all_tasks(self):
        acc_all_tasks = []
        tasks_num = int(self.numclass / self.task_size)
        for i in range(tasks_num):
            classes = [self.task_size * i, self.task_size * i + self.task_size]
            _, test_loader = self._get_train_and_test_dataloader(classes)
            accuracy = self._test(test_loader, i)
            print('test accuracy of task_%d = %.5f' % ((i + 1), accuracy))
            acc_all_tasks.append(accuracy)
        avg_acc = sum(acc_all_tasks) / tasks_num
        acc_all_tasks.append(avg_acc)
        # save acc of integration
        file_path = 'result/acc_all_fine_%d.txt' % (self.numclass)
        filename = open(file_path, 'w')
        for value in acc_all_tasks:
            filename.write(str(value))
        filename.close()

    def _compute_loss(self, imgs, target):
        output = self.model(imgs)
        target = target - self.numclass + self.task_size
        target = get_one_hot(target, self.task_size)
        output, target = output.to(device), target.to(device)
        return F.binary_cross_entropy_with_logits(output, target)

    def afterTrain(self, accuracy):
        filename = './model/5_increment:%d_net.pkl' % (self.numclass - self.task_size)
        torch.save(self.model, filename)
        inputs = torch.randn((1, 3, 64, 64)).to(device)
        flops, params = profile(self.model, (inputs,), verbose=False)
        print('param:' + str(params) + "|FLOPs:" + str(flops))

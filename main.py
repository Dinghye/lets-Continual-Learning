from Finetuning import fine_model
from utils.configs import class_list

# Main args are here, see more config in utils/configs.py
parser = 3

batch_size = 12
epochs = 1
learning_rate = 0.1
img_size = 64
dataset = 'UCMerced'  # 'AID','RSICB256','NWPU','cifar10','cifar100'
numclass = int(class_list[dataset] / parser)
task_size = int(class_list[dataset] / parser)

for i in range(parser):
    print("**************learning task %d " % (i + 1))

    model = fine_model(numclass, batch_size, epochs, learning_rate, task_size, dataset=dataset,
                       img_size=img_size)
    model.beforeTrain()
    accuracy = model.train()
    model.afterTrain(accuracy)
    numclass = numclass + task_size

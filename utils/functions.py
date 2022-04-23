from utils import time
import matplotlib.pyplot as plt
import numpy as np


# plot feature map
def draw_features(width,height,x,savename):
    tic= time.time()
    fig = plt.figure(figsize=(16, 16))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    for i in range(width*height):
        plt.subplot(height,width, i + 1)
        plt.axis('off')
        # plt.tight_layout()
        img = x[0, i, :, :]
        pmin = np.min(img)
        pmax = np.max(img)
        img = (img - pmin) / (pmax - pmin + 0.000001)
        plt.imshow(img, cmap='gray')
        print("{}/{}".format(i,width*height))
    fig.savefig(savename, dpi=100)
    fig.clf()
    plt.close()
    print("time:{}".format(time.time() - tic))


# plot loss per task
def plot_loss_one_task(list,path):
    iters = range(len(list))

    plt.figure()

    plt.plot(iters, list, 'b', label='training loss')
    plt.title('Training loss')
    plt.xlabel('iters')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(path,dpi=300)
    # plt.show()


# plot accuracy per task
def plot_acc_one_task(list,path):
    iters = range(len(list))

    plt.figure()

    plt.plot(iters, list, 'g', label='test accuracy')
    plt.title('test accuracy')
    plt.xlabel('iters')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig(path,dpi=300)
    # plt.show()

# plot accuracy for all tasks



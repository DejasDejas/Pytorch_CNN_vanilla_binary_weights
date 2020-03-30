from torch.autograd import Variable
import torch.nn.functional as F
import time
from torch import save
import matplotlib.pyplot as plt
import numpy as np


# Training procedure
def train(use_gpu, model, train_loader, optimizer, slope):

    model.train()
    train_loss = 0
    train_acc = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_gpu:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model((data, slope))
        loss = F.nll_loss(output, target)
        acc = calculate_accuracy(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.data
        train_acc += acc.data

    train_loss /= len(train_loader)
    train_loss = train_loss.item()
    train_acc /= len(train_loader)
    train_acc = train_acc.item()
    return train_loss, train_acc


def training(use_gpu, model, names_model, nb_epoch, train_loader, valid_loader, optimizer, plot_result, get_slope):

    best_valid_loss = float('inf')
    loss_values_train = []
    acc_values_train = []
    loss_values_valid = []
    acc_values_valid = []

    for epoch in range(nb_epoch):
        slope = get_slope(epoch)
        print('# Epoch : {} - Slope : {}'.format(epoch, slope))
        start_time = time.time()
        train_loss, train_acc = train(use_gpu, model, train_loader, optimizer, slope)
        valid_loss, valid_acc = test(use_gpu, model, valid_loader, get_slope, epoch)

        loss_values_train.append(train_loss)
        acc_values_train.append(train_acc)
        loss_values_valid.append(valid_loss)
        acc_values_valid.append(valid_acc)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save(model.state_dict(), './trained_models/MNIST/' + names_model + '.pt')

        end_time = time.time()
        epoch_minutes, epoch_secs = epoch_time(start_time, end_time)

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_minutes}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

    if plot_result:
        plot_loss_acc(loss_values_train, acc_values_train, loss_values_valid, acc_values_valid, names_model)
    return loss_values_train, acc_values_train, loss_values_valid, acc_values_valid


# Testing procedure
def test(use_gpu, model, test_loader, get_slope, epoch):

    slope = get_slope(epoch)
    model.eval()
    test_loss = 0.0
    correct = 0.0
    for data, target in test_loader:
        if use_gpu:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model((data, slope))
        test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    test_acc = correct / len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, int(correct), len(test_loader.dataset),
        100. * test_acc))
    return test_loss, test_acc


def calculate_accuracy(fx, y):
    prediction = fx.argmax(1, keepdim=True)
    correct = prediction.eq(y.view_as(prediction)).sum()
    acc = correct.float() / prediction.shape[0]
    return acc


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_minutes = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_minutes * 60))
    return elapsed_minutes, elapsed_secs


def plot_loss_acc(loss_values_train, acc_values_train, loss_values_valid, acc_values_valid, name_model):
    # summarize history for accuracy
    plt.plot(np.array(acc_values_train))
    plt.plot(np.array(acc_values_valid))
    plt.title('model accuracy all_binary_model')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.xlim(0, 10)
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('results/MNIST_results/results_loss_acc/acc_model_' + name_model + '.png')
    plt.show()
    # summarize history for loss
    plt.plot(np.array(loss_values_train))
    plt.plot(np.array(loss_values_valid))
    plt.title('model loss all_binary_model')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.xlim(0, 10)
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('results/MNIST_results/results_loss_acc/loss_model_' + name_model + '.png')
    plt.show()
    
    return

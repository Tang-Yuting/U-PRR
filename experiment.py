import os
import torch.utils.data as Data
import torchvision
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
from matplotlib.pyplot import MultipleLocator

from datasets import make_training_dataset
from helper import load_dataset, get_matrix, get_model, get_loss


if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("gpu")


def get_args():
    parser = argparse.ArgumentParser(
        description='U-PRR implementation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--set_size', type=int, default=6000)
    parser.add_argument('--set_number', type=int, default=10)
    parser.add_argument('--class_number', type=int, default=10)
    parser.add_argument('--matrix', type=str, default='diagonal_same_matrix')
    parser.add_argument('--dia', type=int, default=0.5)
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--model', type=str, default='mlp')
    parser.add_argument('--loss', type=str, default='U-PRR')
    parser.add_argument('--batch_size', type=int, default=6000)
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--gradient_ascent', type=float, default=5)
    parser.add_argument('--combination', type=float, default=0.5)

    args = parser.parse_args()
    return args


def experiment(args):
    (x_train, y_train), (x_test, y_test) = load_dataset(args.dataset)
    matrix = get_matrix(args.matrix, args.dia)

    X, Y, S = make_training_dataset(((x_train, y_train), (x_test, y_test)), args.set_size, matrix,
                                    args.set_number, args.class_number)
    torch_dataset = Data.TensorDataset(X, S, Y)

    x_test = np.squeeze(x_test)
    x_test = x_test.reshape((10000, -1)).to(device)

    train_loader = torch.utils.data.DataLoader(torch_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=0, drop_last=True)

    torch.manual_seed(0)
    np.random.seed(0)

    model = get_model(args.model).to(device)
    criterion = get_loss(args.loss, matrix, args.class_number, args.set_number,
                         args.gradient_ascent, args.combination)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    y_loss_list = []
    for epoch in range(args.epoch):
        for step, (b_x, b_y, b_true_y) in enumerate(train_loader):
            b_x = np.squeeze(b_x)
            b_x = b_x.reshape((args.batch_size, -1)).to(device)
            y_pred = model(b_x).cpu()
            loss = criterion(b_y, y_pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        """----------------------------------test-----------------------------------"""
        y_test_pred = model(x_test).cpu()
        y_test_pred = torch.tensor(torch.max(y_test_pred, 1)[1].data.numpy(), dtype=torch.float64)
        y_diff = y_test_pred - y_test
        y_loss = torch.tensor((len(y_test) - len(y_diff[y_diff == 0])) / 10000, dtype=torch.float64)
        y_loss_list.append(y_loss)

        print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test loss: %.4f' % y_loss)

    # plot
    def figure_error(x, y, i):
        plt.plot(x, y, '.-', label=i, color='red')
        plt.title('Bag_Risk vs. Epoches')
        plt.ylabel('risk')

    x = range(0, args.epoch)
    figure_error(x, y_loss_list, 'Test Err')
    plt.legend(loc='lower left')
    plt.axis([0, 500, 0, 1])
    plt.grid()
    y_major_locator = MultipleLocator(0.1)
    ax = plt.gca()
    ax.yaxis.set_major_locator(y_major_locator)
    os.makedirs('MNIST_plot_baseline_picture', exist_ok=True)
    plt.savefig(f'./MNIST_plot_baseline_picture/(MINST)U-PRR.jpg')
    plt.show()

    return 0


if __name__ == '__main__':
    args = get_args()
    print("dataset: {}".format(args.dataset))
    print("loss: {}".format(args.loss))

    experiment(args)












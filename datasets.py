import os
import torchvision
import torch
import numpy as np
import torchvision.transforms as transforms
import wget


# get MNIST dataset
def get_mnist():
    DOWNLOAD_MNIST = True
    train_data = torchvision.datasets.MNIST(root='./mnist/',
                                            train=True,
                                            download=DOWNLOAD_MNIST,
                                            transform=torchvision.transforms.ToTensor())
    test_data = torchvision.datasets.MNIST(root='./mnist/',
                                           train=False,
                                           download=DOWNLOAD_MNIST,
                                           transform=torchvision.transforms.ToTensor())
    x_train = torch.unsqueeze(train_data.train_data, dim=1).type(torch.FloatTensor) / 255.
    y_train = train_data.train_labels
    x_test = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor) / 255.
    y_test = test_data.test_labels
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    return (x_train, y_train), (x_test, y_test)


# get Fashion_MNIST dataset
def get_Fashion_mnist():
    DOWNLOAD_F_MNIST = True
    train_data = torchvision.datasets.FashionMNIST(root='./F_mnist/',
                                                   train=True,
                                                   download=DOWNLOAD_F_MNIST,
                                                   transform=torchvision.transforms.ToTensor())
    test_data = torchvision.datasets.FashionMNIST(root='./F_mnist/',
                                                  train=False,
                                                  download=DOWNLOAD_F_MNIST,
                                                  transform=torchvision.transforms.ToTensor())
    x_train = torch.unsqueeze(train_data.train_data, dim=1).type(torch.FloatTensor) / 255.
    y_train = train_data.train_labels
    x_test = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor) / 255.
    y_test = test_data.test_labels
    return (x_train, y_train), (x_test, y_test)


# get Kuzushiji_MNIST dataset
def get_kuzushiji():
    files = ['kmnist-train-imgs.npz', 'kmnist-train-labels.npz', 'kmnist-test-imgs.npz', 'kmnist-test-labels.npz']
    url = 'http://codh.rois.ac.jp/kmnist/dataset/kmnist/'
    file_dir = 'kuzushiji_imagess/'
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    for file in files:
        if not os.path.exists(file_dir + file):
            file_url = url + file
            wget.download(file_url, file_dir + file)

    x_train = np.load('kuzushiji_imagess/kmnist-train-imgs.npz')['arr_0']
    y_train = np.load('kuzushiji_imagess/kmnist-train-labels.npz')['arr_0']
    x_test = np.load('kuzushiji_imagess/kmnist-test-imgs.npz')['arr_0']
    y_test = np.load('kuzushiji_imagess/kmnist-test-labels.npz')['arr_0']
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
    y_train = torch.from_numpy(y_train).type(torch.LongTensor)
    x_test = torch.from_numpy(x_test).type(torch.FloatTensor)
    y_test = torch.from_numpy(y_test).type(torch.LongTensor)
    return (x_train, y_train), (x_test, y_test)


def get_kuzushiji_49():
    x_train = np.load('kuzushiji_imagess/k49-train-imgs.npz')['arr_0']
    y_train = np.load('kuzushiji_imagess/k49-train-labels.npz')['arr_0']
    x_test = np.load('kuzushiji_imagess/k49-test-imgs.npz')['arr_0']
    y_test = np.load('kuzushiji_imagess/k49-test-labels.npz')['arr_0']
    x_train = x_train.reshape(232365, 784)
    x_test = x_test.reshape(38547, 784)
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
    y_train = torch.from_numpy(y_train).type(torch.LongTensor)
    x_test = torch.from_numpy(x_test).type(torch.FloatTensor)
    y_test = torch.from_numpy(y_test).type(torch.LongTensor)
    return (x_train, y_train), (x_test, y_test)


def get_cifar10():
    train_data = torchvision.datasets.CIFAR10(root='./data',
                                              train=True,
                                              download=True,
                                              transform=transforms.ToTensor())
    test_data = torchvision.datasets.CIFAR10(root='./data',
                                             train=False,
                                             download=True,
                                             transform=transforms.ToTensor())

    x_train = train_data.data
    y_train = torch.Tensor(train_data.targets)
    x_test = test_data.data
    y_test = torch.Tensor(test_data.targets)
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
    y_train = y_train.type(torch.LongTensor)
    x_test = torch.from_numpy(x_test).type(torch.FloatTensor)
    y_test = y_test.type(torch.LongTensor)

    x_train = x_train.permute(0, 3, 1, 2)
    x_test = x_test.permute(0, 3, 1, 2)
    return (x_train, y_train), (x_test, y_test)


def get_SVHN():
    DOWNLOAD_SVHN = True
    train_data = torchvision.datasets.SVHN(root='./svhn/',
                                            split = 'train',
                                            download=DOWNLOAD_SVHN,
                                            transform=torchvision.transforms.ToTensor())
    test_data = torchvision.datasets.SVHN(root='./svhn/',
                                           split = 'test',
                                           download=DOWNLOAD_SVHN,
                                           transform=torchvision.transforms.ToTensor())
    extra_data = torchvision.datasets.SVHN(root='./svhn/',
                                           split = 'extra',
                                           download=DOWNLOAD_SVHN,
                                           transform=torchvision.transforms.ToTensor())
    x_train = train_data.data.astype('float32') / 255.
    y_train = train_data.labels
    x_test = test_data.data.astype('float32') / 255.
    y_test = test_data.labels
    x_extra = extra_data.data.astype('float32') / 255.
    y_extra = extra_data.labels
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)
    x_extra = torch.from_numpy(x_extra)
    y_extra = torch.from_numpy(y_extra)

    x_train = torch.cat((x_train, x_extra), 0)
    y_train = torch.cat((y_train, y_extra), 0)
    return (x_train, y_train), (x_test, y_test)


def make_training_dataset(dataset, bag_size, class_prior, bag_number, class_number):

    torch.manual_seed(0)
    np.random.seed(0)

    def make_single_bag(dataset_, bag_size_, class_prior_):
        (trainX, trainY), (testX, testY) = dataset_
        labels = np.unique(trainY)
        X, Y = np.asarray(trainX, dtype=np.float32), np.asarray(trainY, dtype=np.int32)
        assert (len(X) == len(Y))
        perm = np.random.permutation(len(X))
        X, Y = X[perm], Y[perm]

        class_size = (np.array(bag_size_) * np.array(class_prior_)).astype(int)

        Y_index = Y.reshape(X.shape[0], )
        bag_X = X[Y_index == labels[0]][:class_size[0]]
        bag_Y = np.zeros((class_size[0], 1), dtype=np.int32)

        for i in range(class_number - 2):
            bag_X = np.concatenate((bag_X, X[Y_index == labels[i + 1]][:class_size[i + 1]]), axis=0)
            bag_Y = np.concatenate((bag_Y, (i + 1) * np.ones((class_size[i + 1], 1), dtype=np.int32)), axis=0)

        bag_X = np.concatenate((bag_X, X[Y_index == labels[class_number-1]][:(bag_size_-len(bag_X))]), axis=0)
        bag_Y = np.concatenate((bag_Y, (class_number-1) * np.ones(((bag_size_-len(bag_Y)), 1), dtype=np.int32)), axis=0)
        perm_ = np.random.permutation(len(bag_X))
        bag_X, bag_Y = bag_X[perm_], bag_Y[perm_]
        bag_X, bag_Y = torch.from_numpy(bag_X), torch.from_numpy(bag_Y)
        bag_Y = bag_Y.squeeze(dim=1)
        return bag_X, bag_Y

    X, Y = make_single_bag(dataset, bag_size, class_prior[0])
    S = np.zeros((bag_size, 1), dtype=np.int32)  # S: bag label

    for j in range(bag_number - 1):
        temp_X, temp_Y = make_single_bag(dataset, bag_size, class_prior[j+1])
        X = np.concatenate((X, temp_X), axis=0)
        Y = np.concatenate((Y, temp_Y), axis=0)
        S = np.concatenate((S, (j+1) * np.ones((bag_size, 1), dtype=np.int32)), axis=0)

    # print("len(X), len(Y), len(S)", len(X), len(Y), len(S))
    assert (len(X) == len(Y))
    assert (len(X) == len(S))
    perm = np.random.permutation(len(Y))
    X, Y, S = X[perm], Y[perm], S[perm]
    X, Y, S = torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(S)
    S = S.squeeze(dim=1)

    return X, Y, S


# biased proportion label
def add_label(X, S, theta, bag_size):
    bag_label = np.unique(S)
    S_index = S.reshape(X.shape[0], )
    bag_X = X[S_index == bag_label[0]][:bag_size]
    temp = torch.argmax(theta[bag_label[0]])
    bag_Y = temp * np.ones((bag_size, 1), dtype=np.int32)

    for i in range(len(bag_label) - 1):
        bag_X = np.concatenate((bag_X, X[S_index == bag_label[i+1]][:bag_size]), axis=0)
        temp = torch.argmax(theta[bag_label[i+1]])
        bag_Y = np.concatenate((bag_Y, temp * np.ones((bag_size, 1), dtype=np.int32)), axis=0)

    perm_ = np.random.permutation(len(bag_X))
    bag_X, bag_Y = bag_X[perm_], bag_Y[perm_]  # 打乱顺序
    bag_X, bag_Y = torch.from_numpy(bag_X), torch.from_numpy(bag_Y)
    bag_Y = bag_Y.squeeze(dim=1)
    return bag_X, bag_Y

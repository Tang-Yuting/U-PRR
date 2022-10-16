from datasets import get_mnist, get_kuzushiji, get_Fashion_mnist, get_kuzushiji_49, get_cifar10, get_SVHN,  make_training_dataset
from loss import CE_loss, Unbiased, U_PRR, U_correct, U_flood, Prop, VATLoss
from model import MLP, MLP_dropout, resnet20
from gene_matrix import generate_diagonal_same_matrix, generate_dia_dominate_matrix, gene_noise_diff


def load_dataset(dataset):
    if dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = get_mnist()
        return (x_train, y_train), (x_test, y_test)
    elif dataset == 'kuzushiji':
        (x_train, y_train), (x_test, y_test) = get_kuzushiji()
        return (x_train, y_train), (x_test, y_test)
    elif dataset == 'fashion':
        (x_train, y_train), (x_test, y_test) = get_Fashion_mnist()
        return (x_train, y_train), (x_test, y_test)
    elif dataset == 'kuzushiji_49':
        (x_train, y_train), (x_test, y_test) = get_kuzushiji_49()
        return (x_train, y_train), (x_test, y_test)
    elif dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = get_cifar10()
        return (x_train, y_train), (x_test, y_test)
    elif dataset == 'SVHN':
        (x_train, y_train), (x_test, y_test) = get_SVHN()
        return (x_train, y_train), (x_test, y_test)


def get_matrix(matrix, dia):
    if matrix == 'diagonal_same_matrix':
        return generate_diagonal_same_matrix(dia)
    elif matrix == 'dia_dominate_matrix':
        return generate_dia_dominate_matrix()


def get_model(model):
    if model == 'mlp':
        return MLP(28 * 28, 300, 300, 300, 300, 10)
    elif model == 'resnet':
        return resnet20()


def get_loss(loss, matrix, Class_Number, Bag_Number, GA, Comb):
    if loss == 'U-PRR':
        return U_PRR(matrix, Class_Number, Bag_Number, GA, Comb)

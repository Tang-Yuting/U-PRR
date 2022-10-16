import torch
import numpy as np

torch.manual_seed(3)
np.random.seed(3)


def generate_diagonal_same_matrix(dia):
    theta_eye = dia
    theta_one = (1 - dia)/10
    theta = torch.ones(10, 10) * theta_one + torch.eye(10) * theta_eye
    return theta


def generate_dia_dominate_matrix():
    torch.manual_seed(3)
    np.random.seed(3)
    matrix_ = np.random.rand(10, 10)/10
    matrix_sum = matrix_.sum(axis=1)
    matrix_sum_dia = torch.eye(10) * (1 - matrix_sum)
    matrix = matrix_sum_dia + matrix_
    return matrix


def gene_noise_diff(matrix, noise_rate):
    torch.manual_seed(3)
    np.random.seed(3)
    noisy_matrix_pu_ = np.random.uniform(-1, 1, (10, 10))
    noisy_matrix_pu = np.sign(noisy_matrix_pu_)
    noisy_matrix = (matrix * noisy_matrix_pu) * noise_rate
    noisy_matrix_sum = noisy_matrix.sum(axis=1)
    noisy_matrix_sum_dia = torch.eye(10) * noisy_matrix_sum
    matrix = noisy_matrix_sum_dia - noisy_matrix
    return matrix

import numpy as np

import torch
import torch.nn as nn

from einops import rearrange

from algorithms.libs.ordering import get_matrix


# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
# optical encoders
# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#


def sum_noise(y, snr):
    if snr > 0:
        sigma = torch.sum(torch.pow(y, 2)) / ((y.numel()) * 10 ** (snr / 10))
        noise = torch.normal(mean=0, std=torch.sqrt(sigma).item(), size=y.shape)
        noise = noise.to(y.device)
        return y + noise

    else:
        return y


class LinearSPC(nn.Module):
    def __init__(self, input_shape, compression_ratio, sensing_matrix='zig_zag', snr=-1, device='cuda'):
        super(LinearSPC, self).__init__()
        M, N, L = input_shape
        self.snr = snr

        if sensing_matrix in ['cake_cutting', 'zig_zag']:
            Hf = get_matrix(M * N, sensing_matrix).astype('float32')

        if sensing_matrix == 'random_binary':
            np.random.seed(0)
            Hf = np.random.randint(0, 2, (M * N, M * N)).astype('float32')
            # Hf = 2 * Hf - 1

        self.image_size = M
        self.num_measurements = M * N
        self.num_channels = L

        # build measurements

        H = Hf[:int(compression_ratio * self.num_measurements), :]
        self.HtH = torch.nn.Parameter(torch.from_numpy(np.matmul(H.T, H)), requires_grad=False).to(device)
        self.HtH_rhoI_inv = None

        self.H = torch.nn.Parameter(torch.from_numpy(H), requires_grad=False).to(device)
        self.H_inv = torch.nn.Parameter(torch.from_numpy(np.linalg.pinv(H)), requires_grad=False).to(device)

    def vec2image(self, vec):
        return vec.reshape(vec.size(0), self.num_channels, self.image_size, self.image_size).permute(0, 1, 3, 2)

    def image2vec(self, img):
        return rearrange(img, 'b c m n -> b (c n m)')

    def compute_HtHrhoIi(self, rho):
        with torch.no_grad():
            return torch.inverse(self.HtH + rho * torch.eye(self.HtH.shape[0], device='cuda'))

    def get_measurement(self, x):
        y = torch.matmul(self.H, x.T).T
        return sum_noise(y, self.snr)

    def get_transpose(self, y):
        return torch.matmul(self.H.T, y.T).T

    def get_inverse(self, x):
        return torch.matmul(self.H_inv, x.T).T

    def close_solution(self, x, y, beta, lambda_):
        # HtHrhoIi = self.compute_HtHrhoIi(beta)
        # return self.vec2image(torch.bmm(HtHrhoIi, b).squeeze(-1))
        A = self.HtH + beta * torch.eye(self.HtH.shape[0], device=x.device)
        b = (self.get_transpose(y) + self.image2vec(beta.unsqueeze(-1) * x - lambda_)).unsqueeze(-1)

        with torch.no_grad():
            return self.vec2image(torch.linalg.solve(A, b))

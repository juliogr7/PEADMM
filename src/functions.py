import pdb
import torch


class MeasurementError:
    def __init__(self, A, b, some=False):
        self.A, self.b = A, b
        self.A = self.A.to(b.device)
        self._A_u, self._A_s, self._A_v = None, None, None
        self._A_eigv, self._At_A = None, None
        self.some = some

    @property
    def At_A(self):
        """Matrix A.t() @ A"""
        if self._At_A is not None:
            return self._At_A
        self._At_A = self.A.t() @ self.A
        return self._At_A

    @property
    def A_u(self):
        """U matrix from SVD(A)"""
        if self._A_u is not None:
            return self._A_u
        self._A_u, self._A_s, self._A_v = torch.svd(self.A, some=self.some)
        return self._A_u

    @property
    def A_s(self):
        """Vector of singular values of A"""
        if self._A_s is not None:
            return self._A_s

        try:
            self._A_u, self._A_s, self._A_v = torch.svd(self.A, some=self.some)
        except RuntimeError:
            if self.A.is_cuda:
                _A_u, _A_s, _A_v = torch.svd(self.A.cpu(), some=self.some)
                self._A_s, self._A_v = _A_s.cuda(), _A_v.cuda()
            else:
                raise RuntimeError('could not perform svd on cpu')

        return self._A_s

    @property
    def A_v(self):
        """V matrix from SVD(A)"""
        if self._A_v is not None:
            return self._A_v
        self._A_u, self._A_s, self._A_v = torch.svd(self.A, some=self.some)
        return self._A_v

    @property
    def A_eigv(self):
        """Vector of eigenvalues of A"""
        if self._A_eigv is not None:
            return self._A_eigv
        eigv = torch.zeros(self.A.shape[1])
        # eigv = torch.zeros_like(self.A_s)


        if 'cuda' in str(self.A.device):
            eigv = eigv.cuda()

        eigv[:len(self.A_s)] = self.A_s ** 2
        self._A_eigv = eigv
        return self._A_eigv

    @property
    def L(self):
        """Lipschitz constant of gradient: largest eigenvalue of A.t() @ A"""
        return self.A_s[0] ** 2

    @property
    def mu(self):
        """Strong convexity constant: smallest eigenvalue of A.t() @ A"""
        return self.A_s[-1] ** 2

    def original_exact(self, g, x, z, beta, lambda_):
        """Original Exact minimization of augmented lagrangian along x variable"""
        b_ = (- lambda_ + beta * g(z)).view(-1) + self.A.t() @ self.b
        result = (1 / (self.A_eigv.view(-1, 1) + beta) * self.A_v.t()) @ b_
        #inv = self.A_v @ (1 / (self.A_eigv.view(-1, 1) + beta) * self.A_v.t())
        result = self.A_v @ result
        return result.view(x.shape).detach()

    def original_exact_bactched(self, g, x, z, beta, lambda_):
        """Exact minimization of augmented lagrangian along x variable"""
        B, _, H, W = x.shape
        part1 = (-lambda_ + beta * g(z)).view(B, -1).to(x.device)
        
        part2 = torch.matmul(self.A.t(),self.b)
        part2 = part2.reshape(B, -1)
        b_ = (part1 + part2).T
        
        b_ = b_.T.view(B, H, W)
        beta = beta.view(B, 1, 1)

        temp = torch.matmul(self.A_v.T, b_) / (self.A_eigv.view(1, 1, x.shape[2]) + beta)
        result = torch.matmul(self.A_v, temp).unsqueeze(1)

        return result.detach()
    
    def x_exact_classic(self, g, x, z, beta, lambda_):

        I = torch.eye(self.A.shape[0], device=x.device).expand(x.shape[0], -1, -1)
        rhs = (self.A.t() @ self.b.squeeze(1)) + beta.view(x.shape[0],1,1) * g(z).squeeze(1) - lambda_.squeeze(1)

        inv_mat = torch.inverse(self.A.t() @ self.A + beta.view(x.shape[0],1,1) * I)  # Like Thikonov regularization I think
        result = torch.bmm(inv_mat, rhs).unsqueeze(1)

        return result.detach()

    def grad(self, y):
        """Gradient"""
        grad = self.At_A @ y.view(-1) - self.A.t() @ self.b
        return grad.view(y.shape).detach()

    def __call__(self, y):
        """Function call"""
        y = y.squeeze(1)
        b = self.b.squeeze(1)
        return torch.norm(torch.matmul(self.A, y) - b) ** 2 / (2 * y.shape[0])
    



class L1distance:
    def __init__(self, b):
        self.b = b

    def __call__(self, x):
        return torch.norm(x - self.b, p=1)

    def prox(self, w):
        """argmin_x (x - b)_1 + 1/2 (w - x)_2^2"""
        return self.prox_l1(w - self.b)

    @staticmethod
    def prox_l1(x):
        result = torch.sign(x) * torch.max(torch.abs(x) - 1, torch.zeros_like(x))
        return result


class L2distance_sq:
    def __init__(self, b):
        self.b = b

    def grad(self, y):
        return y - self.b

    def __call__(self, x):
        return torch.norm(x - self.b) ** 2 / 2

    @property
    def L(self):
        """Lipschitz constant of gradient"""
        return 1.

    def x_exact(self, g, x, z, beta, lambda_):
        return ((self.b - lambda_ + beta * g(z)) / (1 + beta)).detach()

    def grad(self, y):
        return y - self.b


def inf_norm(x):
    return torch.max(torch.abs(x))

class ZeroFn:
    def __init__(self):
        pass

    def __call__(self, x):
        return 0

    def grad(self, y):
        return 0

    @property
    def L(self):
        return 0

    def x_exact(self, g, x, z, beta, lambda_):
        return (- lambda_ / beta + g(z)).detach()


class PhaseRetrieval:
    def __init__(self, A, b):
        self.A, self.b = A, b ** 2

    def __call__(self, y):
        """Function call"""
        return torch.norm((self.A @ y.view(-1)) ** 2 - self.b) ** 2 / 2

def measure_fn(A):
    """Make noisy measurements of signal x"""
    def me(x, std):
        b = A @ x.view(-1)
        return b + torch.zeros_like(b).normal_(std=std)

    return me
def tv_norm2D(x):
    """Total Variation norm for 2D signal x"""
    return (
        torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) +
        torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    )

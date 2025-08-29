import torch
import torch.nn.functional as F
from torch.optim import Optimizer


def pgm(classifier, n_iters, epsilon, p, min=0., max=1.):
    """
    Untargeted adversarial attack based on the projected gradient method

    Args:
        classifier (nn.Module): classifier network
        n_iters (int): number of iterations
        epsilon (float): maximum perturbation size
        p (float): p-norm measuring the size of the perturbation
        min (float): minimum possible value of the resulting tensor
        max (float): maximum possible value of the resulting tensor

    Returns:
        callable: function that computes adversarial examples
    """
    def perturb(x, y):
        """
        Args:
            x (torch.Tensor): batch of samples
            y (torch.Tensor): batch of true labels
        """
        if p == 1:
            raise NotImplementedError('p=1 not correctly implemented')
        x = x.clone().detach()
        x_0 = x.clone().detach()
        lr = epsilon / n_iters
        x.requires_grad = True
        optim = PGD([x], lr=lr, p=p)
        classifier.eval()

        for _ in classifier.parameters():
            _.requires_grad = False

        for i in range(n_iters):
            optim.zero_grad()
            loss = - F.cross_entropy(classifier(x), y, reduction='sum')
            loss.backward()
            optim.step()
            with torch.no_grad():
                x.data = (project(x - x_0, p=p, r=epsilon) + x_0).data
                torch.clamp_(x, min=min, max=max)

        return x, y

    return perturb


def project(x, p=2, r=1.):
    """
    projection onto a ball of certain radius

    Args:
        x (torch.Tensor): tensor to project
        p (float): p-norm value. Can be one of {1, 2, float('inf')}
        r (float): radius of the ball

    Return:
        torch.Tensor: projected value of x
    """
    n = x.shape[0]
    if p == 1:
        for i in range(x.shape[0]):
            x[i] = r * proj_l1(x[i] / r)
        return x
    elif p == 2:
        norm = torch.norm(x.view(n, -1), p=2, dim=1)
        x.view(n, -1).div_(norm[:, None])
        return x * r
    elif p == float('inf'):
        return torch.clamp(x, min=-r, max=r)
    else:
        raise ValueError('unsupported order: {}'.format(str(p)))


def proj_l1(v):
    r"""Projection onto L1 ball.
    Args:
        v (array)
    Returns:
        array: Result.
    References:
        J. Duchi, S. Shalev-Shwartz, and Y. Singer, "Efficient projections onto
        the l1-ball for learning in high dimensions" 2008.
    """
    v_ = v.view(-1)

    if torch.norm(v, 1) <= 1:
        return v
    else:
        numel = len(v_)
        s = torch.flip(torch.sort(torch.abs(v_))[0], dims=[0])
        vect = torch.arange(numel, device=s.device).float()
        st = (torch.cumsum(s, dim=0) - s) / (vect + 1)
        idx = torch.nonzero((s - st) > 0).max().long()
        result = soft_thresh(st[idx], v_)
        return result.reshape(v.shape)


def soft_thresh(lambda_, input):
    r"""Soft thresholding"""
    abs_input = torch.abs(input)
    sign = torch.sign(input)
    mag = abs_input - lambda_
    mag = (torch.abs(mag) + mag) / 2
    return mag * sign


class PGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
    """
    def __init__(self, params, lr, p=2):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        self.p = p
        defaults = dict(lr=lr, p_=p)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super(PGD, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                direction = self.dual_argmax(d_p, p=self.p)
                p.data.add_(-group['lr'], direction)

        return loss

    @staticmethod
    def dual_argmax(x, p):
        n = x.shape[0]
        if p == 1:
            sign, abs_ = torch.sign(x), torch.abs(x)
            max_val, _ = torch.max(abs_.view(n, -1), dim=1)
            mask = abs_.view(n, -1) == max_val[:, None]
            mask = torch.reshape(mask, x.shape)
            return sign * mask.float()
        elif p == 2:
            norm = torch.norm(x.view(n, -1), p=2, dim=1)
            x.view(n, -1).div_(norm[:, None])
            return x
        elif p == float('inf'):
            return torch.sign(x)
        else:
            raise ValueError('unsupported order: {}'.format(str(p)))


def _test_error(c):
    corr = correct(c)

    def test(x, y):
        return (1 - corr(x, y) / x.shape[0]) * 100

    return test


def correct(c):
    def test(x, y):
        _, pred = torch.max(c(x), dim=1)
        return torch.sum(pred == y).float()

    return test


def test_error(c, transform=None):
    corr = correct(c)

    def test(data):
        total = 0
        samples = 0

        if transform is not None:
            total_transformed = 0

        for x, y in data:
            total += corr(x, y)
            if transform is not None:
                x, y = transform(x, y)
                total_transformed += corr(x, y)

            samples += x.shape[0]

        error = (samples - total) / samples * 100

        if transform is not None:
            transform_error = (samples - total_transformed) / samples * 100
            return error.item(), transform_error.item()

        return error

    return test


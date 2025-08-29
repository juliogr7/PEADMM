import pdb
import inspect, sys
import torch

from collections import defaultdict


class Callback:
    def __init__(self, **variables):
        self.variables = variables
        self._n_iter = None
        self.values = defaultdict(list)

    @property
    def n_iter(self):
        return self._n_iter

    @n_iter.setter
    def n_iter(self, value):
        if type(value) is not int:
            raise ValueError('n_iter should be of type int')
        self._n_iter = value
        self.t = 0
        value = str(value)
        digits = str(len(value))
        self.iter_format = '[iter {0!s:>' + digits + '}/' + str(value) + ']'

    @staticmethod
    def _var_msg(name, value):
        if 'value' in name:
            return ' {0}: {1:>7.9e}'.format(name, value)
        return ' {0}: {1:>7.3e}'.format(name, value)

    def _iter_header(self):
        return self.iter_format.format(self.t + 1, self.n_iter)

    def clear(self):
        self.values = defaultdict(list)
        self._n_iter = None

    def __call__(self, namespace, name=''):
        if self.n_iter is None:
            self.n_iter = eval('n_iter', {}, namespace) + 1

        msg = name + ' ' + self._iter_header()

        for var, expr in self.variables.items():
            try:
                value = eval(expr, None, namespace)
            except Exception as e:
                value = None
            try:
                value = value.item()
            except AttributeError as e:
                pass

            self.values[var].append(value)

            if value is not None:
                msg += self._var_msg(var, value)

        end = '\r' if self.t < self.n_iter - 1 else '\n'
        print(msg, end=end)
        self.t += 1

    def __getitem__(self, item):
        return self.values[item]


def random_z(distribution, shape):
    """
    Generate a random noise tensor

    Args:
        distribution (str): name of the distribution, either 'uniform' or 'normal'
        shape (list): dimensions of the desired tensor

    Return:
        torch.tensor: random noise tensor
    """
    if distribution == 'uniform':
        z = torch.rand(shape) * 2 - 1
    elif distribution == 'normal':
        z = torch.randn(shape)
    else:
        raise ValueError('Unrecognized distribution: %s' % distribution)

    return z


class Logger(object):
    """
    Utility class for printing both to the terminal and to a log file

    Args:
        log_file (str): path to the log file
    """
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass


def line_search(gamma_0, theta=0.9):
    """
    Line search constructor

    Args:
       gamma_0 (float): initial step size
       theta (float): backtracking multiplier. theta \in (0, 1)
    """
    gamma = gamma_0

    def ls(fn, z):
        """
        Find the first iterate that satisfies the backtracking line search condition
        fn(x_1) <= fn(x_0) + <x_1 - x_0, fn'(x_0)> + 1 / (2 * gamma) ||x_1 - x_0||^2
        for decreasing stepsizes gamma = gamma_0 * theta^i, i >= 0.

        Args:
            fn (callable): function
            z: variable

        Return:
            torch.tensor: next iterate
        """
        nonlocal gamma

        try:
            z.grad.zero_()
        except AttributeError as e:
            pass

        fn_0 = fn(z)
        fn_0.backward()

        while True:
            z_1 = z.data - gamma * z.grad.data
            ip = torch.sum((z_1 - z) * z.grad)
            grad_ns = torch.norm(z.grad) ** 2
            fn_1 = fn(z_1)

            if fn_1 <= fn_0 + ip + gamma / 2 * grad_ns:
                gamma = gamma / theta
                break
            else:
                gamma = gamma * theta

        return z_1, torch.sqrt(grad_ns)

    return ls

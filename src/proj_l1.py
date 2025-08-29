import torch


def proj_l1(v):
    """Projection onto L1 ball.
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
        result = soft_thresh_torch(st[idx], v_)
        return result.reshape(v.shape)


def soft_thresh_torch(lambda_, input):
    abs_input = torch.abs(input)
    sign = torch.sign(input)
    mag = abs_input - lambda_
    mag = (torch.abs(mag) + mag) / 2
    return mag * sign

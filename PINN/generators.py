import torch

def generator_1dspatial(size, x_min, x_max, random=True):
    r"""Return a generator that generates 1D points range from x_min to x_max

    :param size:
        Number of points to generated when ``__next__`` is invoked.
    :type size: int
    :param x_min:
        Lower bound of x.
    :type x_min: float
    :param x_max:
        Upper bound of x.
    :type x_max: float
    :param random:
        - If set to False, then return equally spaced points range from ``x_min`` to ``x_max``.
        - If set to True then generate points randomly.

        Defaults to True.
    :type random: bool
    """
    seg_len = (x_max-x_min) / size
    linspace_lo = x_min + seg_len*0.5
    linspace_hi = x_max - seg_len*0.5
    center = torch.linspace(linspace_lo, linspace_hi, size)
    noise_lo = -seg_len*0.5
    while True:
        if random:
            noise = seg_len*torch.rand(size) + noise_lo
            yield center + noise
        else:
            yield center


def generator_2dspatial_segment(size, start, end, random=True):
    r"""Return a generator that generates 2D points in a line segment.

    :param size:
        Number of points to generated when `__next__` is invoked.
    :type size: int
    :param x_min:
        Lower bound of x.
    :type x_min: float
    :param x_max:
        Upper bound of x.
    :type x_max: float
    :param y_min:
        Lower bound of y.
    :type y_min: float
    :param y_max:
        Upper bound of y.
    :type y_max: float
    :param random:

        - If set to False, then return a grid where the points are equally spaced in the x and y dimension.
        - If set to True then generate points randomly.

        Defaults to True.
    :type random: bool
    """
    x1, y1 = start
    x2, y2 = end
    step = 1./size
    center = torch.linspace(0. + 0.5*step, 1. - 0.5*step, size)
    noise_lo = -step*0.5
    while True:
        if random:
            noise = step*torch.rand(size) + noise_lo
            center = center + noise
        yield x1 + (x2-x1)*center, y1 + (y2-y1)*center


def generator_2dspatial_rectangle(size, x_min, x_max, y_min, y_max, random=True):
    r"""Return a generator that generates 2D points in a rectangle.

    :param size:
        Number of points to generated when `__next__` is invoked.
    :type size: int
    :param start:
        The starting point of the line segment.
    :type start: tuple[float, float]
    :param end:
        The ending point of the line segment.
    :type end: tuple[float, float]
    :param random:
        - If set to False, then return eqally spaced points range from `start` to `end`.
        - If set to Rrue then generate points randomly.

        Defaults to True.
    :type random: bool
    """
    x_size, y_size = size
    x_generator = generator_1dspatial(x_size, x_min, x_max, random)
    y_generator = generator_1dspatial(y_size, y_min, y_max, random)
    while True:
        x = next(x_generator)
        y = next(y_generator)
        xy = torch.cartesian_prod(x, y)
        xx = torch.squeeze(xy[:, 0])
        yy = torch.squeeze(xy[:, 1])
        yield xx, yy
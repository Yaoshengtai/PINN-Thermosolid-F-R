
class BoundaryCondition:
    r"""A boundary condition. It is used to initialize ``temporal.Approximator``\s.

    :param form: The form of the boundary condition.

        - For a 1D time-dependent problem, if the boundary condition demands that :math:`B(u, x) = 0`,
          then ``form`` should be a function that maps :math:`u, x, t` to :math:`B(u, x)`.
        - For a 2D steady-state problem, if the boundary condition demands that :math:`B(u, x, y) = 0`,
          then ``form`` should be a function that maps :math:`u, x, y` to :math:`B(u, x, y)`.
        - For a 2D steady-state system, if the boundary condition demands that :math:`B(u_i, x, y) = 0`,
          then ``form`` should be a function that maps :math:`u_1, u_2, ..., u_n, x, y` to :math:`B(u_i, x, y)`.
        - For a 2D time-dependent problem, if the boundary condition demands that :math:`B(u, x, y) = 0`,
          then ``form`` should be a function that maps :math:`u, x, y, t` to :math:`B(u_i, x, y)`.

        Basically the function signature of ``form`` should be
        the same as the ``pde`` function of the given ``temporal.Approximator``.
    :type form: callable
    :param points_generator:
        A generator that generates points on the boundary.
        It can be a `temporal.generator_1dspatial`, `temporal.generator_2dspatial_segment`,
        or a generator written by user.
    :type points_genrator: generator
    """
    def __init__(self, form, points_generator,weight):
        self.form = form
        self.points_generator = points_generator
        self.weight=weight
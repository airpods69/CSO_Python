from scipy.io import loadmat
import numpy as np
import numpy.matlib as matlib

class benchmarks:

    fhd = None
    f_bias = None

    def __init__(self, initial_flag: int) -> None:
        self.initial_flag = initial_flag


    def sphere_shift_func(self, x):
        """
        Returns a column vector containing the sum of each row from x
        """

        ps, D = np.shape(x)

        if self.initial_flag == 0:
            self.o_sphere_shift_func = loadmat('benchmarks/sphere_shift_func_data.mat')['o']
            if self.o_sphere_shift_func.size >= D:
                self.o_sphere_shift_func = self.o_sphere_shift_func[0:D]
            else:
                self.o_sphere_shift_func = -100 + 200 * np.random.rand(1, D)

            self.initial_flag = 1

        x = x - matlib.repmat(self.o_sphere_shift_func, ps, 1)
        return np.sum(np.power(x, 2), 1)

    def schwefel_func(self, x):
        ps, D = np.shape(x)

        if self.initial_flag == 0:
            self.o_schwefel_func = loadmat('benchmarks/schwefel_shift_func_data.mat')['o']


            if self.o_schwefel_func.size >= D:
                self.o_schwefel_func = self.o_schwefel_func[0:D]
            else:
                self.o_schwefel_func = -100 + 200 * np.random.rand(1, D)

            self.initial_flag = 1

        x = x - matlib.repmat(self.o_schwefel_func, ps, 1)
        return np.abs(x).max(axis = 1)

    def rosenbrock_shift_func(self, x):
        ps, D = np.shape(x)

        if self.initial_flag == 0:
            self.o_rosenbrock_shift_func = loadmat('benchmarks/rosenbrock_shift_func_data.mat')['o']

            if self.o_rosenbrock_shift_func.size >= D:
                self.o_rosenbrock_shift_func = self.o_rosenbrock_shift_func[0:D]
            else:
                self.o_rosenbrock_shift_func = -90 + 180 * np.random.rand(1,D)

            self.initial_flag = 1

        x = x - matlib.repmat(self.o_rosenbrock_shift_func, ps, 1) + 1
        f = np.sum(100 * np.power(np.power(x[:, 0:D], 2) - x[:, 0:D], 2) + np.power((x[:, 0:D ] - 1),2), 1) # Chances of error here due to the complexity of statement

        return f

    def rastrigin_shift_func(self, x):
        ps, D = np.shape(x)

        if self.initial_flag == 0:
            self.o_rastrigin_shift_func = loadmat('benchmarks/rastrigin_shift_func_data.mat')['o']

            if self.o_rastrigin_shift_func.size >= D:
                self.o_rastrigin_shift_func = self.o_rastrigin_shift_func[0:D]
            else:
                self.o_rastrigin_shift_func = -5 + 10 * np.random.rand(1,D)

            self.initial_flag = 1

        x = x - matlib.repmat(self.o_rastrigin_shift_func, ps, 1)
        f = np.sum(np.power(x, 2) - 10 * np.cos(2 * np.pi * x) + 10, 1 )
        return f







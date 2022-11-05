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
            self.o_sphere_shift_func = loadmat('benchmarks/sphere_shift_func_data.mat')

            if len(self.o_sphere_shift_func) >= D:
                self.o_sphere_shift_func = self.o_sphere_shift_func[0:D]
            else:
                self.o_sphere_shift_func = -100 + 200 * np.random.rand(1, D)

        self.initial_flag = 1

        x = x - matlib.repmat(self.o_sphere_shift_func, ps, 1)
        # print(x)
        # print(x.shape)
        # print(np.sum(x ** 2, 1).shape)
        # exit()
        return np.sum(x ** 2, 1)

    def schwefel_func(self, x):
        ps, D = np.shape(x)

        if self.initial_flag == 0:
            self.o_schwefel_func = loadmat('benchmarks/schwefel_shift_func_data.mat')


            if len(self.o_schwefel_func) >= D:
                self.o_schwefel_func = self.o_schwefel_func[0:D]
            else:
                self.o_schwefel_func = -100 + 200 * np.random.rand(1, D)

        self.initial_flag = 1

        x = x - matlib.repmat(self.o_schwefel_func, ps, 1)
        return np.abs(x).max(axis = 1)




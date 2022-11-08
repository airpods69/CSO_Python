import numpy as np

def init(funcid, d):
    """
    initializing parameters for cso
    """

    def lu_init(funcid: int, n: int):
        """
        switch statement for lu
        """
        if funcid in list(range(1,3 + 1)):
            lu = np.array([-100 * np.ones((n)), 100 * np.ones((n))])

        elif funcid == 4:
            lu = np.array([-5 * np.ones((n)), 5 * np.ones((n))])

        elif funcid == 5:
            lu = np.array([-600 * np.ones((n)), 600 * np.ones((n))])

        elif funcid == 6:
            lu = np.array([-32 * np.ones((n)), 32 * np.ones((n))])

        else:
            lu = np.array([-100 * np.ones((n)), 100 * np.ones((n))])

        return lu

    if funcid in [1, 4, 5, 6]:
        if d >= 2000:
            phi = 0.2
        elif d >= 1000:
            phi = 0.15
        elif d >= 500:
            phi = 0.1
        else:
            phi = 0

    else:
        if d >= 2000:
            phi = 0.2
        elif d >= 1000:
            phi = 0.1
        elif d >= 500:
            phi = 0.05
        else:
            phi = 0

    if d >= 5000:
        m = 1500
    elif d >= 2000:
        m = 1000
    elif d >= 1000:
        m = 500
    elif d >= 100:
        m = 100
    else:
        m = 50

    lu = lu_init(funcid, d)

    return lu, phi, m


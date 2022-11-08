import numpy as np

def main_loop(v:np.ndarray, p, phi: int, d: int, m: int, fitness):

    rlist = np.random.permutation(m).T
    rpairs = np.vstack([rlist[0:int(np.ceil(m/2))], rlist[m//2: m]]).T

    center = np.ones((int(np.ceil(m/2)), 1)) @ np.mean(p,axis=0).reshape(1, -1)

    mask = fitness[rpairs[:,0]] > fitness[rpairs[:,1]]


    losers = (np.multiply(mask, rpairs[:, 0]) + np.multiply(np.logical_not(mask), rpairs[:, 1])).T
    winners = (np.multiply(np.logical_not(mask), rpairs[:, 0]) + np.multiply(mask, rpairs[:, 1])).T

    randco1 = np.random.rand(int(np.ceil(m/2)), d);
    randco2 = np.random.rand(int(np.ceil(m/2)), d);
    randco3 = np.random.rand(int(np.ceil(m/2)), d);
    # print(np.multiply(randco1, v[losers][:]))
    # print((np.multiply(randco2, p[winners][:]) - p[losers][:] + phi * np.multiply(randco3, (center - p[losers][:]))))
    # exit()

    x = np.multiply(randco1, v[losers])
    y = np.multiply(randco2, p[winners] - p[losers]) + phi * np.multiply(randco3, (center - p[losers]))

    v[losers] = x + y

    p[losers] = p[losers] + v[losers]

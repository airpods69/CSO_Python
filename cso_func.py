import multiprocessing
from typing import List
from initialize import init
from benchmarks.benchmark_func import benchmarks

import numpy as np
import numpy.matlib as matlib


d = 1000
maxfe = d * 500

def run(funcid: int, x_dict):
    global d, maxfe

    initial_flag = 0
    benchmark = benchmarks(initial_flag)
    lu, phi, m = init(funcid, d)

    XRRmin = matlib.repmat(lu[0], m, 1)
    XRRmax = matlib.repmat(lu[1], m, 1)

    p = XRRmin + np.multiply(np.subtract(XRRmax, XRRmin), np.random.rand(m,d))

    if funcid == 1:
        name = "sphere_shift_func"
        fitness = benchmark.sphere_shift_func(x = p)

    elif funcid == 2:
        name ="schwefel_func"
        fitness = benchmark.schwefel_func(x = p)

    elif funcid == 3:
        name = "rosenbrock_shift_func"
        fitness = benchmark.rosenbrock_shift_func(x = p)

    elif funcid == 4:
        name = "rastrigin_shift_func"
        fitness = benchmark.rastrigin_shift_func(x = p)

    else:
        name = "sphere_shift_func"
        fitness = benchmark.sphere_shift_func(x = p)

    # add the rest here

    v = np.zeros((m, d))
    bestever = 1e200

    FES = m
    gen = 0

    while FES < maxfe:

        rlist = np.random.permutation(m).T
        rpairs = np.vstack([rlist[0:int(np.ceil(m/2))], rlist[m//2: m]]).T

        center = np.ones((int(np.ceil(m/2)), 1)) @ np.mean(p,axis=0).reshape(1, -1)

        mask = fitness[rpairs[:,0]] > fitness[rpairs[:,1]]

        losers = (np.multiply(mask, rpairs[:, 0]) + np.multiply(np.logical_not(mask), rpairs[:, 1])).T
        winners = (np.multiply(np.logical_not(mask), rpairs[:, 0]) + np.multiply(mask, rpairs[:, 1])).T

        randco1 = np.random.rand(int(np.ceil(m/2)), d);
        randco2 = np.random.rand(int(np.ceil(m/2)), d);
        randco3 = np.random.rand(int(np.ceil(m/2)), d);

        x = np.multiply(randco1, v[losers])
        y = np.multiply(randco2, p[winners] - p[losers]) + phi * np.multiply(randco3, (center - p[losers]))

        v[losers] = x + y

        p[losers] = p[losers] + v[losers]

        for i in range(0, int(np.ceil(m/2))):
            x = np.maximum(p[losers[i], :], lu[0, :])
            y = np.minimum(p[losers[i], :], lu[1, :])
            p[losers[i], :] = x
            p[losers[i], :] = y

        if funcid == 1:
            fitness[losers] = benchmark.sphere_shift_func(x = p[losers])

        elif funcid == 2:
            fitness[losers] = benchmark.schwefel_func(x = p[losers])

        elif funcid == 3:
            fitness[losers] = benchmark.rosenbrock_shift_func(x = p[losers])

        elif funcid == 4:
            fitness[losers] = benchmark.rastrigin_shift_func(x = p[losers])

        else:
            fitness[losers] = benchmark.sphere_shift_func(x = p[losers])

        bestever = np.minimum(bestever, np.min(fitness))
        print(f"Best Fitness: {bestever:.10}")

        FES += int(np.ceil(m/2))
        gen += 1

    x_dict[name] = bestever


def main():

    from multiprocessing import Process

    processes = []

    man = multiprocessing.Manager()
    x_dict = man.dict()

    for funcid in range(1, 5): # cause 1 to 4 jaana hai for now
        processes.append(Process(target = run, args=(funcid, x_dict)))
        processes[funcid - 1].start()

    for p in processes:
        p.join()

    print(x_dict)

if __name__ == "__main__":
    main()


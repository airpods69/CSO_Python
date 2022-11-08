import numpy as np
import numpy.matlib as matlib

from benchmarks.benchmark_func import benchmarks

np.random.seed(1337)

d = 1000
maxfe = d * 5000

runnum = 1

results = np.zeros((2, runnum))


m = None
lu = [[]]

for funcid in range (1, 2 + 1):

    if funcid == 2:
        break

    n = d
    initial_flag = 0
    benchmark = benchmarks(initial_flag=initial_flag)

    if funcid == 1:
        lu = np.array([-100 * np.ones((n)), 100 * np.ones((n))])

    elif funcid == 2:
        lu = np.array([-100 * np.ones((n)), 100 * np.ones((n))])


    if funcid == 1 and True: # Instead of true we will be adding more conditions of funcid
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
    elif d >= 1000:
        m = 100

    for run in range(0, runnum):

        XRRmin = matlib.repmat(lu[0], m, 1)
        XRRmax = matlib.repmat(lu[1], m, 1)

        p = XRRmin + np.multiply(np.subtract(XRRmax, XRRmin), np.random.rand(m,d))

        if funcid == 1:
            fitness = benchmark.sphere_shift_func(x = p)

        elif funcid == 2:
            fitness = benchmark.schwefel_func(x = p)

        v = np.zeros((m, d))

        bestever = 1e200

        FES = m
        gen = 0


        while FES < maxfe:

            rlist = np.random.permutation(m).T
            rpairs = np.vstack([rlist[0:int(np.ceil(m/2))], rlist[m//2: m]]).T



            _1 = np.ones((int(np.ceil(m/2)), 1))
            _2 = np.mean(p,axis=0).reshape(1, -1)

            center =  _1 @ _2

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


            for i in range(0, int(np.ceil(m/2))):
                x = np.maximum(p[losers[i], :], lu[0, :])
                y = np.minimum(p[losers[i], :], lu[1, :])
                p[losers[i], :] = x
                p[losers[i], :] = y

            if funcid == 1:
                fitness[losers] = benchmark.sphere_shift_func(x = p[losers, :])

            elif funcid == 2:
                fitness[losers] = benchmark.schwefel_func(x = p[losers])

            bestever = np.minimum(bestever, np.min(fitness))

            print("Best Fitness: {}".format(bestever))
            FES += int(np.ceil(m/2))

            gen += 1

        results[funcid][runnum - 1] = bestever
        print("Run no {} Done".format(run))


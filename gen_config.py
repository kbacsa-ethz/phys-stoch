import os
import numpy as np
import random
import argparse


np.random.seed(42)
random.seed(42)


def main(cfg):

    # random masses between 0 and 2
    if cfg.lk:
        masses = np.around(cfg.ndof * np.random.rand(cfg.ndof), 2)
    else:
        masses = np.array(cfg.ndof * [1.])

    m = np.diag(masses)

    #springs = np.sort(np.around(2 * np.random.rand(cfg.ndof + 1), 2))[::-1]
    springs = np.array([3., .5, 0.])

    spring_diagonal = springs[:-1] + springs[1:]
    spring_lu = -springs[1:-1]
    k = np.diag(spring_diagonal) + np.diag(spring_lu, 1) + np.diag(spring_lu, -1)

    if cfg.type == 'dissipative':
        capacitors = np.around(0.5 * np.random.rand(cfg.ndof + 1), 3)
        capacitors_diagonal = capacitors[:-1] + capacitors[1:]
        capacitors_lu = -capacitors[1:-1]
        c = np.diag(capacitors_diagonal) + np.diag(capacitors_lu, 1) + np.diag(capacitors_lu, -1)
    else:
        c = np.zeros_like(m)

    m_string = ",".join(list(map(str, masses.flatten())))
    k_string = ",".join(list(map(str, k.flatten())))
    c_string = ",".join(list(map(str, c.flatten())))

    observables = list(map(str, range(3 * cfg.ndof)))
    if cfg.select == "random":
        observables = random.sample(observables, int(len(observables) * 0.75))
    else:
        observables = observables[:cfg.ndof] + observables[2*cfg.ndof:]

    with open(os.path.join(args.rp, "config", "{}springmass_{}.ini".format(cfg.ndof, cfg.type)), "w") as filep:
        filep.write("[System]\n")
        filep.write("Name = {}springmass_{}\n".format(cfg.ndof, cfg.type))
        filep.write("M = " + m_string + "\n")
        filep.write("C = " + c_string + "\n")
        filep.write("K = " + k_string + "\n")
        filep.write("\n")
        filep.write("[Forces]\n")
        filep.write("Type = free\n")
        filep.write("Amplitude = 1.5\n")
        filep.write("Frequency = 0.05\n")
        filep.write("Shift = 0.25\n")
        filep.write("Inputs = 0,1\n")
        filep.write("\n")

        filep.write("[Simulation]\n")
        filep.write("Seed = 42\n")
        filep.write("Iterations = 500\n")
        filep.write("Observations = " + ",".join(observables)+ "\n")
        filep.write("Noise = " + ",".join([str(cfg.noise)] * 2 * cfg.ndof) + "\n")
        filep.write("Absolute = 1.0e-8\n")
        filep.write("Relative = 1.0e-6\n")
        filep.write("Delta = 0.1\n")
        filep.write("Time = 60.\n")
    return 0


if __name__ == '__main__':
    np.random.seed(42)

    # parse config
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-rp', type=str, default='.')
    parser.add_argument('-type', type=str, default='free')
    parser.add_argument('-select', type=str, default='notrandom')
    parser.add_argument('-ndof', type=int, default=2)
    parser.add_argument('-noise', type=float, default=0.05)
    parser.add_argument('-lk', action='store_true')
    args = parser.parse_args()

    main(args)

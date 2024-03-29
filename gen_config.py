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

    springs = np.around(2 * np.random.rand(cfg.ndof + 1), 2)
    spring_diagonal = springs[:-1] + springs[1:]
    spring_lu = -springs[1:-1]
    k = np.diag(spring_diagonal) + np.diag(spring_lu, 1) + np.diag(spring_lu, -1)

    if cfg.type == 'dissipative':
        capacitors = np.around(0.7 * np.random.rand(cfg.ndof + 1), 3)
        capacitors_diagonal = capacitors[:-1] + capacitors[1:]
        capacitors_lu = -capacitors[1:-1]
        c = np.diag(capacitors_diagonal) + np.diag(capacitors_lu, 1) + np.diag(capacitors_lu, -1)
    else:
        c = np.zeros_like(m)

    m_string = ",".join(list(map(str, masses.flatten())))
    k_string = ",".join(list(map(str, k.flatten())))
    c_string = ",".join(list(map(str, c.flatten())))

    observables = cfg.select

    with open(os.path.join(args.rp, "config", "{}_{}_{}_{}_{}_{}.ini".format(cfg.ndof, cfg.sys, cfg.dynamics, cfg.ext, cfg.type, cfg.select)), "w") as filep:
        filep.write("[System]\n")
        filep.write("Name = {}_springmass_{}_{}_{}\n".format(cfg.ndof, cfg.dynamics, cfg.ext, cfg.type))
        filep.write("M = " + m_string + "\n")
        filep.write("C = " + c_string + "\n")
        filep.write("K = " + k_string + "\n")
        filep.write("Dynamics: " + cfg.dynamics + "\n")

        filep.write("\n")
        filep.write("[Forces]\n")
        filep.write("Type = {}\n".format(cfg.ext))
        filep.write("Amplitude = {}\n".format(cfg.amplitude))
        filep.write("Frequency = {}\n".format(cfg.freq))
        filep.write("Shift = {}\n".format(cfg.shift))
        filep.write("Inputs = 0\n")  # keep default for now (input on first degree of freedom)
        filep.write("\n")

        filep.write("[Simulation]\n")
        filep.write("Seed = 42\n")
        filep.write("Lower_x = {}\n".format(cfg.l_x))
        filep.write("Upper_x = {}\n".format(cfg.u_x))
        filep.write("Lower_xdot = {}\n".format(cfg.l_y))
        filep.write("Upper_xdot = {}\n".format(cfg.u_y))
        filep.write("Iterations = {}\n".format(cfg.n_iter))
        filep.write("Observations = " + observables + "\n")
        # higher orders have lower noise, assumes several of different orders

        obs_idx = [int(x) for x in observables.split(',')]

        noise_list = []
        for idx in obs_idx:
            if idx < cfg.ndof:
                noise_list += [str(cfg.noise/2)]
            elif idx < 2 * cfg.ndof:
                noise_list += [str(cfg.noise/1.5)]
            else:
                noise_list += [str(cfg.noise)]

        filep.write("Noise = " + ",".join(noise_list) + "\n")
        filep.write("Absolute = 1.0e-8\n")
        filep.write("Relative = 1.0e-6\n")
        filep.write("Delta = 0.01\n")
        filep.write("Time = 60.\n")
    return 0


if __name__ == '__main__':
    np.random.seed(42)

    # parse config
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-rp', type=str, default='.')
    parser.add_argument('-type', type=str, default='free')
    parser.add_argument('-n_iter', type=int, default=50)
    parser.add_argument('-dynamics', type=str, default='duffing')
    parser.add_argument('-sys', type=str, default='springmass')
    parser.add_argument('-ext', type=str, default='free')
    parser.add_argument('-amplitude', type=float, default=15.0)
    parser.add_argument('-freq', type=float, default=0.5)
    parser.add_argument('-shift', type=int, default=100)
    parser.add_argument('-select', type=str, default='0,1,4,5')
    parser.add_argument('-ndof', type=int, default=2)
    parser.add_argument('-l_x', type=float, default=-2)
    parser.add_argument('-l_y', type=float, default=-2)
    parser.add_argument('-u_x', type=float, default=2)
    parser.add_argument('-u_y', type=float, default=2)
    parser.add_argument('-noise', type=float, default=20)
    parser.add_argument('-lk', action='store_true')
    args = parser.parse_args()

    main(args)

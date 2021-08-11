import numpy as np


if __name__ == '__main__':
    np.random.seed(42)

    ndof = 20

    # random masses between 0 and 2
    masses = np.round(2*np.random.rand(ndof), 2)
    springs = np.round(2*np.random.rand(ndof+1))

    spring_diagonal = springs[:-1] + springs[1:]
    spring_lu = -springs[1:-1]

    k = np.diag(spring_diagonal) + np.diag(spring_lu, 1) + np.diag(spring_lu, -1)
    m = np.diag(masses)
    c = np.zeros_like(m)

    m_string = ",".join(list(map(str, masses.flatten())))
    k_string = ",".join(list(map(str, k.flatten())))
    c_string = ",".join(list(map(str, c.flatten())))

    with open("config/%dspringmass_free.ini" % ndof, "w") as filep:
        filep.write("[System]\n")
        filep.write("Name = %dspringmass\n" % ndof)
        filep.write("M = " + m_string + "\n")
        filep.write("C = " + c_string + "\n")
        filep.write("K = " + k_string + "\n")
        filep.write("\n")
        filep.write("[Forces]\n")
        filep.write("Type = free\n")
        filep.write("Amplitude = 1.5\n")
        filep.write("Frequency = 0.05\n")
        filep.write("Shift = 0.25\n")
        filep.write("Inputs = 0, 1\n")
        filep.write("\n")

        filep.write("[Simulation]\n")
        filep.write("Seed = 42\n")
        filep.write("Iterations = 50\n")
        filep.write("Observations = " + ",".join(list(map(str, range(2*ndof)))) + "\n")
        filep.write("Noise = " + ",".join(["0.05"] * 2 * ndof) + "\n")
        filep.write("Absolute = 1.0e-8\n")
        filep.write("Relative = 1.0e-6\n")
        filep.write("Delta = 0.1\n")
        filep.write("Time = 60.\n")











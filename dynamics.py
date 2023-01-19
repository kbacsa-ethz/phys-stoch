import numpy as np

from sympy import symbols
from sympy.physics import mechanics

from sympy import Dummy, lambdify


def linear(w, t, p):
    m, c, k, ext = p

    n_dof = m.shape[0]
    A = np.concatenate(
        [
            np.concatenate([np.zeros([n_dof, n_dof]), np.eye(n_dof)], axis=1),  # link velocities
            np.concatenate([-np.linalg.solve(m, k), -np.linalg.solve(m, c)], axis=1),  # movement equations
        ], axis=0)

    f = A @ w + np.concatenate([np.zeros(n_dof), np.linalg.solve(m, ext(t))])
    return f


def duffing(w, t, p):
    m, c, k, k2, ext = p

    n_dof = m.shape[0]
    A = np.concatenate(
        [
            np.concatenate([np.zeros([n_dof, n_dof]), np.eye(n_dof)], axis=1),  # link velocities
            np.concatenate([-np.linalg.solve(m, k), -np.linalg.solve(m, c)], axis=1),  # movement equations
        ], axis=0)

    B = np.concatenate(
        [
            np.concatenate([np.zeros([n_dof, n_dof]), np.zeros([n_dof, n_dof])], axis=1),
            np.concatenate([-np.linalg.solve(m, k2), np.zeros([n_dof, n_dof])], axis=1),  # duffing term
        ], axis=0)

    nonlinear = np.zeros_like(w)
    nonlinear[:n_dof] = w[:n_dof] ** 3
    nonlinear = B @ nonlinear
    force = np.concatenate([np.zeros(n_dof), np.linalg.solve(m, ext(t))])
    f = A @ w + nonlinear + force
    return f


def halfcar(w, t, p):
    m, c, c2, k, k2, k3, ext = p

    n_dof = m.shape[0]
    A = np.concatenate(
        [
            np.concatenate([np.zeros([n_dof, n_dof]), np.eye(n_dof)], axis=1),  # link velocities
            np.concatenate([-np.linalg.solve(m, k), -np.linalg.solve(m, c)], axis=1),  # movement equations
        ], axis=0)

    B = np.concatenate(
        [
            np.concatenate([np.zeros([n_dof, n_dof]), np.zeros([n_dof, n_dof])], axis=1),
            np.concatenate([-np.linalg.solve(m, k2), -np.linalg.solve(m, c2)], axis=1),  # quadratic term
        ], axis=0)

    C = np.concatenate(
        [
            np.concatenate([np.zeros([n_dof, n_dof]), np.zeros([n_dof, n_dof])], axis=1),
            np.concatenate([-np.linalg.solve(m, k3), np.zeros([n_dof, n_dof])], axis=1),  # cubic term
        ], axis=0)

    quadratic = w ** 2
    quadratic = B @ quadratic
    cubic = w ** 3
    cubic = C @ cubic
    force = np.concatenate([np.zeros(n_dof), np.linalg.solve(m, ext(t))])
    f = A @ w + quadratic + cubic + force
    return f


def pendulum(n, lengths=None, masses=1):
    """Integrate a multi-pendulum with `n` sections"""

    # code from https://jakevdp.github.io/blog/2017/03/08/triple-pendulum-chaos/

    # -------------------------------------------------
    # Step 1: construct the pendulum model

    # Generalized coordinates and velocities
    # (in this case, angular positions & velocities of each mass)
    q = mechanics.dynamicsymbols('q:{0}'.format(n))
    u = mechanics.dynamicsymbols('u:{0}'.format(n))

    # mass and length
    m = symbols('m:{0}'.format(n))
    l = symbols('l:{0}'.format(n))

    # gravity and time symbols
    g, t = symbols('g,t')

    # --------------------------------------------------
    # Step 2: build the model using Kane's Method

    # Create pivot point reference frame
    A = mechanics.ReferenceFrame('A')
    P = mechanics.Point('P')
    P.set_vel(A, 0)

    # lists to hold particles, forces, and kinetic ODEs
    # for each pendulum in the chain
    particles = []
    forces = []
    kinetic_odes = []

    for i in range(n):
        # Create a reference frame following the i^th mass
        Ai = A.orientnew('A' + str(i), 'Axis', [q[i], A.z])
        Ai.set_ang_vel(A, u[i] * A.z)

        # Create a point in this reference frame
        Pi = P.locatenew('P' + str(i), l[i] * Ai.x)
        Pi.v2pt_theory(P, A, Ai)

        # Create a new particle of mass m[i] at this point
        Pai = mechanics.Particle('Pa' + str(i), Pi, m[i])
        particles.append(Pai)

        # Set forces & compute kinematic ODE
        forces.append((Pi, m[i] * g * A.x))
        kinetic_odes.append(q[i].diff(t) - u[i])
        P = Pi

    # Generate equations of motion
    KM = mechanics.KanesMethod(A, q_ind=q, u_ind=u,
                               kd_eqs=kinetic_odes)
    fr, fr_star = KM.kanes_equations(particles, forces)

    # -----------------------------------------------------
    # Step 3: numerically evaluate equations and integrate

    # lengths and masses
    if lengths is None:
        lengths = 16 * np.ones(n) / n
    lengths = np.broadcast_to(lengths, n)
    masses = np.broadcast_to(masses, n)

    # Fixed parameters: gravitational constant, lengths, and masses
    parameters = [g] + list(l) + list(m)
    parameter_vals = [9.81] + list(lengths) + list(masses)

    # define symbols for unknown parameters
    unknowns = [Dummy() for i in q + u]
    unknown_dict = dict(zip(q + u, unknowns))
    kds = KM.kindiffdict()

    # substitute unknown symbols for qdot terms
    mm_sym = KM.mass_matrix_full.subs(kds).subs(unknown_dict)
    fo_sym = KM.forcing_full.subs(kds).subs(unknown_dict)

    # create functions for numerical calculation
    mm_func = lambdify(unknowns + parameters, mm_sym)
    fo_func = lambdify(unknowns + parameters, fo_sym)

    # function which computes the derivatives of parameters
    def gradient(w, t, p):
        vals = np.concatenate((w, p))
        sol = np.linalg.solve(mm_func(*vals), fo_func(*vals))
        return np.array(sol).T[0]

    # ODE integration
    return gradient, parameter_vals

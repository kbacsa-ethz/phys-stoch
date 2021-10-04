import numpy as np


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
            np.concatenate([-np.linalg.solve(m, k2), np.zeros([n_dof, n_dof])], axis=1),   # duffing term
        ], axis=0)

    nonlinear = np.zeros_like(w)
    nonlinear[:n_dof] = w[:n_dof]**3
    nonlinear = B @ nonlinear
    force = np.concatenate([np.zeros(n_dof), np.linalg.solve(m, ext(t))])
    f = A @ w + nonlinear + force
    return f

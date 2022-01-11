# code inspired by https://github.com/nghiaho12/rigid_transform_3D

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal


# Input: expects 2xN matrix of points
# Returns R,t
# R = 2x2 rotation matrix
# t = 2x1 column vector


def rigid_transform_2D(A, B):
    #assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 2:
        raise Exception(f"matrix A is not 2xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 2:
        raise Exception(f"matrix B is not 2xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 2x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    # if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[1, :] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B
    return R @ A + t


# Code from Eleni's matlab script
def least_square_transform(traj, dt, name):
    fs = 1 / dt
    shift = 0
    avg = traj.mean()
    traj = traj - avg
    learned = traj[:, :2]
    ground = traj[shift:, 2:4]
    N = min(learned.shape[0], ground.shape[0])
    f1, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    f1.suptitle('{} welch spectrums'.format(name), fontsize=16)
    freq, welch = scipy.signal.welch(learned.T, fs=fs, window='hamming')
    ax1.semilogy(freq, welch[0], label='learned 1')
    ax1.semilogy(freq, welch[1], label='learned 2')
    ax1.set_xlabel('Frequency [Hz]')
    ax1.set_ylabel('Power/freq [dB/Hz]')
    ax1.set_title('Welch power spectral density estimate')
    ax1.legend()
    freq, welch = scipy.signal.welch(ground.T, fs=fs, window='hamming')
    ax2.semilogy(freq, welch[0], label='true 1')
    ax2.semilogy(freq, welch[1], label='true 2')
    ax2.set_xlabel('Frequency [Hz]')
    ax2.set_ylabel('Power/freq [dB/Hz]')
    ax2.set_title('Welch power spectral density estimate')
    ax2.legend()

    inter = range(N)
    learned = learned[inter]
    ground = ground[inter]
    T, _, _, _ = np.linalg.lstsq(learned, ground, rcond=None)
    learned_rot = learned @ T
    f2, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    f2.suptitle('{} phase space'.format(name), fontsize=16)
    ax1.plot(learned_rot[inter, 0], learned_rot[inter, 1])
    ax1.set_title('Learned rotated')
    ax2.plot(ground[inter, 0], ground[inter, 1])
    ax2.set_title('Ground truth')

    return f1, f2

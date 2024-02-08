import numpy as np
import scipy.linalg


def QMult(q1, q2):
    r = (q1[:, 0] * q2[:, 0] - np.sum(q1[:, 1:4] * q2[:, 1:4], axis=1)).reshape(q1.shape[0], 1)
    v = q1[:, 0].reshape(q1.shape[0], 1) * q2[:, 1:4] + q2[:, 0].reshape(q2.shape[0], 1) * q1[:, 1:4] + np.cross(
        q1[:, 1:4], q2[:, 1:4])
    return np.hstack((r, v))


def QConj(q):
    return np.hstack((q[:, 0].reshape(q.shape[0], 1), -q[:, 1:4].reshape(q.shape[0], 3)))


def QInv(q):
    qabs = np.linalg.norm(q, axis=1).reshape(q.shape[0], 1)
    return QConj(q) / (qabs ** 2)


def QRotate(q1, v):
    if (q1.shape[0] == 1):  # broadcasting a single quaternion
        q1r = np.tile(q1, (v.shape[0], 1))
    else:
        q1r = q1
    return QMult(QMult(q1r, np.hstack((np.zeros((v.shape[0], 1)), v))), QInv(q1r))[:, 1:4]


def QExp(q):
    absv = np.linalg.norm(q[:, 1:4], axis=1).reshape(q.shape[0], 1)
    if np.abs(absv) < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    else:
        eq = np.exp(q[:, 0]) * np.hstack((np.cos(absv).reshape(absv.shape[0], 1), np.sin(absv) * q[:, 1:4] / absv))
        # print("eq norm", np.linalg.norm(eq))
        return eq


def Q2RotMatrix(q):
    corrv = q[:, 1:4].transpose() @ q[:, 1:4]
    R = np.array([[1.0 - 2.0 * (corrv[1, 1] + corrv[2, 2]), 2.0 * (corrv[0, 1] - (q[:, 0] * q[:, 3])[0]), 2 * (corrv[0, 2] + (q[:, 0] * q[:, 2])[0])],
                 [2.0 * (corrv[1, 0] + (q[:, 0] * q[:, 3])[0]), 1.0 - 2.0 * (corrv[2, 2] + corrv[0, 0]), 2.0 * (corrv[1, 2] - (q[:, 0] * q[:, 1])[0])],
                 [2.0 * (corrv[2, 0] - (q[:, 0] * q[:, 2])[0]), 2.0 * (corrv[2, 1] + (q[:, 0] * q[:, 1])[0]),1.0 - 2.0 * (corrv[0, 0] + corrv[1, 1])]])
    # R[0, 0] = 1.0 - 2.0 * (corrv[1, 1] + corrv[2, 2])
    # R[1, 1] = 1.0 - 2.0 * (corrv[2, 2] + corrv[0, 0])
    # R[2, 2] = 1.0 - 2.0 * (corrv[0, 0] + corrv[1, 1])
    # R[0, 1] = 2.0 * (corrv[0, 1] - q[:, 0] * q[:, 3])
    # R[1, 0] = 2.0 * (corrv[1, 0] + q[:, 0] * q[:, 3])
    # R[0, 2] = 2.0 * (corrv[0, 2] + q[:, 0] * q[:, 2])
    # R[2, 0] = 2.0 * (corrv[2, 0] - q[:, 0] * q[:, 2])
    # R[1, 2] = 2.0 * (corrv[1, 2] - q[:, 0] * q[:, 1])
     #R[2, 1] = 2.0 * (corrv[2, 1] + q[:, 0] * q[:, 1])
    return R

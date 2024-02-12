import os
import sys
import pickle
import numpy as np

sys.path.append(os.path.join('..', 'code'))
from RBSFunctions import Mesh, resolve_position_constraint, resolve_velocity_constraint
from RBSLoadVis import load_mesh_file
from Quaternion import *


if __name__ == '__main__':

    data_path = os.path.join('..', 'data')  # Replace with the path to your folder
    CRCoeff = 0.5
    numSamples = 100

    collisionCounter = numSamples
    pickle_file_path = data_path + os.path.sep + 'constraint-resolution.data'
    with open(pickle_file_path, 'rb') as pickle_file:
        loaded_data = pickle.load(pickle_file)

    correctedPositionsUpper = [np.zeros_like(loaded_data['randomLinVelocities']), np.zeros_like(loaded_data['randomLinVelocities'])]
    correctedPositionsLower = [np.zeros_like(loaded_data['randomLinVelocities']), np.zeros_like(loaded_data['randomLinVelocities'])]
    correctedLinVelocities = [np.zeros_like(loaded_data['randomLinVelocities']), np.zeros_like(loaded_data['randomLinVelocities'])]
    correctedAngVelocities = [np.zeros_like(loaded_data['randomLinVelocities']), np.zeros_like(loaded_data['randomLinVelocities'])]
    positionWasValidUpper = [False for _ in range(numSamples)]
    positionWasValidLower = [False for _ in range(numSamples)]
    velocityWasValid = [False for _ in range(numSamples)]

    for i in range(loaded_data['randomLinVelocities'].shape[0]):
        dummyMesh1 = Mesh("dummy mesh1", loaded_data['origVertices1'], loaded_data['faces1'], loaded_data['tets1'],
                          1.0,
                          np.zeros([1, 3]),
                          np.array([1.0, 0.0, 0.0, 0.0]),
                          False)
        dummyMesh3 = Mesh("dummy mesh2", loaded_data['origVertices2'], loaded_data['faces2'], loaded_data['tets2'],
                          1.0,
                          np.array(loaded_data['randomPositions'][i, :], copy=True).reshape(1, 3),
                          np.array(loaded_data['randomOrientations'][i, :], copy=True).reshape(1, 4), False)

        refValue = loaded_data['refValues'][i]
        dummyMesh3.linVelocity = np.array(loaded_data['randomLinVelocities'][i, :], copy=True).reshape(1, 3)
        dummyMesh3.angVelocity = np.array(loaded_data['randomAngVelocities'][i, :], copy=True).reshape(1, 3)
        refValueLower = 0.8 * refValue
        refValueUpper = 1.2 * refValue

        origConstPoint1 = dummyMesh1.origVertices[0, :].reshape(1, 3)
        origConstPoint2 = dummyMesh3.origVertices[0, :].reshape(1, 3)
        currConstPoint1 = QRotate(dummyMesh1.orientation, origConstPoint1) + dummyMesh1.position
        currConstPoint2 = QRotate(dummyMesh3.orientation, origConstPoint1) + dummyMesh3.position

        currPositions = (dummyMesh1.position, dummyMesh3.position)
        currConstPoints = (currConstPoint1, currConstPoint2)
        currlinVelocities = (dummyMesh1.linVelocity, dummyMesh3.linVelocity)
        currAngVelocities = (dummyMesh1.angVelocity, dummyMesh3.angVelocity)

        invInertiaTensors = (dummyMesh1.get_rotated_inv_IT(), dummyMesh3.get_rotated_inv_IT())
        invMasses = (dummyMesh1.invMass, dummyMesh3.invMass)

        # Testing actual function
        positionWasValidUpper[i], correctedPositionsUpperLocal = resolve_position_constraint(currPositions,
                                                                                             currConstPoints,
                                                                                             refValueUpper,
                                                                                             invMasses, True, 1e-6)
        positionWasValidLower[i], correctedPositionsLowerLocal = resolve_position_constraint(currPositions,
                                                                                             currConstPoints,
                                                                                             refValueLower,
                                                                                             invMasses, False, 1e-6)

        velocityWasValid[i], correctedLinVelocitiesLocal, correctedAngVelocitiesLocal = resolve_velocity_constraint(currPositions, currConstPoints, currlinVelocities, currAngVelocities, invMasses,
                                    invInertiaTensors, 10e-6)

        correctedPositionsUpper[0][i,:] = correctedPositionsUpperLocal[0]
        correctedPositionsUpper[1][i, :] = correctedPositionsUpperLocal[1]
        correctedPositionsLower[0][i,:] = correctedPositionsLowerLocal[0]
        correctedPositionsLower[1][i, :] = correctedPositionsLowerLocal[1]
        correctedLinVelocities[0][i,:] = correctedLinVelocitiesLocal[0]
        correctedLinVelocities[1][i, :] = correctedLinVelocitiesLocal[1]
        correctedAngVelocities[0][i,:] = correctedAngVelocitiesLocal[0]
        correctedAngVelocities[1][i, :] = correctedAngVelocitiesLocal[1]

    print("COM error for upper-bound constraint Mesh1: ", np.max(np.abs(loaded_data['correctedPositionsUpper'][0] - correctedPositionsUpper[0])))
    print("COM error for upper-bound constraint Mesh2: ",
          np.max(np.abs(loaded_data['correctedPositionsUpper'][1] - correctedPositionsUpper[1])))
    print("COM error for lower-bound constraint Mesh1: ",
          np.max(np.abs(loaded_data['correctedPositionsLower'][0] - correctedPositionsLower[0])))
    print("COM error for lower-bound constraint Mesh2: ",
          np.max(np.abs(loaded_data['correctedPositionsLower'][1] - correctedPositionsLower[1])))

    print("Linear velocity error for Mesh1: ",
          np.max(np.abs(loaded_data['correctedLinVelocities'][0] - correctedLinVelocities[0])))
    print("Linear velocity error for Mesh2: ",
          np.max(np.abs(loaded_data['correctedLinVelocities'][1] - correctedLinVelocities[1])))
    print("Linear angular error for Mesh1: ",
          np.max(np.abs(loaded_data['correctedAngVelocities'][0] - correctedAngVelocities[0])))
    print("Linear angular error for Mesh2: ",
          np.max(np.abs(loaded_data['correctedAngVelocities'][1] - correctedAngVelocities[1])))
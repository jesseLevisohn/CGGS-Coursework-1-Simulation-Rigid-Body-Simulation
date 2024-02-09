import os
import sys
import pickle
import numpy as np

sys.path.append(os.path.join('..', 'code'))
from RBSFunctions import Mesh, resolve_collision
from RBSLoadVis import load_mesh_file
from Quaternion import *
from InitializeCCD import initialize_libccd


def random_vectors_in_sphere(radius, num_samples):
    # Generate random angles
    theta = np.random.uniform(0, np.pi, num_samples)
    phi = np.random.uniform(0, 2 * np.pi, num_samples)

    # Generate random radii
    r = radius * np.power(np.random.rand(num_samples),
                          1 / 3)  # cubic root to ensure uniform distribution within the sphere

    # Convert spherical coordinates to Cartesian coordinates
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return np.column_stack((x, y, z))


if __name__ == '__main__':

    initialize_libccd()
    data_path = os.path.join('..', 'data')  # Replace with the path to your folder
    CRCoeff = 0.5
    numSamples = 100

    collisionCounter = numSamples
    pickle_file_path = data_path + os.path.sep + 'collision-resolution.data'
    with open(pickle_file_path, 'rb') as pickle_file:
        loaded_data = pickle.load(pickle_file)

    resultPositions = [np.zeros_like(loaded_data['randomPositions']), np.zeros_like(loaded_data['randomPositions'])]
    resultOrientations = [np.zeros_like(loaded_data['randomOrientations']),
                          np.zeros_like(loaded_data['randomOrientations'])]
    resultLinVelocities = [np.zeros_like(loaded_data['randomLinVelocities']),
                           np.zeros_like(loaded_data['randomLinVelocities'])]
    resultAngVelocities = [np.zeros_like(loaded_data['randomAngVelocities']),
                           np.zeros_like(loaded_data['randomAngVelocities'])]

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
        # dummyMesh2.position = np.array(loaded_data['randomPositions'][i, :], copy=True).reshape(1, 3)
        # dummyMesh2.orientation = np.array(loaded_data['randomOrientations'][i, :], copy=True).reshape(1, 4)
        dummyMesh3.linVelocity = np.array(loaded_data['randomLinVelocities'][i, :], copy=True).reshape(1, 3)
        dummyMesh3.angVelocity = np.array(loaded_data['randomAngVelocities'][i, :], copy=True).reshape(1, 3)
        #isCollision, depth, intNormal, intPosition = dummyMesh1.detect_collision(dummyMesh2)
        if loaded_data['isCollision'][i]:
            resolve_collision(dummyMesh1, dummyMesh3, loaded_data['depth'][i], loaded_data['intNormal'][i,:], loaded_data['intPosition'][i,:], CRCoeff)
        else:  # nothing should have changed
            collisionCounter -= 1
        # else nothing should have changed

        resultLinVelocities[0][i] = np.array(dummyMesh1.linVelocity, copy=True).reshape(1, 3)
        resultLinVelocities[1][i] = np.array(dummyMesh3.linVelocity, copy=True).reshape(1, 3)
        resultAngVelocities[0][i] = np.array(dummyMesh1.angVelocity, copy=True).reshape(1, 3)
        resultAngVelocities[1][i] = np.array(dummyMesh3.angVelocity, copy=True).reshape(1, 3)
        resultPositions[0][i] = np.array(dummyMesh1.position, copy=True).reshape(1, 3)
        resultPositions[1][i] = np.array(dummyMesh3.position, copy=True).reshape(1, 3)
        resultOrientations[0][i] = np.array(dummyMesh1.orientation, copy=True).reshape(1, 4)
        resultOrientations[1][i] = np.array(dummyMesh3.orientation, copy=True).reshape(1, 4)

    print("Linear velocity error mesh1: ",
          np.max(np.abs(loaded_data['resultLinVelocities'][0] - resultLinVelocities[0])))
    print("Linear velocity error mesh2: ",
          np.max(np.abs(loaded_data['resultLinVelocities'][1] - resultLinVelocities[1])))
    print("Angular velocity error mesh1: ",
          np.max(np.abs(loaded_data['resultAngVelocities'][0] - resultAngVelocities[0])))
    print("Angular velocity error mesh2: ",
          np.max(np.abs(loaded_data['resultAngVelocities'][1] - resultAngVelocities[1])))
    print("COM error mesh1: ", np.max(np.abs(loaded_data['resultPositions'][0] - resultPositions[0])))
    print("COM error mesh2: ", np.max(np.abs(loaded_data['resultPositions'][1] - resultPositions[1])))
    print("Orientation error mesh1: ", np.max(np.abs(loaded_data['resultOrientations'][0] - resultOrientations[0])))
    print("Orientation error mesh2: ", np.max(np.abs(loaded_data['resultOrientations'][1] - resultOrientations[1])))
    print("Collision counter: ", collisionCounter)

import os
import sys
import pickle
import numpy as np

sys.path.append(os.path.join('..', 'code'))
from RBSFunctions import Mesh, resolve_collision
from RBSLoadVis import load_mesh_file
from Quaternion import *

if __name__ == '__main__':

    data_path = os.path.join('..', 'data')  # Replace with the path to your folder

    mesh_file_path = os.path.join(data_path, 'cylinder.mesh')  # it doesn't matter
    origVertices, faces, tets = load_mesh_file(mesh_file_path)

    pickle_file_path = data_path + os.path.sep + 'integration.data'
    with open(pickle_file_path, 'rb') as pickle_file:
        loaded_data = pickle.load(pickle_file)

    dummyMesh2 = Mesh("dummy mesh", origVertices, faces, tets, 1.0, loaded_data['randomPositions'][0, :],
                      loaded_data['randomOrientations'][0, :],
                      False)
    resultLinVelocities = np.zeros_like(loaded_data['randomLinVelocities'])
    resultAngVelocities = np.zeros_like(loaded_data['randomAngVelocities'])
    resultPositions = np.zeros_like(loaded_data['randomPositions'])
    resultOrientations = np.zeros_like(loaded_data['randomOrientations'])
    for i in range(0,1): #range(loaded_data['randomLinVelocities'].shape[0]):
        dummyMesh2.position = np.array(loaded_data['randomPositions'][i, :], copy=True).reshape(1,3)
        dummyMesh2.orientation = np.array(loaded_data['randomOrientations'][i, :], copy=True).reshape(1,4)
        dummyMesh2.linVelocity = np.array(loaded_data['randomLinVelocities'][i, :], copy=True).reshape(1,3)
        dummyMesh2.angVelocity = np.array(loaded_data['randomAngVelocities'][i, :], copy=True).reshape(1,3)
        dummyMesh2.integrate_timestep(0.1)
        resultLinVelocities[i, :] = np.array(dummyMesh2.linVelocity, copy=True).reshape(1,3)
        resultAngVelocities[i, :] = np.array(dummyMesh2.angVelocity, copy=True).reshape(1,3)
        resultPositions[i, :] = np.array(dummyMesh2.position, copy=True).reshape(1,3)
        resultOrientations[i, :] = np.array(dummyMesh2.orientation, copy=True).reshape(1,4)

    print("Linear velocity error: ", np.max(np.abs(loaded_data['resultLinVelocities'] - resultLinVelocities)))
    print("Angular velocity error: ", np.max(np.abs(loaded_data['resultAngVelocities'] - resultAngVelocities)))
    print("COM error: ", np.max(np.abs(loaded_data['resultPositions'] - resultPositions)))
    print("Orientation error: ", np.max(np.abs(loaded_data['resultOrientations'] - resultOrientations)))

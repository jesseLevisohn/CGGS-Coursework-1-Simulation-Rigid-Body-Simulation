import os
import numpy as np
from RBSFunctions import Mesh
import cppyy
import scipy.linalg

def update_visual_constraints(ps_mesh, vertices, constEdges):
    constVertices = vertices[constEdges]

    constVertices = constVertices.reshape(2 * constVertices.shape[0], 3)
    curveNetIndices = np.arange(0, constVertices.shape[0])
    curveNetIndices = curveNetIndices.reshape(int(len(curveNetIndices) / 2), 2)
    return constVertices, curveNetIndices

    return constVertices, constEdges


def load_off_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Parse the vertices and faces from the OFF file
    num_vertices, num_faces, _ = map(int, lines[1].split())

    vertices = np.array([list(map(float, line.split())) for line in lines[2:2 + num_vertices]])
    faces = np.array([list(map(int, line.split()))[1:] for line in lines[2 + num_vertices:]])

    return vertices, faces


def load_constraint_file(file_path, meshes):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Parse the vertices and faces from the OFF file
    numConstraints = int(lines[0].strip())
    constraints = []
    for lineIndex in range(numConstraints):
        parts = lines[1 + lineIndex].split()
        fullConstraint = list(map(int, parts[0:4])) + list(map(float, parts[4:6]))

        # computing original distance and putting it as the ref value for the constraints
        origDistance = np.linalg.norm(
            meshes[fullConstraint[0]].currVertices[fullConstraint[1]] - meshes[fullConstraint[2]].currVertices[
                fullConstraint[3]])
        lbConstraints = fullConstraint[0:4]+list((fullConstraint[4] * origDistance,False))
        ubConstraints = fullConstraint[0:4]+list((fullConstraint[5] * origDistance, True))
        constraints.append(lbConstraints)
        constraints.append(ubConstraints)
    return constraints


def load_scene_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Parse the vertices and faces from the OFF file
    numMeshes = int(lines[0].strip())
    meshes = []
    for lineIndex in range(numMeshes):
        elements = lines[1 + lineIndex].split()
        directory, filename = os.path.split(file_path)

        mesh_file_path = os.path.join(directory, elements[0])
        origVertices, faces, tets = load_mesh_file(mesh_file_path)
        meshes.append(
            Mesh(elements[0], origVertices, faces, tets, np.array(elements[1], dtype=float),
                 np.array(elements[3:6], dtype=float).reshape(1, 3),
                 np.array(elements[6:10], dtype=float), bool(int(elements[2]))))

    return meshes


def flatten_meshes(meshes, flattenFaces=True, constraints=[]):
    allVertices = np.empty((0, 3))
    allFaces = np.empty((0, 3))
    allDensities = np.empty((0))
    allConstEdges = np.zeros([len(constraints), 2], dtype=int)
    faceOffsets = np.zeros(len(meshes))
    faceOffsets[0] = 0
    currFaceOffset = 0
    currIndex = 0
    for mesh in meshes:
        allVertices = np.concatenate((allVertices, mesh.currVertices))
        if flattenFaces:
            allFaces = np.concatenate((allFaces, mesh.faces + currFaceOffset))
            allDensities = np.concatenate((allDensities, np.full(mesh.currVertices.shape[0], mesh.density)))
            faceOffsets[currIndex] = currFaceOffset
            currFaceOffset += mesh.currVertices.shape[0]
            currIndex += 1

    # Translating constraints
    if (flattenFaces):
        for constraintIndex in range(len(constraints)):
            allConstEdges[constraintIndex, :] = [
                faceOffsets[constraints[constraintIndex][0]] + constraints[constraintIndex][1],
                faceOffsets[constraints[constraintIndex][2]] + constraints[constraintIndex][3]]

    return allVertices, allFaces, allDensities, allConstEdges


def load_mesh_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Parse the vertices and faces from the OFF file
    num_vertices = int(lines[3].strip())

    vertices = np.array([list(map(float, line.split())) for line in lines[4:4 + num_vertices]])[:, 0:3]

    facesStart = 4 + num_vertices  # location of "triangles" reserved word
    num_faces = int(lines[facesStart + 1].strip())

    faces = np.array([list(map(int, line.split())) for line in lines[facesStart + 2:facesStart + 2 + num_faces]])[:,
            0:3]

    tetsStart = facesStart + 2 + num_faces
    num_tets = int(lines[tetsStart + 1].strip())
    tets = np.array([list(map(int, line.split())) for line in lines[tetsStart + 2:tetsStart + 2 + num_tets]])[:, 0:4]

    # resetting to zero-indexing
    if np.min(faces) == 1:
        faces -= 1

    if np.min(tets) == 1:
        tets -= 1

    return vertices, faces, tets


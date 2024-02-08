import os
import numpy as np
import cppyy
import scipy.linalg
from Quaternion import *



def resolve_velocity_constraint(currPositions, currConstPoints, currLinVelocities, currAngVelocities, invMasses,
                                invInertiaTensors, tolerance):

    #Complete code
    velocityWasValid = True #stub
    correctedLinVelocities = currLinVelocities #stub
    correctedAngVelocities = currAngVelocities #stub
    return velocityWasValid, correctedLinVelocities, correctedAngVelocities


def resolve_position_constraint(currPositions, currConstPoints, refValue, invMasses, upper, tolerance):

    ####Complete code
    positionWasValid = True #stub
    correctedPositions = currPositions #stub
    return positionWasValid, correctedPositions


def run_timestep(meshes, groundMesh, timeStep, CRCoeff, enforceConstraints=True, constraints=np.empty((0, 4)),
                 maxIterations=1000, tolerance=0.0):
    for mesh in meshes:
        mesh.integrate_timestep(timeStep)

    for mesh1 in meshes:
        for mesh2 in meshes[meshes.index(mesh1) + 1:]:  # TODO: use range
            isCollision, depth, intNormal, intPosition = mesh1.detect_collision(mesh2)
            if isCollision:
                resolve_collision(mesh1, mesh2, depth, intNormal, intPosition, CRCoeff)

    # resolving the collision with the ground by generating a pseudo-mesh for the ground
    for mesh in meshes:
        minyIndex = int(np.argmin(mesh.currVertices, axis=0)[1])
        #    #linear resolution
        if mesh.currVertices[minyIndex, 1] <= 0.0:
            resolve_collision(mesh, groundMesh, mesh.currVertices[minyIndex, 1],
                              np.array([0.0, 1.0, 0.0]).reshape(1, 3), mesh.currVertices[minyIndex, :],
                              CRCoeff)

    if (enforceConstraints):
        currIteration = 0
        zeroStreak = 0
        currConstIndex = 0
        while zeroStreak < len(constraints) and currIteration * len(constraints) < maxIterations:
            currConstraint = constraints[currConstIndex]
            origConstPoint1 = meshes[currConstraint[0]].origVertices[currConstraint[1], :].reshape(1, 3)
            origConstPoint2 = meshes[currConstraint[2]].origVertices[currConstraint[3], :].reshape(1, 3)
            refValue = currConstraint[4]

            currConstPoint1 = QRotate(meshes[currConstraint[0]].orientation, origConstPoint1) + meshes[
                currConstraint[0]].position
            currConstPoint2 = QRotate(meshes[currConstraint[2]].orientation, origConstPoint2) + meshes[
                currConstraint[2]].position

            currPositions = (meshes[currConstraint[0]].position, meshes[currConstraint[2]].position)
            currConstPoints = (currConstPoint1, currConstPoint2)
            currLinVelocities = (meshes[currConstraint[0]].linVelocity, meshes[currConstraint[2]].linVelocity)
            currAngVelocities = (meshes[currConstraint[0]].angVelocity, meshes[currConstraint[2]].angVelocity)

            invInertiaTensors = (
                meshes[currConstraint[0]].get_rotated_inv_IT(), meshes[currConstraint[2]].get_rotated_inv_IT())

            invMasses = (meshes[currConstraint[0]].invMass, meshes[currConstraint[2]].invMass)
            Upper = currConstraint[5]

            positionWasValid, correctedPositions = resolve_position_constraint(currPositions, currConstPoints, refValue,
                                                                               invMasses, Upper, tolerance)

            if not positionWasValid:
                velocityWasValid, correctedLinVelocities, correctedAngVelocities = resolve_velocity_constraint(
                currPositions, currConstPoints, currLinVelocities, currAngVelocities, invMasses, invInertiaTensors, tolerance)
            else:
                velocityWasValid = True  #position constraint was not violated

            if velocityWasValid and positionWasValid:
                zeroStreak += 1
            else:
                # only update the COM and angular velocity, don't both updating all currV because it might change again during this loop!
                zeroStreak = 0
                if not velocityWasValid:
                    meshes[currConstraint[0]].linVelocity = correctedLinVelocities[0]
                    meshes[currConstraint[2]].comVelocity = correctedLinVelocities[1]

                    meshes[currConstraint[0]].angVelocity = correctedAngVelocities[0]
                    meshes[currConstraint[2]].angVelocity = correctedAngVelocities[1]

                if not positionWasValid:
                    meshes[currConstraint[0]].position = correctedPositions[0]
                    meshes[currConstraint[2]].position = correctedPositions[1]

            currIteration += 1
            currConstIndex = (currConstIndex + 1) % (len(constraints))

            # if currIteration * len(constraints) >= maxIterations:
            #    print("Constraint resolution reached maxIterations without resolving!")

        # Updating actual currvertices
        for mesh in meshes:
            mesh.currVertices = QRotate(mesh.orientation, mesh.origVertices) + mesh.position


# Resolving the collision between two meshes by both correcting positions, and setting impulses to correct velocities
def resolve_collision(m1, m2, depth, contactNormal, penPoint, CRCoeff):


    #####Compelte code here for resolving interpenetration

    m1.currVertices = QRotate(m1.orientation, m1.origVertices) + m1.position
    m2.currVertices = QRotate(m2.orientation, m2.origVertices) + m2.position

    #####Complete code here for resolving velocities


class Mesh:

    def __init__(self, name, origVertices, faces, tets, density, position, orientation, isFixed):
        # Instance attributes
        self.name = name
        self.origVertices = origVertices
        self.density = density
        self.faces = faces
        self.tets = tets
        self.position = position
        self.orientation = orientation.reshape(1, 4)
        self.orientation /= np.linalg.norm(self.orientation)
        self.isFixed = isFixed
        self.position = position
        self.linVelocity = np.zeros([1, 3])
        self.angVelocity = np.zeros([1, 3])
        self.currImpulses = []

        if self.origVertices.shape[0] > 0:
            self.init_static_properties()
            # initial orientation
            self.currVertices = QRotate(self.orientation, self.origVertices) + position
            self.invMass = 1.0 / self.mass

        if self.origVertices.shape[0] == 0 or self.isFixed:
            self.invIT = np.zeros([3, 3])
            self.invMass = 0.0

    def init_static_properties(self):
        # obtaining the natural COM of the original vertices an putting it to (0,0,0) so it's easier later
        e01 = self.origVertices[self.tets[:, 1], :] - self.origVertices[self.tets[:, 0], :]
        e02 = self.origVertices[self.tets[:, 2], :] - self.origVertices[self.tets[:, 0], :]
        e03 = self.origVertices[self.tets[:, 3], :] - self.origVertices[self.tets[:, 0], :]
        tetCentroids = (self.origVertices[self.tets[:, 0], :] + self.origVertices[self.tets[:, 1],
                                                                :] + self.origVertices[self.tets[:, 2],
                                                                     :] + self.origVertices[self.tets[:, 3], :]) / 4.0
        tetVolumes = np.abs(np.sum(e01 * np.cross(e02, e03), axis=1)) / 6.0
        totalVolume = np.sum(tetVolumes)

        naturalCOM = np.sum(tetCentroids * tetVolumes[:, np.newaxis], axis=0) / totalVolume
        self.origVertices -= naturalCOM

        self.mass = self.density * totalVolume

        # computing inertia tensor of each tet around the now COM of (0,0,0)
        sumMat = np.identity(4) + 1.0
        xvecs = self.origVertices[self.tets, 0]
        yvecs = self.origVertices[self.tets, 1]
        zvecs = self.origVertices[self.tets, 2]
        # Ordering: 00,11,22, 12 10 20
        IValues = np.zeros((self.tets.shape[0], 6))
        for i in range(len(self.tets)):
            IValues[i, 0] = (yvecs[i, :] @ sumMat @ np.transpose(yvecs[i, :]) + zvecs[i, :] @ sumMat @ np.transpose(
                zvecs[i, :])) / 120.0
            IValues[i, 1] = (xvecs[i, :] @ sumMat @ np.transpose(xvecs[i, :]) + zvecs[i, :] @ sumMat @ np.transpose(
                zvecs[i, :])) / 120.0
            IValues[i, 2] = (xvecs[i, :] @ sumMat @ np.transpose(xvecs[i, :]) + yvecs[i, :] @ sumMat @ np.transpose(
                yvecs[i, :])) / 120.0
            IValues[i, 3] = -(yvecs[i, :] @ sumMat @ np.transpose(zvecs[i, :])) / 120.0
            IValues[i, 4] = -(xvecs[i, :] @ sumMat @ np.transpose(yvecs[i, :])) / 120.0
            IValues[i, 5] = -(zvecs[i, :] @ sumMat @ np.transpose(xvecs[i, :])) / 120.0

        totalI = self.density * np.sum(tetVolumes[:, np.newaxis] * IValues, axis=0)
        self.IT = np.array([[totalI[0], totalI[4], totalI[5]],
                            [totalI[4], totalI[1], totalI[3]],
                            [totalI[5], totalI[3], totalI[2]]])

        # testing
        # r = np.linalg.norm(self.origVertices, axis=1)
        # self.IT = 0.4*np.eye(3,3)*self.mass*(np.mean(r)**2)
        self.invIT = np.linalg.inv(self.IT)

    def get_rotated_inv_IT(self):
        ####Complete code
        return np.identity((3)) #stub

    # Return true if all dimensions are overlapping
    def bounding_box_collision(self, m):
        selfBBox = [np.min(self.currVertices, axis=0), np.max(self.currVertices, axis=0)]
        mBBox = [np.min(m.currVertices, axis=0), np.max(m.currVertices, axis=0)]
        return np.all(selfBBox[1] >= mBBox[0]) and np.all(mBBox[1] >= selfBBox[0])

    def detect_collision(self, m):

        if self.isFixed and m.isFixed:
            return False, [], [], []

        if not self.bounding_box_collision(m):
            return False, [], [], []

        n1 = np.array([float(self.origVertices.shape[0])])
        n2 = np.array([float(m.origVertices.shape[0])])

        # Initialization to be able to pass pointers
        depth = np.array([0.0])
        intNormal = np.array([0.0, 0.0, 0.0])
        intPosition = np.array([0.0, 0.0, 0.0])

        # Call the C++ function with the Eigen matrices and ctypes for double by reference
        isCollision = cppyy.gbl.isCollide(n1, self.currVertices, self.position, n2, m.currVertices, m.position, depth,
                                          intNormal, intPosition)
        return isCollision, depth, intNormal, intPosition

    def integrate_timestep(self, timeStep):
        ####complete code
        self.currVertices = QRotate(self.orientation, self.origVertices) + self.position

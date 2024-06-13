import sys
import numpy as np
import pickle
from pathlib import Path
import argparse
import sys
import cv2
import os
from glob import glob
import random
import shutil
import time
import uuid
import json
from datasets import load_dataset
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from common import *
from convert_obj_to_glb import convert_file


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    print(f'Seed: {seed}')


class Shape(object):
    def __init__(self):
        self.points = []
        self.uvs = []
        self.faces = []
        self.facesUV = []
        self.matNames = []
        self.matStartId = []

    def genShape(self):
        self.points = np.reshape([], (-1,3)).astype(float)
        self.uvs = np.reshape([], (-1,2)).astype(float)
        self.faces = np.reshape([], (-1,3)).astype(int)
        self.facesUV = np.reshape([], (-1,3)).astype(int)
        self.matNames = []
        self.matStartId = []

    def permuteMatIds(self, ratio = 0.25):
        if len(self.matStartId) == 0:
            print("no mats")
            return
        newIds = [self.matStartId[0]]

        for i in range(1, len(self.matStartId)):
            neg = self.matStartId[i] - self.matStartId[i-1]
            negCount = -int(ratio * neg)

            if i != len(self.matStartId) - 1:
                pos = self.matStartId[i+1] - self.matStartId[i]
            else:
                pos = len(self.faces) - self.matStartId[i]

            posCount = int(ratio * pos)

            offset = np.random.permutation(posCount - negCount)[0] + negCount

            newIds.append(self.matStartId[i] + offset)
        self.matStartId = newIds

    def computeNormals(self):
        vec0 = self.points[self.faces[:,1]-1] - self.points[self.faces[:,0]-1]
        vec1 = self.points[self.faces[:,2]-1] - self.points[self.faces[:,1]-1]
        areaNormals = np.cross(vec0, vec1)
        self.normals = self.points.copy()
        vertFNs = np.zeros(len(self.points), int)
        vertFMaps = np.zeros((len(self.points), 200), int)
        for iF, face in enumerate(self.faces):
            for id in face:
                vertFMaps[id-1, vertFNs[id-1]] = iF
                vertFNs[id-1] += 1

        for i in range(len(self.points)):
            faceNormals = areaNormals[vertFMaps[i,:vertFNs[i]]]
            normal = np.average(faceNormals, axis=0)
            self.normals[i] = normalize(normal).reshape(-1)
        return self.normals

    def loadSimpleObj(self, filePath):

        self.points = []
        self.uvs = []
        self.faces = []
        self.facesUV = []
        self.matNames = []
        self.matStartId = []

        with open(filePath, "r") as f:
            #write v
            curFid = 0
            while True:
                lineStr = f.readline()
                if lineStr == "":
                    break
                if lineStr[:2] == "v ":
                    point = [float(val) for val in lineStr[2:-1].split(" ")]
                    self.points.append(point)
                if lineStr[:2] == "vt":
                    point = [float(val) for val in lineStr[3:-1].split(" ")]
                    self.uvs.append(point)
                if lineStr[:len("usemtl")] == "usemtl":
                    self.matStartId.append(curFid)
                    self.matNames.append(lineStr[len("usemtl "):-1])
                if lineStr[:2] == "f ":
                    curFid += 1
                    self.faces.append([])
                    self.facesUV.append([])
                    for oneTerm in lineStr[2:-1].split(" "):
                        iduvids = oneTerm.split("/")
                        self.faces[-1].append(int(iduvids[0]))
                        self.facesUV[-1].append(int(iduvids[1]))

        self.points = np.reshape(self.points, (-1, 3)).astype(float)
        self.uvs = np.reshape(self.uvs, (-1, 2)).astype(float)
        self.faces = np.reshape(self.faces, (-1, 3)).astype(int)
        self.facesUV = np.reshape(self.facesUV, (-1, 3)).astype(int)
        self.matStartId = np.reshape(self.matStartId, -1)

    
    def genObj(self, filePath, bMat = False, bComputeNormal=False, bScaleMesh=False, bMaxDimRange=[0.3, 0.5]):
        """ write a .obj file containing points, normals, uvs, and faces"""
        
        if bScaleMesh:
            # scale max dim to the range of bMaxDimRange
            max_dim = bMaxDimRange[0] + np.random.rand(1)[0] * (bMaxDimRange[1] - bMaxDimRange[0])
            minP = np.min(self.points, axis=0)
            maxP = np.max(self.points, axis=0)
            print('Scale to [%f, %f]\n' % (-max_dim, max_dim))
            scale = max_dim / np.max(maxP)
            self.points = scale * self.points

        if len(self.faces) == 0:
            print( "no mesh")
            return False
        if bComputeNormal:
            self.computeNormals()

        with open(filePath, "w") as f:
            #write v
            for point in self.points:
                f.write("v %f %f %f\n"%(point[0], point[1], point[2]))

            if bComputeNormal:
                for point in self.normals:
                    f.write("vn %f %f %f\n"%(point[0], point[1], point[2]))
                    # f.write("vn %f %f %f\n" % (0, 0, 1))
            # write uv
            for uv in self.uvs:
                f.write("vt %f %f\n" % (uv[0], uv[1]))
            #write face
            # f.write("usemtl mat_%d\n"%matId)
            if not bMat:
                for i,face in enumerate(self.faces):
                    f.write("f %d/%d %d/%d %d/%d\n" %
                            (face[0], self.facesUV[i][0], face[1], self.facesUV[i][1], face[2], self.facesUV[i][2]))
            else:
                for im in range(len(self.matStartId)):
                    f.write("usemtl %s\n"%self.matNames[im])
                    if im == len(self.matStartId) - 1:
                        endId = len(self.faces)
                    else:
                        endId = self.matStartId[im+1]
                    if not bComputeNormal:
                        for i in range(self.matStartId[im], endId):
                            f.write("f %d/%d %d/%d %d/%d\n" %
                                    (self.faces[i][0], self.facesUV[i][0], self.faces[i][1], self.facesUV[i][1], self.faces[i][2], self.facesUV[i][2]))
                    else:
                        for i in range(self.matStartId[im], endId):
                            f.write("f %d/%d/%d %d/%d/%d %d/%d/%d\n" %
                                    (self.faces[i][0], self.facesUV[i][0], self.faces[i][0],
                                     self.faces[i][1], self.facesUV[i][1], self.faces[i][1],
                                     self.faces[i][2], self.facesUV[i][2], self.faces[i][2]))


        mtl_path = filePath.replace(".obj", ".mtl")

        material_ids = []
        with open(mtl_path, "w") as f:
            for im in range(len(self.matStartId)):
                f.write("newmtl %s \n" % (self.matNames[im]))
                # TODO switch to sampling different materials instead of fixed one
                # material_id = random.choice(range(len(all_mat_paths)))
                # mat_path = all_mat_paths[material_id]  # random select a material
                material_ids.append(0)  # TODO switch to material name
                f.write(f"map_Kd {im:02d}_basecolor.png \n")
                f.write(f"map_Ks {im:02d}_metallic.png \n")
                f.write(f"map_Pr {im:02d}_roughness.png \n")
                f.write(f"bump {im:02d}_normal.png \n")

                # symlink the material files to the output folder
                dest_dir = os.path.dirname(filePath)
                material_files = ["basecolor.png", "metallic.png", "normal.png", "roughness.png"]
                for material_file in material_files:
                    src_path = os.path.join(mat_path, material_file)
                    dest_path = os.path.join(dest_dir, f"{im:02d}_{material_file}")
                    rel_src_path = os.path.relpath(src_path, dest_dir)
                    if os.path.exists(dest_path):
                        continue
                    os.symlink(rel_src_path, dest_path)
                    print(f'Created symlink: {rel_src_path} -> {dest_path}')

        return max_dim, material_ids


    def genMultiObj(self, folderPath, bComputeNormal=False):
        if len(self.faces) == 0:
            print("no mesh")
            return False
        if bComputeNormal:
            self.computeNormals()

        if len(self.matStartId) == 0:
            print("No mats")
            return

        for im in range(len(self.matStartId)):
            if im == len(self.matStartId) - 1:
                endId = len(self.faces)
            else:
                endId = self.matStartId[im + 1]

            usePoints = np.zeros(len(self.points), int) - 1
            pointMap = np.zeros(len(self.points), int) - 1
            useTex = np.zeros(len(self.uvs), int) - 1
            texMap = np.zeros(len(self.uvs), int) - 1
            nP = 0
            nT = 0
            for i in range(self.matStartId[im], endId):
                for ii in range(3):
                    if pointMap[self.faces[i][ii]-1] == -1:
                        pointMap[self.faces[i][ii] - 1] = nP + 1
                        usePoints[nP] = self.faces[i][ii] - 1
                        nP += 1
                    if texMap[self.facesUV[i][ii]-1] == -1:
                        texMap[self.facesUV[i][ii] - 1] = nT + 1
                        useTex[nT] = self.facesUV[i][ii] - 1
                        nT += 1

            filePath = folderPath + "/%s.obj"%self.matNames[im]

            with open(filePath, "w") as f:
                #write v
                for ii in range(len(usePoints)):
                    if usePoints[ii] == -1:
                        break
                    f.write("v %f %f %f\n"%(self.points[usePoints[ii]][0], self.points[usePoints[ii]][1], self.points[usePoints[ii]][2]))

                if bComputeNormal:
                    for ii in range(len(usePoints)):
                        if usePoints[ii] == -1:
                            break
                        f.write("vn %f %f %f\n"%(self.normals[usePoints[ii]][0], self.normals[usePoints[ii]][1], self.normals[usePoints[ii]][2]))
                        # f.write("vn %f %f %f\n" % (0, 0, 1))
                # write uv
                for ii in range(len(useTex)):
                    if useTex[ii] == -1:
                        break
                    f.write("vt %f %f\n" % (self.uvs[useTex[ii]][0], self.uvs[useTex[ii]][1]))
                #write face
                # f.write("usemtl mat_%d\n"%matId)


                f.write("usemtl %s\n"%self.matNames[im])

                if not bComputeNormal:
                    for i in range(self.matStartId[im], endId):
                        f.write("f %d/%d %d/%d %d/%d\n" %
                                (pointMap[self.faces[i][0]-1], texMap[self.facesUV[i][0]-1],
                                 pointMap[self.faces[i][1]-1], texMap[self.facesUV[i][1]-1],
                                 pointMap[self.faces[i][2]-1], texMap[self.facesUV[i][2]-1]))
                else:
                    for i in range(self.matStartId[im], endId):
                        f.write("f %d/%d/%d %d/%d/%d %d/%d/%d\n" %
                                (pointMap[self.faces[i][0]-1], texMap[self.facesUV[i][0]-1], pointMap[self.faces[i][0]-1],
                                 pointMap[self.faces[i][1]-1], texMap[self.facesUV[i][1]-1], pointMap[self.faces[i][1]-1],
                                 pointMap[self.faces[i][2]-1], texMap[self.facesUV[i][2]-1], pointMap[self.faces[i][2]-1]))

        return True


    def genMatList(self, filePath):
        with open(filePath, "w") as f:
            for matname in self.matNames:
                f.write("%s\n"%matname)
    def genInfo(self, filePath):
        with open(filePath, "w") as f:
            minP = np.min(self.points, axis=0)
            maxP = np.max(self.points, axis=0)
            print(minP, maxP)
            f.write("%f %f %f\n" % (minP[0], minP[1], minP[2]))
            f.write("%f %f %f\n" % (maxP[0], maxP[1], maxP[2]))


    def translate(self, translation):
        self.points += translation

    def rotate(self, axis, degAngle):
        self.points = rotateVector(self.points, axis, np.deg2rad(degAngle))

    def reCenter(self):
        minP = np.min(self.points, 0)
        maxP = np.max(self.points, 0)

        center = 0.5*minP + 0.5*maxP

        self.translate(-center)

    def addShape(self, otherShape):
        curPN = len(self.points)
        curUN = len(self.uvs)
        curFN = len(self.faces)
        if curPN == 0:
            self.points = np.copy(otherShape.points)
            self.uvs = np.copy(otherShape.uvs)
            self.faces = np.copy(otherShape.faces)
            self.facesUV = np.copy(otherShape.facesUV)
            self.matNames += otherShape.matNames
            self.matStartId = np.append(self.matStartId, otherShape.matStartId + curFN).astype(int)
        else:
            self.points = np.row_stack([self.points, otherShape.points])
            self.uvs = np.row_stack([self.uvs, otherShape.uvs])
            self.faces = np.row_stack([self.faces, otherShape.faces+curPN])
            self.facesUV = np.row_stack([self.facesUV, otherShape.facesUV+curUN])
            self.matNames += otherShape.matNames
            self.matStartId = np.append(self.matStartId, otherShape.matStartId+curFN).astype(int)

    def _addMorphCircle(self, center=(0,0,0), axisA = 1.0, axisB = 1.0, X=(1,0,0), Z=(0,0,1), circelRes = (50, 100), matName = "mat"):
        circelRes = (int(circelRes[0]), int(circelRes[1]))
        
        X = np.reshape(X,3)
        Z = np.reshape(Z,3)
        Y = np.cross(Z, X)
        startPId = len(self.points)
        startUId = len(self.uvs)
        startFaceId = len(self.faces)

        center = np.reshape(center, 3)
        points = []
        uvs = []
        points.append(center)
        uvs.append((0.5, 0.5))

        # create points
        for iy in range(1, circelRes[0]):
            for ix in range(circelRes[1]):
                ra = float(axisA) * iy / (circelRes[0] - 1)
                rb = float(axisB) * iy / (circelRes[0] - 1)


                phi = float(ix) /(circelRes[1]) * 2.0 * np.pi

                x = ra *  np.cos(phi)
                y = rb *  np.sin(phi)


                p = x*X + y*Y + center
                points.append(p)
        # create uvs
        for iy in range(1, circelRes[0]):
            for ix in range(circelRes[1]):
                ra = float(axisA) * iy / (circelRes[0] - 1)
                rb = float(axisB) * iy / (circelRes[0] - 1)

                phi = float(ix) / (circelRes[1]) * 2.0 * np.pi

                x = ra * np.cos(phi)
                y = rb * np.sin(phi)

                ux = float(iy) / (circelRes[0] - 1) * np.cos(phi)
                uy = float(iy) / (circelRes[0] - 1) * np.sin(phi)

                u = 0.5 * ux + 0.5
                v = 0.5 * uy + 0.5
                uvs.append((u, v))

                # r = (x**2 + y**2)**0.5
                # rl = ((axisA*np.cos(phi))**2 + (axisB*np.sin(phi))**2)**0.5
                #
                # if phi > np.pi*7.0/4.0 or phi <= np.pi / 4.0:
                #     ul = 1.0
                #     vl = -0.5 * np.tan(phi) + 0.5
                # elif phi > np.pi / 4.0 and phi <= np.pi * 3.0 / 4.0:
                #     vl = 0.0
                #     ul = 1.0 / np.tan(phi) * 0.5 + 0.5
                # elif phi > np.pi * 3.0 / 4.0 and phi <= np.pi * 5.0 / 4.0:
                #     ul = 0.0
                #     vl = 0.5 * np.tan(phi) + 0.5
                # else: #phi > np.pi * 5.0 / 4.0 and phi <= np.pi * 7.0 / 4.0:
                #     vl = 1.0
                #     ul = -1.0 / np.tan(phi) * 0.5 + 0.5
                # u = (ul-0.5) / rl * r + 0.5
                # v = (vl-0.5) / rl * r + 0.5
                # uvs.append((u, v))

        if startPId == 0:
            self.points = np.reshape(points, (-1, 3))
            self.uvs = np.rehsape(uvs, (-1, 2))
        else:
            self.points = np.row_stack([self.points, points])
            self.uvs = np.row_stack([self.uvs, uvs])

        # create faces
        tempFaces = []
        for iy in range(circelRes[0] - 1):
            for ix in range(circelRes[1]):
                if iy == 0:
                    curId = 1
                    rightId = 1
                    bottomId = 1 + ix + 1
                    if ix == circelRes[1] - 1:
                        rightBottomId = 1 + 1
                    else:
                        rightBottomId = 1 + ix + 1 + 1
                else:
                    curId = 1 + (iy - 1) * circelRes[1] + ix + 1
                    bottomId = 1 + (iy) * circelRes[1] + ix + 1
                    if ix == circelRes[1] - 1:
                        rightId = 1 + (iy - 1) * circelRes[1] + 1
                        rightBottomId = 1 + (iy) * circelRes[1] + 1
                    else:
                        rightId = 1 + (iy - 1) * circelRes[1] + ix + 1 + 1
                        rightBottomId = 1 + (iy) * circelRes[1] + ix + 1 + 1
                if iy != 0:
                    tempFaces.append((curId, rightBottomId, rightId))
                tempFaces.append((curId, bottomId, rightBottomId))

        tempFaces = np.reshape(tempFaces, (-1,3))

        if len(self.faces) == 0:
            self.faces = tempFaces.copy()
            self.facesUV = tempFaces.copy()
            self.matStartId = self.matStartId = np.asarray([0],int)
        else:
            self.faces = np.row_stack([self.faces, tempFaces+startPId])
            self.facesUV = np.row_stack([self.facesUV, tempFaces+startUId])
            self.matStartId = np.append(self.matStartId, [startFaceId])

        self.matNames.append(matName)




class HeightFieldCreator:
    def __init__(self, initSize = (5,5), maxHeight = (-0.2,0.2), bFixCorner = True):
        self.initSize = initSize
        self.bFixCorner = bFixCorner
        self.initNum = self.initSize[0]*self.initSize[1]
        self.maxHeight = maxHeight
        self.heightField = None

    def __initializeHeigthField(self):
        heights = np.random.uniform(self.maxHeight[0], self.maxHeight[1], self.initNum)
        # if self.bFixCorner:
        #     heights[0] = heights[self.initSize[1]-1] = heights[(self.initSize[0]-1)*self.initSize[1]] = heights[-1] = 0
        initHeightField = heights.reshape(self.initSize)
        self.initHeightField = initHeightField
        return initHeightField

    def genHeightField(self, targetSize = (36, 36)):
        halfSize = (int(targetSize[0]/6*5), int(targetSize[1]/6*5))
        if halfSize[0] < self.initSize[0] or halfSize[1] < self.initSize[1]:
            print("target size should be double as init size")
            return None
        initHeight = self.__initializeHeigthField()
        if self.bFixCorner:
            bounder = np.zeros((self.initSize[0]+2, self.initSize[1]+2))
            bounder[1:-1, 1:-1] = initHeight
            initHeight = bounder
        heightField_half = cv2.resize(initHeight, halfSize, interpolation=cv2.INTER_CUBIC)#
        if self.bFixCorner:
            bounder = np.zeros(halfSize)
            bounder[1:-1, 1:-1] = heightField_half[1:-1, 1:-1]
            initHeight = bounder
        heightField = cv2.resize(initHeight, targetSize)  #
        self.heightField = heightField
        self.targetSize = targetSize
        return heightField

    def genObj(self, filePath):
        if type(self.heightField) == type(None):
            print("no generated height fields")
            return False


        with open(filePath, "w") as f:
            #write v
            for iy in range(self.targetSize[0]):
                for ix in range(self.targetSize[1]):
                    f.write("v %f %f %f\n"%
                            (float(ix)/(self.targetSize[1]-1),
                             float(iy)/(self.targetSize[0]-1),
                             self.heightField[iy, ix]))
            #write f
            for iy in range(self.targetSize[0]-1):
                for ix in range(self.targetSize[1]-1):
                    curId = iy * self.targetSize[1] + ix + 1
                    rightId = iy * self.targetSize[1] + ix + 1 +1
                    bottomId = (iy+1) * self.targetSize[1] + ix+1
                    rightBottomId = (iy+1) * self.targetSize[1] + ix + 1+1
                    f.write("f %d %d %d\n"%
                            (curId, rightBottomId, rightId))
                    f.write("f %d %d %d\n" %
                            (curId, bottomId, rightBottomId))
        return True

class Ellipsoid(Shape):
    # meshRes: rows x columns, latitude res x longitude res
    def __init__(self, a = 1.0, b = 1.0, c = 1.0, meshRes = (50, 100)):
        super(Ellipsoid, self).__init__()
        if meshRes[1] % 2 != 0:
            print("WARN: longitude res is supposed to be even")
        self.axisA = a
        self.axisB = b
        self.axisC = c
        self.meshRes = meshRes

        self.numPoints = (self.meshRes[0] - 2) * self.meshRes[1] + 2


    def genShape(self, matName = "mat"):
        super(Ellipsoid, self).__init__()


        self.points.append((0,0,self.axisC))
        self.uvs.append((0,0))


        #create points
        for iy in range(1,self.meshRes[0]-1):
            for ix in range(self.meshRes[1]):
                v = float(iy) / (self.meshRes[0]-1)
                u = float(ix) / (self.meshRes[1]/2)

                theta = np.pi/2.0 - v * np.pi
                phi = u * np.pi

                x = self.axisA * np.cos(theta) * np.cos(phi)
                y = self.axisB * np.cos(theta) * np.sin(phi)
                z = self.axisC * np.sin(theta)

                self.points.append((x,y,z))
        self.points.append((0, 0, -self.axisC))

        #create uvs
        for iy in range(1, self.meshRes[0] - 1):
            for ix in range(self.meshRes[1]):
                v = float(iy) / (self.meshRes[0] - 1)
                u = float(ix) / (self.meshRes[1] / 2)
                if u > 1.0:
                    u = u - 1.0
                self.uvs.append((u,v))
        self.uvs.append((1.0,1.0))

        #create faces
        for iy in range(self.meshRes[0]-1):
            for ix in range(self.meshRes[1]):
                if iy == 0:
                    curId = 1
                    rightId = 1
                    bottomId = 1 + ix + 1
                    if ix == self.meshRes[1] - 1:
                        rightBottomId = 1 + 1
                    else:
                        rightBottomId = 1+ ix + 1 + 1
                elif iy == self.meshRes[0]-2:
                    curId = 1 + (iy-1) * self.meshRes[1] + ix + 1
                    bottomId = 1 + (iy) * self.meshRes[1] + 1
                    if ix == self.meshRes[1] - 1:
                        rightId = 1 + (iy-1) * self.meshRes[1] + 1
                    else:
                        rightId = 1 + (iy-1) * self.meshRes[1] + ix + 1 + 1
                    rightBottomId = 1 + (iy) * self.meshRes[1] + 1
                else:
                    curId = 1 + (iy - 1) * self.meshRes[1] + ix + 1
                    bottomId = 1 + (iy) * self.meshRes[1] + ix + 1
                    if ix == self.meshRes[1] - 1:
                        rightId = 1 + (iy - 1) * self.meshRes[1] + 1
                        rightBottomId = 1 + (iy) * self.meshRes[1] + 1
                    else:
                        rightId = 1 + (iy - 1) * self.meshRes[1] + ix + 1 + 1
                        rightBottomId = 1 + (iy) * self.meshRes[1]+ ix + 1 + 1
                if iy != 0:
                    self.faces.append((curId, rightBottomId, rightId))
                if iy != self.meshRes[0]-2:
                    self.faces.append((curId, bottomId, rightBottomId))

        self.points = np.reshape(self.points, (-1,3)).astype(float)
        self.uvs = np.reshape(self.uvs, (-1,2)).astype(float)
        self.faces = np.reshape(self.faces, (-1, 3)).astype(int)
        self.facesUV = np.copy(self.faces)
        self.matNames = [matName]
        self.matStartId = np.asarray([0],int)

    def applyHeightField(self, heightFields):
        if len(self.points) == 0:
            print("no points")
            return False
        if len(heightFields.shape) != 2 and len(heightFields.shape) != 3:
            print("wrong shape of heightfiels")
            return False
        if len(heightFields.shape) == 3:
            heightField = heightFields[0]


        for i,point in enumerate(self.points):
            uv = self.uvs[i]
            normal = np.reshape(point,-1) / (self.axisA, self.axisB, self.axisC)
            normal = normal / np.linalg.norm(normal)
            xy = uv * (heightField.shape[1], heightField.shape[0])
            h = subPix(heightField, xy[0], xy[1])#cv2.getRectSubPix(heightField, (1,1), (xy[0], xy[1]))

            self.points[i] = point + normal * h


class Cube(Shape):
    """faces:
    front: c = 1
    back: c = -1
    left: a = -1
    right: a = 1
    up: b = 1
    down: b = -1

    """
    def __init__(self, a=1.0, b=1.0, c=1.0, faceRes=(50, 50)):
        super(Cube,self).__init__()
        self.axisA = a
        self.axisB = b
        self.axisC = c
        self.faceRes = faceRes
        self.pointNumPerFace = faceRes[0] * faceRes[1]

        self.numPoints = (self.faceRes[0]) * self.faceRes[1] * 6


    def genShape(self, matName = "mat"):
        """ compute points, uvs, faces according to the parameters for the shape type """
        super(Cube, self).__init__()


        #uvs
        for iy in range(self.faceRes[0]):
            for ix in range(self.faceRes[1]):
                u = float(ix) / (self.faceRes[1] - 1)
                v = float(iy) / (self.faceRes[0] - 1)
                self.uvs.append((u,v))
        self.uvs = np.reshape(self.uvs, (-1,2))

        #face:
        #oneFace:
        oneFaces = []
        for iy in range(self.faceRes[0] - 1):
            for ix in range(self.faceRes[1] - 1):
                curId = iy * self.faceRes[1] + ix + 1
                rightId = iy * self.faceRes[1] + ix + 1 + 1
                bottomId = (iy + 1) * self.faceRes[1] + ix + 1
                rightBottomId = (iy + 1) * self.faceRes[1] + ix + 1 + 1

                oneFaces.append((curId, rightBottomId, rightId))
                oneFaces.append((curId, bottomId, rightBottomId))
        oneFaces = np.reshape(oneFaces, (-1,3)).astype(int)
        self.faces = np.row_stack([oneFaces,
                                      oneFaces + self.pointNumPerFace,
                                      oneFaces + self.pointNumPerFace*2,
                                      oneFaces + self.pointNumPerFace*3,
                                      oneFaces + self.pointNumPerFace*4,
                                      oneFaces + self.pointNumPerFace*5])
        self.facesUV = self.faces.copy()#np.row_stack([oneFaces, oneFaces, oneFaces, oneFaces, oneFaces, oneFaces])

        #points
        #front
        for uv in self.uvs:
            xy = uv * (self.axisA, -self.axisB) * 2.0 + (-self.axisA, self.axisB)
            point = (xy[0], xy[1], self.axisC)
            self.points.append(point)

        # back
        for uv in self.uvs:
            xy = uv * (self.axisA, self.axisB) * 2.0 + (-self.axisA, -self.axisB)
            point = (xy[0], xy[1], -self.axisC)
            self.points.append(point)

        #left
        for uv in self.uvs:
            zy = uv * (self.axisC, -self.axisB) * 2.0 + (-self.axisC, self.axisB)
            point = (-self.axisA, zy[1], zy[0])
            self.points.append(point)

        # right
        for uv in self.uvs:
            zy = uv * (-self.axisC, -self.axisB) * 2.0 + (self.axisC, self.axisB)
            point = (self.axisA, zy[1], zy[0])
            self.points.append(point)

        # up
        for uv in self.uvs:
            xz = uv * (self.axisA, self.axisC) * 2.0 + (-self.axisA, -self.axisC)
            point = (xz[0], self.axisB, xz[1])
            self.points.append(point)

        # down
        for uv in self.uvs:
            xz = uv * (self.axisA, -self.axisC) * 2.0 + (-self.axisA, self.axisC)
            point = (xz[0], -self.axisB, xz[1])
            self.points.append(point)

        self.uvs = np.reshape(np.row_stack([self.uvs, self.uvs, self.uvs, self.uvs, self.uvs, self.uvs]), (-1, 2))

        self.points = np.reshape(self.points, (-1, 3)).astype(float)
        self.uvs = np.reshape(self.uvs, (-1, 2)).astype(float)
        self.faces = np.reshape(self.faces, (-1, 3)).astype(int)
        self.facesUV = np.reshape(self.facesUV, (-1, 3)).astype(int)
        self.matNames = ["%s_%d"%(matName, 0),
                         "%s_%d"%(matName, 1),
                         "%s_%d"%(matName, 2),
                         "%s_%d"%(matName, 3),
                         "%s_%d"%(matName, 4),
                         "%s_%d"%(matName, 5)]
        numFacePerFace = len(oneFaces)
        self.matStartId = np.asarray([0,
                                      numFacePerFace,
                                      numFacePerFace*2,
                                      numFacePerFace*3,
                                      numFacePerFace*4,
                                      numFacePerFace*5],int)

    # def genObj(self, filePath):
    #     if len(self.faces) == 0:
    #         print "no mesh"
    #         return False
    #     with open(filePath, "w") as f:
    #         #write v
    #         for point in self.points:
    #             f.write("v %f %f %f\n"%(point[0], point[1], point[2]))
    #         # write uv
    #         for uv in self.uvs:
    #             f.write("vt %f %f\n" % (uv[0], uv[1]))
    #         #write face
    #         for i,face in enumerate(self.faces):
    #             f.write("f %d/%d %d/%d %d/%d\n" %
    #                     (face[0], (face[0]-1)%self.pointNumPerFace+1,
    #                      face[1], (face[1]-1)%self.pointNumPerFace+1,
    #                      face[2], (face[2]-1)%self.pointNumPerFace+1))
    #     return True


    def applyHeightField(self, heightFields):
        if len(self.points) == 0:
            print("no points")
            return False
        if len(heightFields.shape) != 2 and len(heightFields.shape) != 3:

            print("wrong shape of heightfiels")
            return False

        if len(heightFields.shape) == 2:
            newH = []
            for i in range(6):
                newH.append(heightFields)
            heightFields = newH
        else:
            if heightFields.shape[0] < 6:
                newH = []
                for i in range(6):
                    if i < heightFields.shape[0]:
                        newH.append(heightFields[i])
                    else:
                        newH.append(heightFields[-1])
                heightFields = newH

        # modify points
        # front
        heightField = heightFields[0]
        normal = np.asarray((0,0,1))
        offSet = 0
        for i in range(self.pointNumPerFace):
            uv = self.uvs[i]
            xy = uv * (heightField.shape[1], heightField.shape[0])
            h = subPix(heightField, xy[0], xy[1])
            self.points[i + offSet] = self.points[i+offSet] + h * normal


        # back
        heightField = heightFields[1]
        normal = np.asarray((0, 0, -1))
        offSet = self.pointNumPerFace*1
        for i in range(self.pointNumPerFace):
            uv = self.uvs[i]
            xy = uv * (heightField.shape[1], heightField.shape[0])
            h = subPix(heightField, xy[0], xy[1])
            self.points[i + offSet] = self.points[i + offSet] + h * normal

        # left
        heightField = heightFields[2]
        normal = np.asarray((-1, 0, 0))
        offSet = self.pointNumPerFace * 2
        for i in range(self.pointNumPerFace):
            uv = self.uvs[i]
            xy = uv * (heightField.shape[1], heightField.shape[0])
            h = subPix(heightField, xy[0], xy[1])
            self.points[i + offSet] = self.points[i + offSet] + h * normal

        # right
        heightField = heightFields[3]
        normal = np.asarray((1, 0, 0))
        offSet = self.pointNumPerFace * 3
        for i in range(self.pointNumPerFace):
            uv = self.uvs[i]
            xy = uv * (heightField.shape[1], heightField.shape[0])
            h = subPix(heightField, xy[0], xy[1])
            self.points[i + offSet] = self.points[i + offSet] + h * normal

        # up
        heightField = heightFields[4]
        normal = np.asarray((0, 1, 0))
        offSet = self.pointNumPerFace * 4
        for i in range(self.pointNumPerFace):
            uv = self.uvs[i]
            xy = uv * (heightField.shape[1], heightField.shape[0])
            h = subPix(heightField, xy[0], xy[1])
            self.points[i + offSet] = self.points[i + offSet] + h * normal

        # down
        heightField = heightFields[5]
        normal = np.asarray((0, -1, 0))
        offSet = self.pointNumPerFace * 5
        for i in range(self.pointNumPerFace):
            uv = self.uvs[i]
            xy = uv * (heightField.shape[1], heightField.shape[0])
            h = subPix(heightField, xy[0], xy[1])
            self.points[i + offSet] = self.points[i + offSet] + h * normal

class Cylinder(Shape):
    def __init__(self, a=1.0, b=1.0, c=1.0, meshRes=(50, 150), radiusRes = 20):
        super(Cylinder,self).__init__()

        self.axisA = a
        self.axisB = b
        self.axisC = c
        self.meshRes = meshRes



    def genShape(self, matName = "mat"):
        super(Cylinder, self).__init__()


        # create points
        for iy in range(self.meshRes[0]):
            for ix in range(self.meshRes[1]):
                v = float(iy) / (self.meshRes[0] - 1)
                u = float(ix) / (self.meshRes[1] / 2)

                phi = u * np.pi

                x = self.axisA * np.cos(phi)
                y = self.axisB * np.sin(phi)
                z = self.axisC - self.axisC * v * 2.0

                self.points.append((x, y, z))



        # create uvs
        for iy in range(self.meshRes[0]):
            for ix in range(self.meshRes[1]):
                v = float(iy) / (self.meshRes[0] - 1)
                u = float(ix) / (self.meshRes[1] / 2)
                if u > 1.0:
                    u = u - 1.0
                self.uvs.append((u, v))


        # create faces
        for iy in range(self.meshRes[0]-1):
            for ix in range(self.meshRes[1]):

                curId = iy * self.meshRes[1] + ix + 1
                bottomId = (iy + 1) * self.meshRes[1] + ix + 1
                if ix == self.meshRes[1] - 1:
                    rightId = iy * self.meshRes[1] + 1
                    rightBottomId = (iy + 1) * self.meshRes[1] + 1
                else:
                    rightId = (iy) * self.meshRes[1] + ix + 1 + 1
                    rightBottomId = (iy + 1) * self.meshRes[1] + ix + 1 + 1

                self.faces.append((curId, rightBottomId, rightId))
                self.faces.append((curId, bottomId, rightBottomId))

        # self.points.append((0, 0, self.axisC))
        # self.points.append((0, 0, -self.axisC))
        # self.uvs.append((0.5, 0))
        # self.uvs.append((0.5, 1))
        # #up
        # centerid = self.meshRes[0]*self.meshRes[1] + 1
        # for ix in range(self.meshRes[1]-1):
        #     self.faces.append((centerid, ix+1, ix+2))
        # self.faces.append((centerid, self.meshRes[1], 1))
        #
        # # #down
        # centerid = self.meshRes[0] * self.meshRes[1] + 2
        # for ix in range(self.meshRes[1] - 1):
        #     self.faces.append((centerid, ix+(self.meshRes[0] - 1)*self.meshRes[1] + 2, ix+(self.meshRes[0] - 1)*self.meshRes[1]+1))
        # self.faces.append((centerid, (self.meshRes[0] - 1)*self.meshRes[1]+1, (self.meshRes[0])*self.meshRes[1]))

        self.points = np.reshape(self.points, (-1, 3)).astype(float)
        self.uvs = np.reshape(self.uvs, (-1, 2)).astype(float)
        self.faces = np.reshape(self.faces, (-1, 3)).astype(int)
        self.facesUV = np.copy(self.faces)
        self.matNames = ["%s_0"%matName]
        self.matStartId = self.matStartId = np.asarray([0],int)

        self._addMorphCircle((0, 0, self.axisC), self.axisA, self.axisB, X=(1,0,0), Z=(0,0,1),
                             circelRes=[self.meshRes[0]/2, self.meshRes[1]], matName="%s_1"%matName)

        self._addMorphCircle((0, 0, -self.axisC), self.axisA, self.axisB, X=(1, 0, 0), Z=(0, 0, -1),
                             circelRes=[self.meshRes[0]/2, self.meshRes[1]], matName="%s_2" % matName)



    def applyHeightField(self, heightFields, smoothCircleBoundRate = 0.25):
        if len(self.points) == 0:
            print("no points")
            return False
        if len(heightFields.shape) != 2 and len(heightFields.shape) != 3:

            print("wrong shape of heightfiels")
            return False

        if len(heightFields.shape) == 2:
            newH = []
            for i in range(3):
                newH.append(heightFields)
            heightFields = newH
        else:
            if heightFields.shape[0] < 3:
                newH = []
                for i in range(3):
                    if i < heightFields.shape[0]:
                        newH.append(heightFields[i])
                    else:
                        newH.append(heightFields[-1])
                heightFields = newH

        heightField = heightFields[0]
        i = 0
        for iy in range(self.meshRes[0]):
            for ix in range(self.meshRes[1]):
                u = float(ix) / (self.meshRes[1] / 2)

                phi = u * np.pi

                x = self.axisA * np.cos(phi)
                y = self.axisB * np.sin(phi)

                normal = np.reshape((x,y,0),-1)/ (self.axisA, self.axisB, self.axisC)
                normal = normal / np.linalg.norm(normal)
                xy = self.uvs[i] * (heightField.shape[1], heightField.shape[0])
                h = subPix(heightField, xy[0], xy[1])  # cv2.getRectSubPix(heightField, (1,1), (xy[0], xy[1]))

                self.points[i] += normal * h
                i+=1

        heightField = heightFields[1]
        circelRes = [int(self.meshRes[0] / 2), int(self.meshRes[1])]
        for iy in range(circelRes[0] - 1):
            for ix in range(circelRes[1]):
                normal = np.reshape((0, 0, 1),-1)
                xy = self.uvs[i] * (heightField.shape[1], heightField.shape[0])
                h = subPix(heightField, xy[0], xy[1])  # cv2.getRectSubPix(heightField, (1,1), (xy[0], xy[1]))

                l = np.linalg.norm(self.uvs[i] * 2 - 1.0)
                if l > smoothCircleBoundRate:
                    r = (1.0-l)/(1.0-smoothCircleBoundRate)
                    h = (1.0-(r-1.0)**2.0)* h

                self.points[i] += normal * h
                i+=1

        heightField = heightFields[2]
        circelRes = [int(self.meshRes[0] / 2), int(self.meshRes[1])]
        for iy in range(circelRes[0] - 1):
            for ix in range(circelRes[1]):
                normal = np.reshape((0, 0, -1), -1)
                xy = self.uvs[i] * (heightField.shape[1], heightField.shape[0])
                h = subPix(heightField, xy[0], xy[1])  # cv2.getRectSubPix(heightField, (1,1), (xy[0], xy[1]))

                l = np.linalg.norm(self.uvs[i] * 2 - 1.0)
                if l > smoothCircleBoundRate:
                    r = (1.0 - l) / (1.0 - smoothCircleBoundRate)
                    h = (1.0 - (r - 1.0) ** 2.0) * h

                self.points[i] += normal * h
                i += 1



class MultiShape(Shape):
    """
    0: ellipsoid
    1: cube
    2: cylinder

    """
    def __init__(self,
                 numShape = 6, smoothPossibility = 0.1, axisRange = (0.25, 2.0), heightRangeRate = (0, 0.2),
                 translateRangeRate = (0, 0.5), rotateRange = (0, 180), candShapes=[0,1,2]):
        super(MultiShape, self).__init__()
        self.numShape = numShape
        self.smoothPossibility = smoothPossibility
        self.axisRange = axisRange
        self.heightRangeRate = heightRangeRate
        self.translateRangeRate = translateRangeRate
        self.rotateRange = rotateRange
        self.candShapes = candShapes
    def genShape(self, no_hf=False):
        """ For each shape, randomly sample parameters (axis, height field, rotation, translation) and create the shape. """
        super(MultiShape, self).__init__()


        primitive_ids = []
        axis_vals_s = []
        translations = []
        translation1s = []
        rotations = []
        rotation1s = []
        height_fields_s = []
        for iS in range(self.numShape):
            rp = np.random.permutation(self.candShapes)
            axisVals = np.random.uniform(self.axisRange[0], self.axisRange[1], 3)
            hfs = []
            minA = axisVals.min()*2.0
            maxA = axisVals.max()*2.0
            print(minA, maxA)
            maxH = np.random.uniform(self.heightRangeRate[0]*minA, self.heightRangeRate[1]*minA, 6)
            translation = np.random.uniform(self.translateRangeRate[0]*maxA, self.translateRangeRate[1]*maxA, 3)
            translation1 = np.random.uniform(self.translateRangeRate[0] * maxA, self.translateRangeRate[1] * maxA, 3)
            rotation = np.random.uniform(self.rotateRange[0], self.rotateRange[1], 3)
            rotation1 = np.random.uniform(self.rotateRange[0], self.rotateRange[1], 3)
            for ih in range(6):
                smoothR = np.random.uniform(0,1,1)[0]
                if smoothR <= self.smoothPossibility or maxH[ih] == 0:
                    hf = np.zeros((36,36))
                else:
                    hfg = HeightFieldCreator(maxHeight=(-maxH[ih], maxH[ih]))
                    hf = hfg.genHeightField()
                hfs.append(hf)
            hfs = np.reshape(hfs, (6,) + hf.shape)
            if no_hf:
                hfs = np.zeros_like(hfs)

            if rp[0] == 0:
                subShape = Ellipsoid(axisVals[0], axisVals[1], axisVals[2])
            elif rp[0] == 1:
                subShape = Cube(axisVals[0], axisVals[1], axisVals[2])
            elif rp[0] == 2:
                subShape = Cylinder(axisVals[0], axisVals[1], axisVals[2])

            subShape.genShape(matName="mat_shape%d"%iS)
            subShape.applyHeightField(hfs)


            subShape.rotate((1, 0, 0), rotation[0])
            subShape.rotate((0, 1, 0), rotation[1])
            subShape.rotate((0, 0, 1), rotation[2])
            subShape.translate(translation)

            if iS != 0:
                self.rotate((1, 0, 0), rotation1[0])
                self.rotate((0, 1, 0), rotation1[1])
                self.rotate((0, 0, 1), rotation1[2])
                self.translate(translation1)

            self.addShape(subShape)
            primitive_ids.append(rp[0])
            axis_vals_s.append(axisVals)
            translations.append(translation)
            translation1s.append(translation1)
            rotations.append(rotation)
            rotation1s.append(rotation1)
            height_fields_s.append(hfs)

        self.reCenter()
        return primitive_ids, axis_vals_s, translations, translation1s, rotations, rotation1s, height_fields_s


def createShapes(outFolder, shapeNum, subObjNum = 6):
    if not os.path.isdir(outFolder):
        os.makedirs(outFolder)

    for i in range(shapeNum):
        ms = MultiShape(subObjNum)
        subFolder = outFolder + "/Shape__%d"%i
        if not os.path.isdir(subFolder):
            os.makedirs(subFolder)
        ms.genShape()
        ms.genObj(subFolder + "/object.obj", bMat=True)
        ms.genMatList(subFolder + "/object.txt")
        ms.genInfo(subFolder + "/object.info")


def createVarObjShapes(outFolder, shapeIds, uuid_str='', sub_obj_nums=[1, 2, 3, 4, 5, 6, 7, 8, 9], sub_obj_num_poss=[1, 2, 3, 7, 10, 7, 3, 2, 1],
                       bMultiObj=False, bPermuteMat=True, candShapes=[0,1,2],
                       bScaleMesh=False, bMaxDimRange=[0.3, 0.5], smooth_probability=1.0, no_hf=False):
    """
    randomly sample one of subObjNums (each with subObjPoss possibilities) number of sub objects for each scene,
    create a MultiShape, and save the .obj shape, .txt material list, and .info files.
    """
    if len(sub_obj_nums) != len(sub_obj_num_poss):
        print("In correct obj num distribution")
        exit()

    if not os.path.isdir(outFolder):
        os.makedirs(outFolder)


    sub_obj_bound = np.reshape(sub_obj_num_poss, -1).astype(float)
    sub_obj_bound = sub_obj_bound / np.sum(sub_obj_bound)
    sub_obj_bound = np.cumsum(sub_obj_bound)  # normalized, cumulative sum of subObjPoss (possibility)

    if sub_obj_bound[-1] != 1.0:
        print("in correct bound")
        sub_obj_bound[-1] = 1.0  # setting 0.999... to 1.0


    counts = np.zeros(len(sub_obj_nums))

    chooses = np.random.uniform(0, 1.0, len(shapeIds))
    output_paths = []
    shapes_parameters = []
    for ii, i in enumerate(shapeIds):  # for each MultiShape
        shape_parameters = {'uuid_str': uuid_str}

        choose = chooses[ii]
        sub_obj_num = sub_obj_nums[-1]
        for iO in range(len(sub_obj_bound)):
            if choose < sub_obj_bound[iO]:  # randomly choose a sub obj
                sub_obj_num = sub_obj_nums[iO]
                counts[iO] += 1
                break

        shape_parameters['sub_obj_num'] = sub_obj_num
        shape_parameters['sub_objs'] = [{} for _ in range(sub_obj_num)]
        print(f'i: {i}, sub_obj_num: {sub_obj_num}')
        ms = MultiShape(sub_obj_num, candShapes=candShapes, smoothPossibility=smooth_probability)
        # create a folder for each shape, with a uuid name
        new_uuid = str(uuid.uuid4())
        subFolder = Path(outFolder) / new_uuid / 'shape'
        subFolder.mkdir(parents=True, exist_ok=True)
        output_paths.append(subFolder / 'object.obj')
        subFolder = str(subFolder.resolve())

        sub_objs_vals = list(ms.genShape(no_hf=no_hf))
        if bPermuteMat:
            ms.permuteMatIds()

        if bMultiObj:
            ms.genMultiObj(subFolder, bComputeNormal=True)

        max_dim, material_ids = ms.genObj(subFolder + "/object.obj", bMat=True, bComputeNormal=True, bScaleMesh=bScaleMesh, bMaxDimRange=bMaxDimRange)
        shape_parameters['max_dim'] = max_dim
        sub_objs_vals.append(material_ids)
        for i_key, key in enumerate(['primitive_id', 'axis_vals', 'translation', 'translation1', 'rotation', 'rotation1', 'height_fields', 'material_id']):
            for iS in range(sub_obj_num):
                shape_parameters['sub_objs'][iS][key] = sub_objs_vals[i_key][iS].tolist() if isinstance(sub_objs_vals[i_key][iS], np.ndarray) else sub_objs_vals[i_key][iS]
        ms.genMatList(subFolder + "/object.txt")
        ms.genInfo(subFolder + "/object.info")

        shapes_parameters.append(shape_parameters)
    print(counts)
    return output_paths, shapes_parameters


mat_keys = ["name", "basecolor", "metallic", "normal", "roughness"]


def get_matsynth_material(base_output_dir):
    ds = load_dataset(
        "gvecchio/MatSynth",
        streaming=True,
    )

    # remove unwanted columns
    # ds = ds.remove_columns(["diffuse", "specular", "displacement", "opacity", "blend_mask"])
    # or keep only specified columns
    ds = ds.select_columns(mat_keys)
    # shuffle data
    ds = ds.shuffle(buffer_size=1)

    # filter data matching a specific criteria, e.g.: only CC0 materials
    # ds = ds.filter(lambda x: x["metadata"]["license"] == "CC0")
    # filter out data from Deschaintre et al. 2018
    # ds = ds.filter(lambda x: x["metadata"]["source"] != "deschaintre_2020")

    # save files for a single material
    for i, x in enumerate(ds['train']):
        save_dir = Path(f"{base_output_dir}{x['name']}")
        save_dir.mkdir(parents=True, exist_ok=True)
        for k in mat_keys:
            if k == "name":
                continue
            x[k].resize((512,512)).save(save_dir / f'{k}.png')  # TODO material resolution, default seems to be 4k, which results in 1GB .glb file per object
        break
    return str(save_dir.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="create shapes")
    parser.add_argument('--output_dir', default='outputs', help='output directory')
    parser.add_argument('--num_shapes', default=1, type=int, help='number of shapes to create')
    parser.add_argument('--dont_convert_to_glb', default=False, action='store_true', help='converts the generated objs to glbs')
    parser.add_argument('--uuid_str', default='', type=str, help='uuid to use for the shape')
    parser.add_argument('--seed', default=0, type=int, help='seed for random number generation')
    parser.add_argument('--smooth_probability', default=1.0, type=float, help='possibility of smoothing the height field')
    parser.add_argument('--sub_obj_num_poss', type=str, default='5,5,5,4,4,3,2,1,1', help='comma separated list of possibilities for number of sub objects')
    parser.add_argument('--no_hf', default=False, action='store_true', help='do not use height field')

    args = parser.parse_args()
    args.sub_obj_num_poss = [int(x) for x in args.sub_obj_num_poss.split(',')]
    seed_everything(args.seed)

    start_time = time.time()
    out_dir = args.output_dir
    num_shapes = args.num_shapes

    mat_path = get_matsynth_material(out_dir)

    output_paths, shapes_parameters = createVarObjShapes(out_dir, range(num_shapes), uuid_str=args.uuid_str, bMultiObj=False, bPermuteMat=True, bScaleMesh=True, bMaxDimRange=[0.3, 0.45], smooth_probability=args.smooth_probability, sub_obj_nums=list(range(1, len(args.sub_obj_num_poss)+1)), sub_obj_num_poss=args.sub_obj_num_poss, no_hf=args.no_hf)
    shape_generation_time = time.time()
    print('Saved shapes to', out_dir)

    if args.dont_convert_to_glb:
        pass
    else:
        if len(output_paths) == 1:
            json_output_fn = str(output_paths[0]).replace('object.obj', f'{args.uuid_str.split("/")[-1]}_original_parameters.json')
            with open(json_output_fn, 'w') as f:
                json.dump(shapes_parameters[0], f, indent=4, cls=NpEncoder)
            print(f'Saved {json_output_fn}')
            convert_file(output_paths[0])
            convert_time = time.time()
        else:
            raise NotImplementedError('Converting multiple files to glb is not yet supported')
        print(f'TIME - create_shapes.py: shape_generation_time: {shape_generation_time - start_time:.2f}s, convert_time: {convert_time - shape_generation_time:.2f}s')


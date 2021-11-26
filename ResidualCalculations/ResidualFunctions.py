# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 20:22:00 2021

@author: Luca Hategan
"""
import numpy as np
import math
import pandas as pd
import time
from scipy.fft import fft, ifft, dct, idct
import matplotlib.pyplot as mp

def renameCols(InputFileList, BodyParts):
    ColsToUse = [Str for StrInit in BodyParts for Str in [StrInit]*2]
    Reset2ColNames = [[] for _ in range(len(InputFileList))]
    for ColNames in ColsToUse:
        if ColNames in ColsToUse and ColNames + "_x" not in ColsToUse:
            ColsToUse[ColsToUse.index(ColNames)] = ColNames + "_x"
        else:
            ColsToUse[ColsToUse.index(ColNames)] = ColNames + "_y"
    for Ind, Files in enumerate(InputFileList):
        for Frames in Files:
            ColsToDrop = [Cols for Cols in Frames.columns.values if Cols % 3 == 0]
            Frames = Frames.drop(ColsToDrop, axis = 1)
            Frames = Frames.rename(columns={Cols:ColsToUse[Ind] for Cols, Ind in zip(Frames.columns.values, range(len(ColsToUse)))})
            Reset2ColNames[Ind].append(Frames)
    return(Reset2ColNames)


def circlingBehaviour(VectorList, CriticalAngle):
    RotationalHashMap = {"Angle":0, "CCWAngle":0, "CW":0, "CCW":0, "PartialCW":0, "PartialCCW":0}
    AngleVector = []
    CCWAngleVector = []
    for Vec1, Vec2 in zip(VectorList[:-1], VectorList[1:]):
        if Vec1 != Vec2:
            theta = math.degrees(math.acos((np.dot(Vec1, Vec2))/((np.linalg.norm(Vec1))*(np.linalg.norm(Vec2)))))
            print(theta)
            time.sleep(0.2)
        else:
            theta = 0.0
        #Lagrange Identity for CW/CCW
        Lagrange = np.linalg.norm(Vec1)*np.linalg.norm(Vec2)*np.sin(theta)
        #CrossProduct for CW/CCW
        CrossProduct = np.cross(Vec1, Vec2)
        if (Lagrange >= 0):
            RotationalHashMap["Angle"] += theta
            AngleVector.append(RotationalHashMap["Angle"])
            CCWAngleVector.append(0)
            if RotationalHashMap["Angle"] >= CriticalAngle:
                RotationalHashMap["Angle"] = 0
                RotationalHashMap["CW"] += 1
        else:
            RotationalHashMap["CCWAngle"] += theta
            CCWAngleVector.append(RotationalHashMap["CCWAngle"])
            AngleVector.append(RotationalHashMap["Angle"])
            if RotationalHashMap["CCWAngle"] >= CriticalAngle:
                RotationalHashMap["CCWAngle"] = 0
                RotationalHashMap["CCW"] += 1
    return(RotationalHashMap, AngleVector, CCWAngleVector)

    # mp.plot([i[0] for i in VectorList[27000:28000]], [j[1] for j in VectorList[27000:28000]])
    # mp.show()

    # print(RotationalHashMap)
    # mp.plot(np.array(AngleVector[27000:28000]))
    # mp.xlabel("Frame Index")
    # mp.ylabel("Angle-theta, Degrees")
    # mp.title("Consecutive frame Midpoint-Head vector angles")
    # mp.show()

    # mp.plot(np.array(CCWAngleVector))
    # mp.xlabel("Frame Index")
    # mp.ylabel("Angle-theta, Degrees")
    # mp.title("Consecutive frame Midpoint-Head vector angles")
    # mp.show()

    # breakpoint()
    # FFT = fft(np.array(np.cos(AngleVector[26500:28000])))
    # mp.plot(FFT)
    # mp.xlabel("Frame Index")
    # mp.ylabel("fourier-transformed, frequency")
    # mp.title("fourier-transform of cicling data")
    # mp.show()



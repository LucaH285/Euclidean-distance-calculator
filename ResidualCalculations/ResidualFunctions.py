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
import numpy.fft as npfft

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

def aggregateCircling(Midpoints, VectorList, MaxY, MaxX):
    RotationalHashMap = {"Angle":0, "CWAngle":0, "CCWAngle":0, "CW":0, "CCW":0, "Total":0, "PartialCW":0, "PartialCCW":0}
    AngleVector = []
    for Vec1, Vec2 in zip(VectorList[:-1], VectorList[1:]):
        if Vec1 != Vec2:
            theta = math.degrees(math.acos((np.dot(Vec1, Vec2))/((np.linalg.norm(Vec1))*(np.linalg.norm(Vec2)))))
        else:
            theta = 0.0 
        # ClockwiseRotationMatrix = [[np.cos(theta), np.sin(theta)], 
        #                            [-1*np.sin(theta), np.cos(theta)]]    
        # CounterClockwiseRotationMatrix = [[np.cos(theta), -1*np.sin(theta)], 
        #                                   [np.sin(theta), np.cos(theta)]] 
        #print(np.cross(np.dot(np.array(ClockwiseRotationMatrix), Vec1), Vec2))
        # time.sleep(0.1)
        if RotationalHashMap["Angle"] < 358:
            RotationalHashMap["Angle"] += theta
            RotationalHashMap["PartialCW"] += theta/360
            AngleVector.append(RotationalHashMap["Angle"])
        elif RotationalHashMap["Angle"] >= 358:
            RotationalHashMap["Angle"] = 0
            RotationalHashMap["PartialCW"] = 0
            RotationalHashMap["Total"] += 1
            
    print(RotationalHashMap)
    
    mp.plot(np.array(AngleVector[10000:12000]))
    mp.xlabel("Frame Index")
    mp.ylabel("Angle-theta, Degrees")
    mp.title("Consecutive frame Midpoint-Head vector angles")
    mp.show()
    breakpoint()



def circlingBehaviour(VectorList, CriticalAngle):
    RotationalHashMap = {"Angle":0, "CCWAngle":0, "CW":0, "CCW":0, "PartialCW":0, "PartialCCW":0}
    AngleVector = []
    CCWAngleVector = []
    for Vec1, Vec2 in zip(VectorList[:-1], VectorList[1:]):
        if Vec1 != Vec2:
            theta = math.degrees(math.acos((np.dot(Vec1, Vec2))/((np.linalg.norm(Vec1))*(np.linalg.norm(Vec2)))))
        else:
            theta = 0.0
        #Lagrange Identity for CW/CCW
        Lagrange = np.linalg.norm(Vec1)*np.linalg.norm(Vec2)*np.sin(theta)
        #CrossProduct for CW/CCW
        CrossProduct = np.cross(Vec1, Vec2)
        if (CrossProduct >= 0):
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


def circlingBehavior2(VectorList, CriticalAngle):
    RotationalHashMap = {"Angle":0, "CCWAngle":0, "CW":0, "CCW":0, "PartialCW":0, "PartialCCW":0}
    AngleVector = []
    CCWAngleVector = []
    for Vectors in VectorList:
        if Vectors[0] != Vectors[1]:
            theta = math.degrees(math.acos((np.dot(Vectors[0], Vectors[1]))/((np.linalg.norm(Vectors[0]))*(np.linalg.norm(Vectors[1])))))
        else:
            theta = 0.0
        #Lagrange Identity for CW/CCW
        Lagrange = np.linalg.norm(Vectors[0])*np.linalg.norm(Vectors[1])*np.sin(theta)
        #CrossProduct for CW/CCW
        CrossProduct = np.cross(Vectors[0], Vectors[1])
        if (CrossProduct >= 0):
            RotationalHashMap["Angle"] += theta
            AngleVector.append(RotationalHashMap["Angle"])
            CCWAngleVector.append(RotationalHashMap["CCWAngle"])
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
    print(RotationalHashMap)
    
    mp.plot(np.array(AngleVector[10000:12000]))
    mp.xlabel("Frame Index")
    mp.ylabel("Angle-theta, Degrees")
    mp.title("Consecutive frame Midpoint-Head vector angles")
    mp.show()

    mp.plot(np.array(CCWAngleVector[27000:28000]))
    mp.xlabel("Frame Index")
    mp.ylabel("Angle-theta, Degrees")
    mp.title("Consecutive frame Midpoint-Head vector angles")
    mp.show()

    breakpoint()
    FFT = fft(np.array(np.cos(AngleVector[26500:28000])))
    mp.plot(FFT)
    mp.xlabel("Frame Index")
    mp.ylabel("fourier-transformed, frequency")
    mp.title("fourier-transform of cicling data")
    mp.show()
    
    
    return(RotationalHashMap, AngleVector, CCWAngleVector)

def circlingBehaviour3(Midpoints, VectorList, MaxY, MaxX, CriticalAngle):
    VectorFunction_YVal = lambda midpoints, Val: [[0, Val - Vectors[1]] for Vectors in midpoints]
    VectorFunction_XVal = lambda midpoints, Val: [[Val - Vectors[0], 0] for Vectors in midpoints]
    MaxYVectors = VectorFunction_YVal(midpoints=Midpoints, Val=MaxY)
    MinYVectors = VectorFunction_YVal(midpoints=Midpoints, Val=0)
    MaxXVectors = VectorFunction_XVal(midpoints=Midpoints, Val=MaxX)
    MinXVectors = VectorFunction_XVal(midpoints=Midpoints, Val=0)
    RotationalHashMap = {"Angle":0, "CCWAngle":0, "CW":0, "CCW":0, "PartialCW":0, "PartialCCW":0}
    
    AngleList = []
    for Vectors, MaxVec in zip(VectorList, MaxYVectors):
        Cross = np.cross(MaxVec, Vectors)
        theta = math.degrees(math.acos((np.dot(Vectors, MaxVec))/((np.linalg.norm(Vectors))*(np.linalg.norm(MaxVec)))))
        if Cross >= 0:
            AngleList.append(360 - theta)
        elif Cross < 0:
            AngleList.append(theta)
    IndexOfRotation = []
    for Angles in AngleList:
        if Angles >= CriticalAngle:
            RotationalHashMap["CW"] += 1
            IndexOfRotation.append(AngleList.index(Angles))
 
    for Theta1, Theta2 in zip(AngleList[:-1], AngleList[1:]):
        if ((Theta1 > Theta2)):
            RotationalHashMap["CCWAngle"] += (Theta1 - Theta2)
            if RotationalHashMap["CCWAngle"] >= CriticalAngle:
                RotationalHashMap["CCWAngle"] = 0
                RotationalHashMap["CCW"] += 1
        else:
            RotationalHashMap["CCWAngle"] = 0
         
    print(RotationalHashMap)
    Angles = [np.sin(math.radians(i)) for i in AngleList[0:5000]]
    Direction_X = [i[0] for i in VectorList[0:5000]]
    Direction_Y = [i[0] for i in VectorList[0:5000]]
    NormalizedDirection_X = [((2*((j-min(Direction_X))/(max(Direction_X) - min(Direction_X)))) - 1) for j in Direction_X]
    NormalizedDirection_Y = [((2*((j-min(Direction_Y))/(max(Direction_Y) - min(Direction_Y)))) - 1) for j in Direction_Y]
    mp.plot(Angles)
    mp.plot(NormalizedDirection_X, color="red")
    mp.plot(NormalizedDirection_Y, color = "black")
    mp.xlabel("Frame Index")
    mp.ylabel("Angle-theta, sin(radians)")
    mp.title("Consecutive frame Midpoint-Head vector & North-vector angles")
    mp.show()
    
    coslist = [np.sin(math.radians(1440 - i)) for i in range(0, 1440)]
    mp.plot(coslist)
    mp.show()

    mp.plot(np.abs(npfft.rfft(Angles)[0:100]))
    mp.show()
    breakpoint() 
        

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



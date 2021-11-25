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


def circlingBehaviour(MidLabelVectors):
    Condition = True
    Angle = 0
    Rotations = 0
    AngleVector = []
    thetaVectors = []
    for Vec1, Vec2 in zip(MidLabelVectors[:-1], MidLabelVectors[1:]):
        try:
            theta = math.degrees(math.acos((np.dot(Vec1, Vec2))/((np.linalg.norm(Vec1))*(np.linalg.norm(Vec2)))))
            thetaVectors.append(theta)
        except ValueError:
            pass
        if Angle < 360:
            if (((Vec2[0] - Vec1[0]) > 0) and (Vec2[1] - Vec1[1] < 0)) 
                or (((Vec2[0] - Vec1[0]) < 0) and (Vec2[1] - Vec1[1] < 0))
                or (((Vec2[0] - Vec1[0]) < 0) and (Vec2[1] - Vec1[1] < 0))
                Angle += theta
                AngleVector.append(Angle)
        elif Angle >= 358:
            Angle = 0
            Rotations += 1
    mp.plot(np.array(AngleVector[26500:28000]))
    mp.xlabel("Frame Index")
    mp.ylabel("Angle-theta, Degrees")
    mp.title("Consecutive frame Midpoint-Head vector angles")
    mp.show()
    FFT = fft(np.array(AngleVector[27000:28000]))
    mp.plot(FFT)
    mp.xlabel("Frame Index")
    mp.ylabel("fourier-transformed, frequency")
    mp.title("fourier-transform of cicling data")
    mp.show()
    print(Rotations)
        
def circlingBehaviour2(MidPoints, MidLabelVectors, MaxY):
    MaxYVectors_Fxn = lambda midpoints, maxY: [[0, maxY - Vectors[1]] for Vectors in midpoints]
    MinYVectors_Fxn = lambda midpoints, minY: [[0, minY - Vectors[1]] for Vectors in midpoints]
    MaxYVectors = MaxYVectors_Fxn(MidPoints, maxY=MaxY)
    MinYVectors = MinYVectors_Fxn(MidPoints, minY = 0)
    Angle = 0
    AngleVector = []
    thetavector = []
    Rotations = 0
    for consVector, sampleVector in zip(MaxYVectors, MidLabelVectors):
        theta = math.degrees(np.arccos((np.dot(consVector, sampleVector))/(np.linalg.norm(consVector)*np.linalg.norm(sampleVector))))
        thetavector.append(theta)
        if sampleVector[0] > 0:
            Angle = theta
            AngleVector.append(Angle)
        elif sampleVector[0] <= 0:
            Angle = 360 - theta
            AngleVector.append(Angle)
            if Angle >= 357:
                Angle = 0
                Rotations += 1
    print(max(AngleVector))
    mp.plot(np.array(AngleVector[0:1000]))
    mp.show()
    print(Rotations)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
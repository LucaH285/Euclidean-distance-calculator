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
import cv2

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

# def matrixSkeleton(Labels_To = [], Label_From = "", InputFileList):
    
def roationQuantifier_(PositionVecX, PositionVecY, MaxY, MaxX, CriticalAngle):
    DirectionVectorsFxn = lambda PositionVec1, PositionVec2: [[V2 - V1  for V1, V2 in zip(Vectors1, Vectors2)] for Vectors1, Vectors2 in zip(PositionVec1, PositionVec2)]
    MaxVectorFunction = lambda StartPosition, EndPosition: [[EndVecs[0] - StartVecs[0], EndVecs[1] - StartVecs[1]] for StartVecs, EndVecs in zip(StartPosition, EndPosition)]
    DirectionVectors = DirectionVectorsFxn(PositionVecX, PositionVecY)
    RotationalHashMap = {"CWAngle":0, "CCWAngle":0, "CW":0, "CCW":0.000, "PartialCW":0, "PartialCCW":0}
    
    plotMaxVec_YMax = [[DVectors[0], 0]for DVectors in PositionVecX]
    plotMaxVec_YMin = [[DVectors[0], MaxY]for DVectors in PositionVecX]
    plotMaxVec_XMax = [[MaxX, DVectors[1]]for DVectors in PositionVecX]
    plotMaxVec_XMin = [[0, DVectors[1]]for DVectors in PositionVecX]
    MaxVectors = MaxVectorFunction(PositionVecX, plotMaxVec_YMax)

    AngleList = []
    RotationalMotionCW = []
    RotationalMotionCCW = []
    for V1, V2 in zip(MaxVectors, DirectionVectors):
        #Overcome the cross product by crossing the body-head vectors with the maximum vector
        Cross = np.cross(V1, V2)
        Theta = math.degrees(math.acos((np.dot(V2, V1))/((np.linalg.norm(V2))*(np.linalg.norm(V1)))))
        if Cross > 0:
            AngleList.append(Theta)
        elif Cross < 0:
            Phi = 360 - Theta
            AngleList.append(Phi)
            
    InitialCondition = True
    ReferenceAngle = 0
    # for Theta1, Theta2 in zip(AngleList[:-1], AngleList[1:]):
    #     if ((Theta1 == Theta2) and (InitialCondition == True)):
    #         pass
    #     elif ((Theta1 != Theta2) and (InitialCondition == True)):
    #         InitialCondition == False
    #         ReferenceAngle = Theta1/CriticalAngle
    #     else:
    #         if ((Theta1/CriticalAngle >= ReferenceAngle) and (Theta2/CriticalAngle < CriticalAngle/360)
    #             and (np.cross(DirectionVectors[AngleList.index(Theta1)], DirectionVectors[AngleList.index(Theta2)]) > 0):
                
                
            
                
        

          
    return(RotationalMotionCW, RotationalMotionCCW, DirectionVectors, plotMaxVec_YMax,
           plotMaxVec_YMin, plotMaxVec_XMax, plotMaxVec_XMin)
    
    

def rotationQuantifier(PositionVecX, PositionVecY, MaxY, MaxX, CriticalAngle):
    DirectionVectorsFxn = lambda PositionVec1, PositionVec2: [[V2 - V1  for V1, V2 in zip(Vectors1, Vectors2)] for Vectors1, Vectors2 in zip(PositionVec1, PositionVec2)]
    MaxVectorFunction = lambda StartPosition, EndPosition: [[EndVecs[0] - StartVecs[0], EndVecs[1] - StartVecs[1]] for StartVecs, EndVecs in zip(StartPosition, EndPosition)]
    DirectionVectors = DirectionVectorsFxn(PositionVecX, PositionVecY)
    RotationalHashMap = {"CWAngle":0, "CCWAngle":0, "CW":0, "CCW":0.000, "PartialCW":0, "PartialCCW":0}
    
    plotMaxVec_YMax = [[DVectors[0], 0]for DVectors in PositionVecX]
    plotMaxVec_YMin = [[DVectors[0], MaxY]for DVectors in PositionVecX]
    plotMaxVec_XMax = [[MaxX, DVectors[1]]for DVectors in PositionVecX]
    plotMaxVec_XMin = [[0, DVectors[1]]for DVectors in PositionVecX]
    MaxVectors = MaxVectorFunction(PositionVecX, plotMaxVec_YMax)

    AngleList = []
    RotationalMotionCW = []
    RotationalMotionCCW = []
    for V1, V2 in zip(MaxVectors, DirectionVectors):
        #Overcome the cross product by crossing the body-head vectors with the maximum vector
        Cross = np.cross(V1, V2)
        Theta = math.degrees(math.acos((np.dot(V2, V1))/((np.linalg.norm(V2))*(np.linalg.norm(V1)))))
        if Cross > 0:
            AngleList.append(Theta)
        elif Cross < 0:
            Phi = 360 - Theta
            AngleList.append(Phi)
                
    # Compute the CW and CCW rotations, respectively
    AngleIndex = 0
    AngleIndexCCW = 0
    FrameCount = 0
    Condition = True
    while(Condition):
        TemporaryAngleList = []
        for Theta1, Theta2 in zip(AngleList[:-1], AngleList[1:]):
            """
            Another way is to check if a full revolution has occured before counting
            a full cw/ccw rotation.
            """
            if ((Theta1/CriticalAngle >= CriticalAngle/360) 
                and (Theta2/CriticalAngle < 0.25) and (np.sum(TemporaryAngleList) >= CriticalAngle)
                #This argument controls the frame indecces to make sure that Frames are sufficiently 
                #distanced from each other so as to avoid counting counterclocwise then clockwise motion
                #that passes the critical angle (happens sometimes)
                and ((AngleList.index(Theta1) - AngleIndex) > 30)):
                RotationalHashMap["CW"] += 1
<<<<<<< HEAD
                RotationalMotionCW.append(RotationalHashMap["CW"])
=======
                if ((RotationalHashMap["CW"] - math.modf(RotationalMotionCW[-1])[1]) < 2):
                    RotationalMotionCW.append(RotationalHashMap["CW"])
>>>>>>> parent of 527fc9c (update)
                AngleIndex = AngleList.index(Theta1)
                TemporaryAngleList.clear()
            else:
                if ((Theta2 - Theta1) > 0):
                    TemporaryAngleList.append((Theta2 - Theta1))
                RotationalMotionCW.append(RotationalHashMap["CW"] + Theta1/CriticalAngle) 
<<<<<<< HEAD
                    
            if (((Theta2/CriticalAngle < Theta1/CriticalAngle) or (CriticalAngle < Theta2 <= 360 and 0 <= Theta1 < 90)) 
                  and (np.cross(DirectionVectors[AngleList.index(Theta1)], DirectionVectors[AngleList.index(Theta2)]) < 0)):
                if ((CriticalAngle < Theta2 <= 360 and 0 <= Theta1 < 90) and (RotationalHashMap["CCWAngle"] < CriticalAngle)):
                    CCW_CrossVector = True
                else:
                    CCW_CrossVector = False
                    CCWRotation = (Theta1 - Theta2)/CriticalAngle
                    RotationalHashMap["CCWAngle"] += CCWRotation
                    if RotationalHashMap["CCWAngle"] >= CriticalAngle/360:
                        RotationalHashMap["CCW"] += 1
                        RotationalHashMap["CCWAngle"] = 0
                        FrameCount = 0
=======
                
            if ((Theta2/CriticalAngle < Theta1/CriticalAngle) and ((Theta1 - Theta2) < CriticalAngle)
                  and (np.cross(DirectionVectors[AngleList.index(Theta1)], DirectionVectors[AngleList.index(Theta2)]) < 0)):
                CCWRotation = (Theta1 - Theta2)/CriticalAngle
                RotationalHashMap["CCWAngle"] += CCWRotation
                if RotationalHashMap["CCWAngle"] >= CriticalAngle/360:
                    RotationalHashMap["CCW"] += 1
                    RotationalHashMap["CCWAngle"] = 0
                    FrameCount = 0
>>>>>>> parent of 527fc9c (update)
                RotationalMotionCCW.append(RotationalHashMap["CCW"] + RotationalHashMap["CCWAngle"])
                FrameCount += 1
            else:
                if FrameCount < 5:
                    RotationalHashMap["CCWAngle"] = 0.000
                    FrameCount = 0
                FrameCount -= 1
                RotationalMotionCCW.append(RotationalHashMap["CCW"] + RotationalHashMap["CCWAngle"])
        """
        Rounds the rotations up to the nearest whole integer if > 90% of the rotation has been made in either direction.
        """
        RoundLastCW = math.modf(RotationalMotionCW[-1])[0]
        RoundLastCCW = math.modf(RotationalMotionCCW[-1])[0]
        if ((RoundLastCW > 0.9) or (RoundLastCCW > 0.9)):
            if RoundLastCW > 0.9:
                RotationalHashMap["CW"] += 1
                RotationalMotionCW.append(RotationalHashMap["CW"])
                Condition = False
            elif RoundLastCCW > 0.9:
                RotationalHashMap["CCW"] += 1
                RotationalMotionCCW.append(RotationalHashMap["CCW"])
                Condition = False
        else:
            Condition = False
<<<<<<< HEAD
=======
    
>>>>>>> parent of 527fc9c (update)
    print(RotationalHashMap)
    return(RotationalMotionCW, RotationalMotionCCW, DirectionVectors, plotMaxVec_YMax,
           plotMaxVec_YMin, plotMaxVec_XMax, plotMaxVec_XMin)

def rotationQuantifier2(PositionVecX, PositionVecY, MaxY, MaxX, CriticalAngle):
    MidpointOfLine_Fxn = lambda PosVec1, PosVec2:[[(V1 + V2)/2 for V1, V2 in zip(Vectors1, Vectors2)] for Vectors1, Vectors2 in zip(PosVec1, PosVec2)]
    MidPointToLabel_Fxn = lambda Midpoints, PosVec: [[V2 - V1 for V1, V2 in zip(Vectors1, Vectors2)] for Vectors1, Vectors2 in zip(Midpoints, PosVec)]
    MidPointOfLine = MidpointOfLine_Fxn(PositionVecX, PositionVecY)
    MidPointToLabel_Vector = MidPointToLabel_Fxn(MidPointOfLine, PositionVecY)
    
    
    pass

def TrackOnVideo(Annotations, videoFile, PositionVectorsX, PositionVectorsY, VideoOut,
                 skeleton = []):
    cap = cv2.VideoCapture(videoFile)
    # current_state = False
    # annotation_list = Annotations
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(VideoOut, fourcc, 30.0, (1920, 1080))
    
    def __draw_label(img, text, pos, bg_color):
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        scale = 2
        color = (0, 0, 0)
        thickness = cv2.FILLED
        margin = 5
    
        txt_size = cv2.getTextSize(str(text), font_face, scale, thickness)
    
        end_x = pos[0] + txt_size[0][0] + margin
        end_y = pos[1] - txt_size[0][1] - margin
    
        cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
        cv2.putText(img, str(text), pos, font_face, scale, color, 1, cv2.LINE_AA)
        
    i = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            try:
                __draw_label(frame, f"CW: {round(Annotations[0][i], 3)}", (20, 200), (255,255,255))
                cv2.line(frame, (int(PositionVectorsX[i][0]), int(PositionVectorsX[i][1])), (int(PositionVectorsY[i][0]), int(PositionVectorsY[i][1])), (0, 255, 0), 5, 8, 0)
                __draw_label(frame, f"CCW: {round(Annotations[1][i], 3)}", (20, 250), (0,0,255))
                cv2.line(frame, (int(PositionVectorsX[i][0]), int(PositionVectorsX[i][1])), (int(Annotations[3][i][0]), int(Annotations[3][i][1])), (0, 0, 255), 5, 8, 0)
                cv2.line(frame, (int(PositionVectorsX[i][0]), int(PositionVectorsX[i][1])), (int(Annotations[4][i][0]), int(Annotations[4][i][1])), (255, 0, 0), 5, 8, 0)
                cv2.line(frame, (int(PositionVectorsX[i][0]), int(PositionVectorsX[i][1])), (int(Annotations[5][i][0]), int(Annotations[5][i][1])), (50, 100, 150), 5, 8, 0)
                cv2.line(frame, (int(PositionVectorsX[i][0]), int(PositionVectorsX[i][1])), (int(Annotations[6][i][0]), int(Annotations[6][i][1])), (150, 100, 50), 5, 8, 0)
            except IndexError:
                pass
            out.write(frame)
            cv2.imshow("Frame", frame)
            i += 1 
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            elif cv2.waitKey(1) & 0xFF == ord('p'):
                time.sleep(3)
                
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    



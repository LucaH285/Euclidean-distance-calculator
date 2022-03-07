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

def rotationQuantifier(PositionVecX, PositionVecY, MaxY, MaxX, CriticalAngle, FPS, RecordTime_sec):
    """
    Function controls the clockwise and counterclockwise counting of input videos

    Parameters
    ----------
    Data frames as inputs
    
    Takes the three columns that are associated with a label (X, Y, p-val), handled in the while loop
    changes the 

    Returns
    -------
    The function returns a list of these preprocessed frames.
    returns a list of body parts as well.
    """
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

    RotationList = [[] for _ in range(0, 4)]
    # Compute the CW and CCW rotations, respectively
    AngleIndex = 0
    AngleIndexCCW = 0
    FrameCount = 0
    Condition = True
    while(Condition):
        CCW_CrossVector = False
        #Theta1
        FrameInd = ((RecordTime_sec) * 30)
        for Theta1, Theta2, Ind in zip(AngleList[:-1], AngleList[1:], range(len(AngleList))):
            
            if Ind == FrameInd:
                Time = (FrameInd/FPS)/60
                CWRotations = RotationalHashMap["CW"]
                RotationList[0].append(round(Time, 2))
                RotationList[1].append(FrameInd)
                RotationList[2].append(CWRotations)
                FrameInd += (RecordTime_sec * 30) 
            elif Ind == (len(AngleList) - 2):
                Time = (Ind/FPS)/60
                CWRotations = RotationalHashMap["CW"]
                RotationList[0].append(round(Time, 2))
                RotationList[1].append(Ind)
                RotationList[2].append(CWRotations)
                         
            """
            Another way is to check if a full revolution has occured before counting
            a full cw/ccw rotation.
            """
            if ((Theta1/CriticalAngle >= CriticalAngle/360)
                and (Theta2/CriticalAngle < 0.25) and (CCW_CrossVector is False)
                and (np.cross(DirectionVectors[AngleList.index(Theta1)], DirectionVectors[AngleList.index(Theta2)]) > 0)
                #This argument controls the frame indecces to make sure that Frames are sufficiently 
                #distanced from each other so as to avoid counting counterclocwise then clockwise motion
                #that passes the critical angle (happens sometimes)
                and ((AngleList.index(Theta1) - AngleIndex) > 45)):
                RotationalHashMap["CW"] += 1
                RotationalMotionCW.append(RotationalHashMap["CW"])
                AngleIndex = AngleList.index(Theta1)
            #Resets the CCW_CrossVector when a new rotation is initiated, i.e.: when the rat crosses the central vector
            #in the clockwise direction without completing a full rotation.
            elif ((Theta1/CriticalAngle >= CriticalAngle/360)
                and (Theta2/CriticalAngle < 0.25) and (CCW_CrossVector is True)): #and ((AngleList.index(Theta1) - AngleIndex) > 30)):
                CCW_CrossVector = False
                RotationalMotionCW.append(RotationalHashMap["CW"] + Theta1/CriticalAngle)
            else:
                RotationalMotionCW.append(RotationalHashMap["CW"] + Theta1/CriticalAngle)
                
            if (((Theta2/CriticalAngle < Theta1/CriticalAngle) or (CriticalAngle < Theta2 <= 360 and 0 <= Theta1 < 90)) 
                  and (np.cross(DirectionVectors[AngleList.index(Theta1)], DirectionVectors[AngleList.index(Theta2)]) < 0)):
                if (CriticalAngle < Theta2 <= 360 and 0 <= Theta1 < 90):
                    CCW_CrossVector = True
                    AngleIndex = AngleList.index(Theta1)
            
        FrameInd = (RecordTime_sec * 30)                           
        CW_CrossVector = False
        for Theta1, Theta2, Ind in zip(AngleList[:-1], AngleList[1:], range(len(AngleList))):
            
            if Ind == FrameInd:
                CCWRotations = RotationalHashMap["CCW"]
                RotationList[3].append(CCWRotations)
                FrameInd += (RecordTime_sec * 30) 
            elif Ind == (len(AngleList) - 2):
                CCWRotations = RotationalHashMap["CCW"]
                RotationList[3].append(CCWRotations)
            
            #If the second angle in the list passes the 360 point first and 
            if ((Theta2/CriticalAngle >= CriticalAngle/360) and (Theta1/CriticalAngle < 0.25) and (CW_CrossVector is False)
                and ((AngleList.index(Theta2) - AngleIndexCCW) > 45) 
                and (np.cross(DirectionVectors[AngleList.index(Theta1)], DirectionVectors[AngleList.index(Theta2)]) < 0)): 
                RotationalHashMap["CCW"] += 1
                RotationalMotionCCW.append(RotationalHashMap["CCW"])
                #Count the AngleIndex when only when it crosses in the counter clockwise direction
                AngleIndexCCW = AngleList.index(Theta2)
                
                
            elif ((Theta2/CriticalAngle >= CriticalAngle/360) and (Theta1/CriticalAngle < 0.25) and (CW_CrossVector is True)
                and ((AngleList.index(Theta2) - AngleIndexCCW) < 45)):
                AngleIndexCCW = AngleList.index(Theta2)
                CW_CrossVector = False
                RotationalMotionCCW.append(RotationalHashMap["CCW"] + (1 - (Theta2/CriticalAngle)))
            else:
                RotationalMotionCCW.append(RotationalHashMap["CCW"] + (1 - (Theta2/CriticalAngle)))
                
            #Argument should set CW_CrossVector if the animal crosses the central axis (pi/2) in the clockwise direction
            #If Theta1 > 0 and less than 90, if Theta2
            if ((CriticalAngle < Theta1 <= 360 and 0 <= Theta2 < 90)
                and (np.cross(DirectionVectors[AngleList.index(Theta1)], DirectionVectors[AngleList.index(Theta2)]) > 0)):
                if (CriticalAngle < Theta1 <= 360 and 0 <= Theta2 < 90):
                    CW_CrossVector = True
                    AngleIndexCCW = AngleList.index(Theta2)

                    # print(Theta1, Theta2, AngleList.index(Theta1), AngleList.index(Theta2))
                    # print(AngleIndexCCW)
        """
        Rounds the rotations up to the nearest whole integer if > 90% of the rotation has been made in either direction.
        """
        RoundLastCW, RoundLastCWInt = math.modf(RotationalMotionCW[-1])[0], math.modf(RotationalMotionCW[-1])[1]
        RoundLastCCW, RoundLastCCWInt = math.modf(RotationalMotionCCW[-1])[0], math.modf(RotationalMotionCCW[-1])[1]
        if (((RoundLastCW > 0.9) or (RoundLastCW < 0.9 and RoundLastCWInt > RotationalHashMap["CW"])) 
            or ((RoundLastCCW > 0.9) or (RoundLastCCW < 0.9 and RoundLastCCWInt > RotationalHashMap["CCW"]))):
            if (RoundLastCW > 0.9) or (RoundLastCW < 0.9 and RoundLastCWInt > RotationalHashMap["CW"]):
                RotationalHashMap["CW"] += 1
                RotationalMotionCW.append(RotationalHashMap["CW"])
                RotationList[2] = RotationalHashMap["CW"]
                Condition = False
            elif (RoundLastCCW > 0.9) or (RoundLastCCW < 0.9 and RoundLastCCWInt > RotationalHashMap["CCW"]):
                RotationalHashMap["CCW"] += 1
                RotationalMotionCCW.append(RotationalHashMap["CCW"])
                RotationList[3] = RotationalHashMap["CCW"]
                Condition = False
        else:
            Condition = False
    print(RotationalHashMap, FrameCount)
    Columns = ["Elapsed Time (min)", "Frame Count", "Clockwise Rotations", "Counterclockwise Rotations"]
    RotationListFrame = pd.DataFrame({Columns[Ind]:RotationList[Ind] for Ind in range(len(Columns))})
    
    # breakpoint()
    RotationListFrame.to_csv(r'F:\WorkFiles_XCELLeration\Video\DifferentApproaches\PGram.csv')
    
    return(RotationalMotionCW, RotationalMotionCCW, DirectionVectors, plotMaxVec_YMax,
           plotMaxVec_YMin, plotMaxVec_XMax, plotMaxVec_XMin)

def rotationQuantifier2(PositionVecX, PositionVecY, MaxY, MaxX, CriticalAngle):
    MidpointOfLine_Fxn = lambda PosVec1, PosVec2:[[(V1 + V2)/2 for V1, V2 in zip(Vectors1, Vectors2)] for Vectors1, Vectors2 in zip(PosVec1, PosVec2)]
    MidPointToLabel_Fxn = lambda Midpoints, PosVec: [[V2 - V1 for V1, V2 in zip(Vectors1, Vectors2)] for Vectors1, Vectors2 in zip(Midpoints, PosVec)]
   # MaxVectorFunction = lambda 
    MidPointOfLine = MidpointOfLine_Fxn(PositionVecX, PositionVecY)
    MidPointToLabel_Vector = MidPointToLabel_Fxn(MidPointOfLine, PositionVecY)
   # MaxVector = 

    pass

def TrackOnVideo(Annotations, videoFile, PositionVectorsX, PositionVectorsY, VideoOut, PVals):
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
                __draw_label(frame, f"{int(PositionVectorsY[i][0])}, {int(PositionVectorsY[i][1])}", (1500, 200), (100, 50, 255))
                __draw_label(frame, f"Frame: {i}", (1500, 300), (0,0,255))
                cv2.circle(frame, (int(PositionVectorsY[i][0]), int(PositionVectorsY[i][1])), radius = 0, color=(0, 0, 255), thickness=-1)
                if len(PVals) != 0:
                    if PVals[0][i] >= 4.0:
                        cv2.line(frame, (int(PositionVectorsX[i][0]), int(PositionVectorsX[i][1])), (int(PositionVectorsY[i][0]), int(PositionVectorsY[i][1])), (0, 0, 0), 5, 8, 0)
                    elif PVals[0][i] >= 3.0 and PVals[0][i] < 4.0:
                        cv2.line(frame, (int(PositionVectorsX[i][0]), int(PositionVectorsX[i][1])), (int(PositionVectorsY[i][0]), int(PositionVectorsY[i][1])), (0, 0, 255), 5, 8, 0)
                    #Left Ear
                    elif PVals[0][i] >= 2.0 and PVals[0][i] < 3.0:
                        cv2.line(frame, (int(PositionVectorsX[i][0]), int(PositionVectorsX[i][1])), (int(PositionVectorsY[i][0]), int(PositionVectorsY[i][1])), (255, 255, 255), 5, 8, 0)
                    #Right Ear
                    elif PVals[0][i] > 1.0 and PVals[0][i] < 2.0:
                        cv2.line(frame, (int(PositionVectorsX[i][0]), int(PositionVectorsX[i][1])), (int(PositionVectorsY[i][0]), int(PositionVectorsY[i][1])), (0, 200, 255), 5, 8, 0)
                
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
    



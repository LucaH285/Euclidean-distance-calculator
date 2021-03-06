# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 15:28:09 2021

@author: Luca Hategan
"""
import pandas as pd
from collections import OrderedDict
import numpy as np
from numpy import linalg as LA
import math
from scipy.integrate import quad
import time
import copy
import itertools


def preprocessor(DataFrame):
    """
    Function responsible for early preprocessing of the input data frames
        - creates a list of body parts labeled by the neural net
        - creates a trimmed frame, such that only relevant numerical data is included (i.e.: x, y coords and p-vals)

    Parameters
    ----------
    Data frames as inputs

    Returns
    -------
    The function returns a list of these preprocessed frames.
    returns a list of body parts as well.

    """
    ResetColNames = {
        DataFrame.columns.values[Ind]:Ind for Ind in range(len(DataFrame.columns.values))
        }
    ProcessedFrame = DataFrame.rename(columns=ResetColNames).drop([0], axis = 1)
    BodyParts = list(OrderedDict.fromkeys(list(ProcessedFrame.iloc[0,])))
    BodyParts = [Names for Names in BodyParts if Names != "bodyparts"]
    TrimmedFrame = ProcessedFrame.iloc[2:,]
    TrimmedFrame = TrimmedFrame.reset_index(drop=True)
    return(TrimmedFrame, BodyParts)

def checkPVals(DataFrame, CutOff):
    """
    Function responsible for processing p-values, namely omitting pvalues and their associated
    coordinates by forward filling the last valid observation as defined by the cutoff limit (user defined)

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
    #This loop assigns the first p-values in all label columns = 1, serve as reference point.
    for Cols in DataFrame.columns.values:
        if Cols % 3 == 0:
            if float(DataFrame[Cols][0]) < CutOff:
                DataFrame.loc[0, Cols] = 1.0
    Cols = 3
    #While loop iterating through every 3rd column (p-val column)
    #ffill = forward fill, propagates last valid observation forward. 
    #Values with p-val < cutoff masked.
    while Cols <= max(DataFrame.columns.values):
        Query = [i for i in range(Cols-2, Cols+1)]
        DataFrame[Query] = DataFrame[Query].mask(pd.to_numeric(DataFrame[Cols], downcast="float") < CutOff).ffill()
        Cols += 3
    return(DataFrame)

def predictLabelLocation(DataFrame, CutOff, LabelsFrom, colNames, PredictLabel):
    """
    Function responsible for processing p-values, namely omitting

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
    OldColumns = list(DataFrame.columns.values)
    FeatureList = ["_x", "_y", "_p-val"]
    NewCols = [f"{ColName}{Feature}" for ColName, Feature in itertools.product(colNames, FeatureList)]
    DataFrame = DataFrame.rename(columns={DataFrame.columns.values[Ind]:NewCols[Ind] for Ind in range(len(NewCols))})
    NewColumns = list(DataFrame.columns.values)
    for Cols in DataFrame.columns.values:
        DataFrame[Cols] = pd.to_numeric(DataFrame[Cols], downcast="float")
    #If the direction vector remains the same, there will be an over-estimation in the new position of the head if you continue to reuse
    #the same direction vector. Maybe scale the direction vector.
    ReferenceDirection = []
    ScaledVec = []
    BodyPart = []
    for Ind, PVals in enumerate(DataFrame[f"{PredictLabel}_p-val"]):
        if (PVals < CutOff):
            ##############
            #Choose surrounding label
            ##############
            AdjacentLabel = [Label for Label in LabelsFrom if DataFrame[f"{Label}_p-val"][Ind] >= CutOff]
            if ((len(AdjacentLabel) != 0) and (Ind != 0)):
                if (DataFrame[f"{PredictLabel}_p-val"][Ind - 1] >= CutOff):
                    ##########
                    #Create Direction Vectors between the first adjacent label available in the list
                    #Parallelogram law
                    ##########
                    DirectionVec = [DataFrame[f"{PredictLabel}_x"][Ind - 1] - DataFrame[f"{AdjacentLabel[0]}_x"][Ind - 1], 
                                    DataFrame[f"{PredictLabel}_y"][Ind - 1] - DataFrame[f"{AdjacentLabel[0]}_y"][Ind - 1]]
                    ReferenceDirection = DirectionVec
                elif ((DataFrame[f"{PredictLabel}_p-val"][Ind - 1] < CutOff) and (len(ReferenceDirection) == 0)):
                    ReferenceDirection = [0, 0]
                ###########
                #Compute the displacement between available surronding label
                #Compute the vector addition (parallelogram law) and scale the Ind - 1 first available adjacent label by it
                ###########
                Displacement = [DataFrame[f"{AdjacentLabel[0]}_x"][Ind] - DataFrame[f"{AdjacentLabel[0]}_x"][Ind - 1],
                                DataFrame[f"{AdjacentLabel[0]}_y"][Ind] - DataFrame[f"{AdjacentLabel[0]}_y"][Ind - 1]]
                Scale = np.add(ReferenceDirection, Displacement)
                DataFrame[f"{PredictLabel}_x"][Ind] = DataFrame[f"{AdjacentLabel[0]}_x"][Ind - 1] + Scale[0]
                DataFrame[f"{PredictLabel}_y"][Ind] = DataFrame[f"{AdjacentLabel[0]}_y"][Ind - 1] + Scale[1]
                DataFrame[f"{PredictLabel}_p-val"][Ind] = 4.5 
                Norm = LA.norm(Scale)
            # elif (len(AdjacentLabel) == 0):
            #     print(AdjacentLabel)

                
    # print(max(ScaledVec), min(ScaledVec), np.std(ScaledVec), np.average(ScaledVec))
    # print(BodyPart[ScaledVec.index(max(ScaledVec))])
    PVAL_PREDICTEDLABEL = list(DataFrame[f"{PredictLabel}_p-val"])
    DataFrame = DataFrame.rename(columns={NewColumns[Ind]: OldColumns[Ind] for Ind in range(len(OldColumns))})
    return(DataFrame, PVAL_PREDICTEDLABEL) 

def predictLabel_RotationMatrix(DataFrame, CutOff, LabelsFrom, colNames, PredictLabel):
    OldColumns = list(DataFrame.columns.values)
    FeatureList = ["_x", "_y", "_p-val"]
    NewCols = [f"{ColName}{Feature}" for ColName, Feature in itertools.product(colNames, FeatureList)]
    DataFrame = DataFrame.rename(columns={DataFrame.columns.values[Ind]:NewCols[Ind] for Ind in range(len(NewCols))})
    NewColumns = list(DataFrame.columns.values)
    for Cols in DataFrame.columns.values:
        DataFrame[Cols] = pd.to_numeric(DataFrame[Cols], downcast="float")
    ReferenceMid = []
    FactorDict = {"Angle_Right":0, "Angle_Left":0}
    VectorAngle = lambda V1, V2: math.acos((np.dot(V2, V1))/((np.linalg.norm(V2))*(np.linalg.norm(V1))))
    
    RotationMatrixCW = lambda Theta: np.array(
        [[math.cos(Theta), math.sin(Theta)], 
         [-1*math.sin(Theta), math.cos(Theta)]]
        )
    
    RotationMatrixCCW = lambda Theta: np.array(
        [[math.cos(Theta), -1*math.sin(Theta)], 
         [math.sin(Theta), math.cos(Theta)]]
        )
   
    for Ind, PVals in enumerate(DataFrame[f"{PredictLabel}_p-val"]):
        AdjacentLabel = [Label for Label in LabelsFrom if DataFrame[f"{Label}_p-val"][Ind] >= CutOff]
        #If the Head label is poorly track initiate this statement
        if (PVals < CutOff):  
            if ((DataFrame[f"{LabelsFrom[0]}_p-val"][Ind] >= CutOff) and (DataFrame[f"{LabelsFrom[1]}_p-val"][Ind] >= CutOff)):
               MidPoint = [(DataFrame[f"{LabelsFrom[0]}_x"][Ind] + DataFrame[f"{LabelsFrom[1]}_x"][Ind])/2,
                           (DataFrame[f"{LabelsFrom[0]}_y"][Ind] + DataFrame[f"{LabelsFrom[1]}_y"][Ind])/2]
               ReferenceMid = MidPoint
               
               DataFrame[f"{PredictLabel}_x"][Ind] = ReferenceMid[0]
               DataFrame[f"{PredictLabel}_y"][Ind] = ReferenceMid[1]
               
               DataFrame[f"{PredictLabel}_p-val"][Ind] = 3.5
                            
            elif (((DataFrame[f"{LabelsFrom[0]}_p-val"][Ind] or DataFrame[f"{LabelsFrom[1]}_p-val"][Ind]) < CutOff)
                  and (len(AdjacentLabel) != 0) and (DataFrame["Body_p-val"][Ind] >= CutOff)):

                #Right
                if ((DataFrame[f"{LabelsFrom[0]}_p-val"][Ind]) >= CutOff):
                     
                    
                    DVec_Right = [DataFrame[f"{LabelsFrom[0]}_x"][Ind] - DataFrame["Body_x"][Ind],
                             DataFrame[f"{LabelsFrom[0]}_y"][Ind] - DataFrame["Body_y"][Ind]]
                    ScaleRoation = np.dot(RotationMatrixCW(Theta=FactorDict["Angle_Right"]), DVec_Right) 
                    DataFrame[f"{PredictLabel}_x"][Ind] = ScaleRoation[0] + DataFrame["Body_x"][Ind]
                    DataFrame[f"{PredictLabel}_y"][Ind] = ScaleRoation[1] + DataFrame["Body_y"][Ind]
                    
                    DataFrame[f"{PredictLabel}_p-val"][Ind] = 4.5 
                    
               #Left 
                elif ((DataFrame[f"{LabelsFrom[1]}_p-val"][Ind]) >= CutOff):
                    
                    DVec_Left = [DataFrame[f"{LabelsFrom[1]}_x"][Ind] - DataFrame["Body_x"][Ind],
                             DataFrame[f"{LabelsFrom[1]}_y"][Ind] - DataFrame["Body_y"][Ind]]
                    ScaleRoation = np.dot(RotationMatrixCCW(Theta=FactorDict["Angle_Left"]), DVec_Left) 
                    DataFrame[f"{PredictLabel}_x"][Ind] = ScaleRoation[0] + DataFrame["Body_x"][Ind]
                    DataFrame[f"{PredictLabel}_y"][Ind] = ScaleRoation[1] + DataFrame["Body_y"][Ind]
                    
                    DataFrame[f"{PredictLabel}_p-val"][Ind] = 2.5

    PVAL_PREDICTEDLABEL = list(DataFrame[f"{PredictLabel}_p-val"])

    DataFrame = DataFrame.rename(columns={NewColumns[Ind]: OldColumns[Ind] for Ind in range(len(OldColumns))})
    return(DataFrame, PVAL_PREDICTEDLABEL)
    

def predictLabel_MidpointAdjacent(DataFrame, CutOff, LabelsFrom, colNames, PredictLabel):
    OldColumns = list(DataFrame.columns.values)
    FeatureList = ["_x", "_y", "_p-val"]
    NewCols = [f"{ColName}{Feature}" for ColName, Feature in itertools.product(colNames, FeatureList)]
    DataFrame = DataFrame.rename(columns={DataFrame.columns.values[Ind]:NewCols[Ind] for Ind in range(len(NewCols))})
    NewColumns = list(DataFrame.columns.values)
    for Cols in DataFrame.columns.values:
        DataFrame[Cols] = pd.to_numeric(DataFrame[Cols], downcast="float")
    ReferenceMid = []
    
    AngleDict = {"Angle_Right":0, "Angle_Left":0}
    VectorAngle = lambda V1, V2: math.acos((np.dot(V2, V1))/((np.linalg.norm(V2))*(np.linalg.norm(V1))))
    
    RotationMatrixCW = lambda Theta: np.array(
        [[np.cos(Theta), np.sin(Theta)], 
         [-1*np.sin(Theta), np.cos(Theta)]]
        )
    
    RotationMatrixCCW = lambda Theta: np.array(
        [[math.cos(Theta), -1*math.sin(Theta)], 
         [math.sin(Theta), math.cos(Theta)]]
        )
    
    # print(RotationMatrixCW(Theta = np.pi/2))
    # print(RotationMatrixCCW(Theta = np.pi/2))
    
    # breakpoint()
    
    AngleList_Right = []
    AngleList_Left = []
    for Ind, PVals in enumerate(DataFrame[f"{PredictLabel}_p-val"]):
        if ((PVals >= CutOff) and (DataFrame["Body_p-val"][Ind] >= CutOff)
            and (DataFrame[f"{LabelsFrom[0]}_p-val"][Ind] >= CutOff) and (DataFrame[f"{LabelsFrom[1]}_p-val"][Ind] >= CutOff)):
            DirectionVectorBody_Head = [DataFrame[f"{PredictLabel}_x"][Ind] - DataFrame["Body_x"][Ind],
                                        DataFrame[f"{PredictLabel}_y"][Ind] - DataFrame["Body_y"][Ind]]
            DirectionVectorR_Ear = [DataFrame[f"{LabelsFrom[0]}_x"][Ind] - DataFrame["Body_x"][Ind],
                                      DataFrame[f"{LabelsFrom[0]}_y"][Ind] - DataFrame["Body_y"][Ind]]
            DirectionVectorL_Ear = [DataFrame[f"{LabelsFrom[1]}_x"][Ind] - DataFrame["Body_x"][Ind],
                                      DataFrame[f"{LabelsFrom[1]}_y"][Ind] - DataFrame["Body_y"][Ind]]
            ThetaRight = VectorAngle(DirectionVectorBody_Head, DirectionVectorR_Ear)
            ThetaLeft = VectorAngle(DirectionVectorBody_Head, DirectionVectorL_Ear)
            AngleList_Right.append(ThetaRight) 
            AngleList_Left.append(ThetaLeft)

    Theta_Right = np.average(AngleList_Right)
    Theta_Left = np.average(AngleList_Left)

    Theta_Right_std = np.std(AngleList_Right)
    Theta_Left_std = np.std(AngleList_Left)

    Counter = 0
    C1 = 0
    C2 = 0
    for Ind, PVals in enumerate(DataFrame[f"{PredictLabel}_p-val"]):
        # AdjacentLabel = [Label for Label in LabelsFrom if DataFrame[f"{Label}_p-val"][Ind] >= CutOff]
        #If the Head label is poorly track initiate this statement
        if (PVals < CutOff):  
            if ((DataFrame[f"{LabelsFrom[0]}_p-val"][Ind] >= CutOff) and (DataFrame[f"{LabelsFrom[1]}_p-val"][Ind] >= CutOff)):
                MidPoint = [(DataFrame[f"{LabelsFrom[0]}_x"][Ind] + DataFrame[f"{LabelsFrom[1]}_x"][Ind])/2,
                            (DataFrame[f"{LabelsFrom[0]}_y"][Ind] + DataFrame[f"{LabelsFrom[1]}_y"][Ind])/2]
                ReferenceMid = MidPoint      
            elif ((DataFrame[f"{LabelsFrom[0]}_p-val"][Ind] < CutOff) or (DataFrame[f"{LabelsFrom[1]}_p-val"][Ind] < CutOff)):
                ##############
                #Choose surrounding label
                ##############
                AdjacentLabel = [Label for Label in LabelsFrom if DataFrame[f"{Label}_p-val"][Ind] >= CutOff]
                if ((len(AdjacentLabel) != 0) and (Ind != 0)):
                    if (DataFrame[f"{PredictLabel}_p-val"][Ind - 1] >= CutOff):
                        ##########
                        #Create Direction Vectors between the first adjacent label available in the list
                        #Parallelogram law
                        ##########
                        DirectionVec = [DataFrame[f"{PredictLabel}_x"][Ind - 1] - DataFrame[f"{AdjacentLabel[0]}_x"][Ind - 1], 
                                        DataFrame[f"{PredictLabel}_y"][Ind - 1] - DataFrame[f"{AdjacentLabel[0]}_y"][Ind - 1]]
                        ReferenceDirection = DirectionVec
                    elif ((DataFrame[f"{PredictLabel}_p-val"][Ind - 1] < CutOff) and (len(ReferenceDirection) == 0)):
                        ReferenceDirection = [0, 0]  
                    ###########
                    #Compute the displacement between available surronding label
                    #Compute the vector addition (parallelogram law) and scale the Ind - 1 first available adjacent label by it
                    ###########
                    Displacement = [DataFrame[f"{AdjacentLabel[0]}_x"][Ind] - DataFrame[f"{AdjacentLabel[0]}_x"][Ind - 1],
                                    DataFrame[f"{AdjacentLabel[0]}_y"][Ind] - DataFrame[f"{AdjacentLabel[0]}_y"][Ind - 1]]
                    Scale = np.add(ReferenceDirection, Displacement)
                    
                    #Reference Mid is a 2D coordinate
                    ReferenceMid = [DataFrame[f"{AdjacentLabel[0]}_x"][Ind - 1] + Scale[0], DataFrame[f"{AdjacentLabel[0]}_y"][Ind - 1] + Scale[1]] 
                
                    if DataFrame["Body_p-val"][Ind] >= CutOff:
                        BodyAdjacent_Vector = [DataFrame[f"{AdjacentLabel[0]}_x"][Ind] - DataFrame["Body_x"][Ind],
                                                DataFrame[f"{AdjacentLabel[0]}_y"][Ind] - DataFrame["Body_y"][Ind]]
                        
                        #Create a vector from the body label to the midpoint
                        ScaledVec = [ReferenceMid[0] - DataFrame["Body_x"][Ind],
                                      ReferenceMid[1] - DataFrame["Body_y"][Ind]]
                        
                        Theta = VectorAngle(BodyAdjacent_Vector, ScaledVec)
                        VectorPosition = np.cross(BodyAdjacent_Vector, ScaledVec)
        
                        
                        if AdjacentLabel[0] == "Left_Ear":
                            if (VectorPosition < 0):
                                C1 += 1
                                RotateAngle = Theta_Left + Theta
                                RotateVector = RotationMatrixCCW(RotateAngle).dot(ScaledVec)
                                ReferenceMid = [DataFrame["Body_x"][Ind] + RotateVector[0], DataFrame["Body_y"][Ind] + RotateVector[1]]
                                DataFrame[f"{PredictLabel}_p-val"][Ind] = 2.5
                                
                            #Correct Position   
                            elif (VectorPosition > 0):
                                C2 += 1
                                if (Theta < (Theta_Left - (2 * Theta_Left_std))):
                                    RotateAngle = Theta_Left - Theta
                                    RotateVector = RotationMatrixCCW(RotateAngle).dot(ScaledVec)
                                    ReferenceMid = [DataFrame["Body_x"][Ind] + RotateVector[0], DataFrame["Body_y"][Ind] + RotateVector[1]]
                                    DataFrame[f"{PredictLabel}_p-val"][Ind] = 2.5
                                elif (Theta > (Theta_Left + (2 * Theta_Left_std))):
                                    RotateAngle = Theta - Theta_Left
                                    RotateVector = RotationMatrixCW(RotateAngle).dot(ScaledVec)
                                    ReferenceMid = [DataFrame["Body_x"][Ind] + RotateVector[0], DataFrame["Body_y"][Ind] + RotateVector[1]]
                                    DataFrame[f"{PredictLabel}_p-val"][Ind] = 2.5
                                else:
                                    DataFrame[f"{PredictLabel}_p-val"][Ind] = 4.0

                        elif AdjacentLabel[0] == "Right_Ear":
                            if VectorPosition < 0:
                                if (Theta < (Theta_Right - (2 * Theta_Right_std))):
                                    RotateAngle = Theta_Right - Theta
                                    RotateVector = RotationMatrixCW(RotateAngle).dot(ScaledVec)
                                    ReferenceMid = [DataFrame["Body_x"][Ind] + RotateVector[0], DataFrame["Body_y"][Ind] + RotateVector[1]]
                                    DataFrame[f"{PredictLabel}_p-val"][Ind] = 1.5
                                elif (Theta > (Theta_Right + (2 * Theta_Right_std))):
                                    RotateAngle = Theta - Theta_Right
                                    RotateVector = RotationMatrixCCW(RotateAngle).dot(ScaledVec)
                                    ReferenceMid = [DataFrame["Body_x"][Ind] + RotateVector[0], DataFrame["Body_y"][Ind] + RotateVector[1]]
                                    DataFrame[f"{PredictLabel}_p-val"][Ind] = 1.5
                                else:
                                    DataFrame[f"{PredictLabel}_p-val"][Ind] = 4.0
                            elif VectorPosition > 0:
                                RotateAngle = Theta_Right + Theta
                                RotateVector = np.dot(RotationMatrixCW(RotateAngle), ScaledVec)
                                ReferenceMid = [DataFrame["Body_x"][Ind] + RotateVector[0], DataFrame["Body_y"][Ind] + RotateVector[1]]
                                DataFrame[f"{PredictLabel}_p-val"][Ind] = 1.5
                    
                    # DataFrame[f"{PredictLabel}_p-val"][Ind] = 4.0
                elif ((len(AdjacentLabel) == 0)):
                    Counter += 1    
            try:
                DataFrame[f"{PredictLabel}_x"][Ind] = ReferenceMid[0]
                DataFrame[f"{PredictLabel}_y"][Ind] = ReferenceMid[1]
            except IndexError:
                pass
            if DataFrame[f"{PredictLabel}_p-val"][Ind] < 1.0:
                DataFrame[f"{PredictLabel}_p-val"][Ind] = 3.5   
    # print(C1, C2)
    # breakpoint()
    PVAL_PREDICTEDLABEL = list(DataFrame[f"{PredictLabel}_p-val"])
    #DataFrame.to_csv(r"F:\WorkFiles_XCELLeration\Video\DifferentApproaches\Combined_PgramRotation.csv")
    DataFrame = DataFrame.rename(columns={NewColumns[Ind]: OldColumns[Ind] for Ind in range(len(OldColumns))})
    return(DataFrame, PVAL_PREDICTEDLABEL)


def computeEuclideanDistance(DataFrame, BodyParts):
    """
    Function responsible for computing the interframe Euclidean Distance
    Applies the 2D Euclidean distance formula between frames on the coordinates of each tracked
    label from DLC.

        d(p, q) = sqrt(sum(q - p) ** 2))

        - where p, q are 2D cartesian coordinates, in this case the coordinate labels
        in sequential frames.

    Parameters
    ----------
    Data frames and body part strings as inputs

    Returns
    -------
    The function returns a list of these frames
    """
    DistanceVectors = []
    ColsToDrop = [Cols for Cols in DataFrame if Cols % 3 == 0]
    DataFrame = DataFrame.drop(ColsToDrop, axis = 1)
    CreateDirectionalVectors = lambda Vec1, Vec2: [Vals2 - Vals1 for Vals1, Vals2 in zip(Vec1, Vec2)]
    ComputeNorm = lambda Vec: np.sqrt(sum(x ** 2 for x in Vec))
    for Cols1, Cols2 in zip(DataFrame.columns.values[:-1], DataFrame.columns.values[1:]):
        if Cols2 - Cols1 == 1:
            VectorizedFrame = list(zip(pd.to_numeric(DataFrame[Cols1], downcast="float"), pd.to_numeric(DataFrame[Cols2], downcast="float")))
            DirectionalVectors = list(map(CreateDirectionalVectors, VectorizedFrame[:-1], VectorizedFrame[1:]))
            Norm = list(map(ComputeNorm, DirectionalVectors))
            DistanceVectors.append(Norm)
    EDFrame = pd.DataFrame(data={BodyParts[Ind]: DistanceVectors[Ind] for Ind in range(len(DistanceVectors))})
    return(EDFrame)

def computeHourlySums(DataFrameList):
    """
    Function responsible for creating hourly sums, that is, the summed Euclidean
    Distance for that hour (or .csv input). This represents the total motility of the
    animal in the given time frame.

    Parameters
    ----------
    Data frame list as input

    Returns
    -------
    A single dataframe containing the sums for that hour (or .csv input). The index will
    act as the hour or timescale for that particular .csv, therefore it is important to ensure
    that .csv files are in order.

    """
    SumLists = []
    for Frames in range(len(DataFrameList)):
        SumFunction = DataFrameList[Frames].apply(np.sum, axis=0)
        SummedFrame = pd.DataFrame(SumFunction)
        SumLists.append(SummedFrame.transpose())
    AdjustedFrame = pd.concat(SumLists).reset_index(drop=True)
    return(AdjustedFrame)

def computeLinearEquations(HourlyFrame):
    """
    Function responsible for creating linear equations from the hourly sums

    Parameters
    ----------
    Data frame as input

    Returns
    -------
    A single dataframe containing the slope, intecept and hourly values of that line

    """
    SlopeFunction = lambda Column: (((Column[Ind2] - Column[Ind1])/(Ind2 - Ind1)) for Ind1, Ind2 in zip(Column.index.values[:-1], Column.index.values[1:]))
    Slope = [list(SlopeFunction(HourlyFrame[Cols])) for Cols in HourlyFrame]
    InterceptFunction = lambda Column, Slopes, Time: ((ColVals - (SlopeVals * TimeVals))
                                                      for ColVals, SlopeVals, TimeVals in zip(Column, Slopes, Time))
    Intercept = [list(InterceptFunction(HourlyFrame[Cols], Slope[rng], list(HourlyFrame.index.values))) for Cols, rng in zip(HourlyFrame, range(len(Slope)))]
    Zipper = [[(slope, intercept, start, end) for slope, intercept, start, end in zip(Col1, Col2, HourlyFrame.index.values[:-1], HourlyFrame.index.values[1:])]
              for Col1, Col2 in zip(Slope, Intercept)]
    LinearEquationFrame = pd.DataFrame(data={
        "LineEqn_{}".format(HourlyFrame.columns.values[Ind]): Zipper[Ind] for Ind in range(len(Zipper))
        })
    return(LinearEquationFrame)

def computeIntegrals(LinearEquationsFrame):
    """
    Function responsible for computing the integral of the linear equation between two
    consecutive time points

    Parameters
    ----------
    Data frame as input

    Returns
    -------
    A single dataframe containing the integral values (Area under curve) for the respective
    linear equation. Between consecutive time points.

    """
    Integral = lambda m, x, b: (m*x) + b
    IntegralList = [[quad(Integral, Vals[2], Vals[3], args = (Vals[0], Vals[1]))[0] for Vals in LinearEquationsFrame[Cols]] for Cols in LinearEquationsFrame]
    ColNames = LinearEquationsFrame.columns.values
    IntegralFrame = pd.DataFrame(data={
        "Integral_{}".format(ColNames[Ind].split("_")[1]):IntegralList[Ind] for Ind in range(len(IntegralList))
        })
    return(IntegralFrame)

#These should be moved to a residual computations folder
def computeAveragePositionStationary(InputFrame, StationaryObjectsList):
    StationaryDict = {StationaryObjectsList[Ind]: [0, 0] for Ind in range(len(StationaryObjectsList))}
    duplicates = [Cols for Cols in InputFrame.columns.values]
    #Know that coordinate data will only ever be 2D
    #Should not operate under that apriori assumption
    for Ind, Cols in enumerate(duplicates):
        if Cols in duplicates and Cols + "_x" not in duplicates:
            duplicates[duplicates.index(Cols)] = Cols + "_x"
        else:
            duplicates[duplicates.index(Cols)] = Cols + "_y"
    InputFrame.columns = duplicates
    for Cols in StationaryObjectsList:
        XCoord = Cols + "_x"
        YCoord = Cols + "_y"
        AverageX = np.average(list(pd.to_numeric(InputFrame[XCoord], downcast="float")))
        AverageY = np.average(list(pd.to_numeric(InputFrame[YCoord], downcast="float")))
        StationaryDict[Cols][0] = AverageX
        StationaryDict[Cols][1] = AverageY
    StationaryFrame = pd.DataFrame(data=StationaryDict)
    StationaryFrame = StationaryFrame.set_index(pd.Series(["x", "y"]))
    return(StationaryFrame)
    

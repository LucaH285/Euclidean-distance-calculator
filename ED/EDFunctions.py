# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 15:28:09 2021

@author: Luca Hategan
"""
import pandas as pd
from collections import OrderedDict
import numpy as np
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

def predictLabelLocation(DataFrame, CutOff, LOI, LabelsFrom, colNames, PredictLabel):
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
    Ind1 = 0
    Ind2 = 0
    for Cols in DataFrame.columns.values:
        if Ind1 <= 2:
            DataFrame = DataFrame.rename(columns={Cols:f"{colNames[Ind2]}{FeatureList[Ind1]}"})
            if Ind1 == 2:
                Ind1 = 0
                Ind2 += 1
            else:
                Ind1 += 1
    NewColumns = list(DataFrame.columns.values)
    for Cols in DataFrame.columns.values:
        DataFrame[Cols] = pd.to_numeric(DataFrame[Cols], downcast="float")
    ReferenceDirection = []
    ReferenceDisplacement = []
    for Ind, PVals in enumerate(DataFrame[f"{PredictLabel}_p-val"]):
        if (PVals < CutOff):
            ##############
            #Choose surrounding label
            ##############
            AdjacentLabel = [Label for Label in LabelsFrom if DataFrame[f"{Label}_p-val"][Ind] >= CutOff]
            if (len(AdjacentLabel) != 0):
                if (DataFrame[f"{PredictLabel}_p-val"][Ind - 1] >= CutOff):
                    DirectionVec = [DataFrame[f"{PredictLabel}_x"][Ind - 1] - DataFrame[f"{AdjacentLabel[0]}_x"][Ind - 1], 
                                    DataFrame[f"{PredictLabel}_y"][Ind - 1] - DataFrame[f"{AdjacentLabel[0]}_y"][Ind - 1]]
                    ReferenceDirection = DirectionVec
                Displacement = [DataFrame[f"{AdjacentLabel[0]}_x"][Ind] - DataFrame[f"{AdjacentLabel[0]}_x"][Ind - 1],
                                DataFrame[f"{AdjacentLabel[0]}_y"][Ind] - DataFrame[f"{AdjacentLabel[0]}_y"][Ind - 1]]
                ReferenceDisplacement = Displacement
                Scale = [Ji + Jj for Ji, Jj in zip(Displacement, ReferenceDirection)]
                DataFrame[f"{PredictLabel}_x"][Ind] = DataFrame[f"{AdjacentLabel[0]}_x"][Ind - 1] + Scale[0]
                DataFrame[f"{PredictLabel}_y"][Ind] = DataFrame[f"{AdjacentLabel[0]}_y"][Ind - 1] + Scale[1]
                DataFrame[f"{PredictLabel}_p-val"] = 1.0
            elif (len(AdjacentLabel) == 0):
                Scale = [Ji + Jj for Ji, Jj in zip(ReferenceDisplacement, ReferenceDirection)]
                DataFrame[f"{PredictLabel}_x"][Ind] = DataFrame[f"{AdjacentLabel[0]}_x"][Ind - 1] + Scale[0]
                DataFrame[f"{PredictLabel}_y"][Ind] = DataFrame[f"{AdjacentLabel[0]}_y"][Ind - 1] + Scale[1]
                DataFrame[f"{PredictLabel}_p-val"] = 1.0
                
                
    DataFrame = DataFrame.rename(columns={NewColumns[Ind]: OldColumns[Ind] for Ind in range(len(OldColumns))})
    return(DataFrame) 

def predictLabel_MidpointAdjacent():
    pass

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
    

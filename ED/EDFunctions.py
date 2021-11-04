# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 15:28:09 2021

@author: Luca Hategan
"""
import pandas as pd
from collections import OrderedDict
import numpy as np

def preprocessor(DataFrame):
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
    FrameList = []
    AdjustFrame = DataFrame[[i for i in range(1, 4)]]
    AdjustFrame = AdjustFrame.mask(pd.to_numeric(AdjustFrame[3], downcast="float") < CutOff).ffill()
    return(AdjustFrame)
    
def computeEuclideanDistance(DataFrame):
    ColsToDrop = [Cols for Cols in DataFrame if Cols % 3 == 0]

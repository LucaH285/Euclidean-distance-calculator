# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 20:22:00 2021

@author: Luca Hategan
"""
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



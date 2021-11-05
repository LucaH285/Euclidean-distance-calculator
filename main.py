# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 14:21:04 2021

@author: Luca Hategan
"""
from ED import FileImport
from ED import EDFunctions
from abc import ABC, abstractmethod

class loadPreprocess:
    def __init__(self, FilePath, PValCutoff):
        self.Source = FilePath
        self.PVal = PValCutoff

    def loadFiles(self):
        Frames = FileImport.Import(self.Source)
        return(Frames)

    def preprocess(self):
        PreprocessedFrames = [EDFunctions.preprocessor(Frames) for Frames in self.loadFiles()]
        PValAdjustedFrames = [EDFunctions.checkPVals(Frames[0], self.PVal) for Frames in PreprocessedFrames]
        BodyPartList = [Parts[1] for Parts in PreprocessedFrames]
        return(PValAdjustedFrames, BodyPartList)

class computeEuclideanDistance(loadPreprocess):
    def computeED(self):
        FrameList = [EDFunctions.computeEuclideanDistance(Frames, self.preprocess()[1][0]) for Frames in self.preprocess()[0]]
        return(FrameList)

    def createHourlySums(self):
        HourlySumLists = EDFunctions.computeHourlySums(self.computeED())
        return(HourlySumLists)


if __name__=="__main__":
    FilePath="/Users/lucahategan/Desktop/For work/work files/drive-download-20200528T164242Z-001"
    Class = computeEuclideanDistance(FilePath, PValCutoff = 0.95)
    Class.createHourlySums()

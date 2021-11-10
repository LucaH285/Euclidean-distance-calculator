# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 14:21:04 2021

@author: Luca Hategan
"""
from ED import FileImport
from ED import EDFunctions
from abc import ABC, abstractmethod
from Graphs import GraphFunctions
#import pandas
import os
import warnings
import shutil

class loadPreprocess(object):
    def __init__(self, FilePath, PValCutoff, FPS):
        self.Source = FilePath
        self.PVal = PValCutoff
        self.FramesPerSecond = FPS

    def loadFiles(self):
        Frames = FileImport.Import(self.Source)
        return(Frames)

    def preprocess(self):
        PreprocessedFrames = [EDFunctions.preprocessor(Frames) for Frames in self.loadFiles()]
        PValAdjustedFrames = [EDFunctions.checkPVals(Frames[0], self.PVal) for Frames in PreprocessedFrames]
        BodyPartList = [Parts[1] for Parts in PreprocessedFrames]
        return(PValAdjustedFrames, BodyPartList)
    
    def computeEuclideanDistance(self):
        FrameList = [EDFunctions.computeEuclideanDistance(Frames, self.preprocess()[1][0]) for Frames in self.preprocess()[0]]
        return(FrameList)

class computations(ABC):
    @abstractmethod
    def compute(self, InputFile):
        pass
    
class computationsWithExport(computations):
    @abstractmethod
    def exportFunction(self):
        pass
    
    @abstractmethod
    def compute(self, InputFile):
        pass
    
class createHourlySum(computationsWithExport):
    
    ListOfFrames = []
    
    def __init__(self, ExportFilePath = ""):
        self.Source = ExportFilePath
    
    def compute(self, InputFile):
        HourlySumLists = EDFunctions.computeHourlySums(InputFile)
        self.ListOfFrames.append(HourlySumLists)
        return(HourlySumLists)
    
    def exportFunction(self):
        if os.path.isdir(self.Source) is True:
            Path = "Hourly_Sums"
            Dir = os.path.join(self.Source, Path)
            if os.path.exists(Dir) is False:
                os.mkdir(Dir)
            else:
                shutil.rmtree(Dir)
                os.mkdir(Dir)
                print("Hourly Sum directory exits, overwriting")
            for Index, Files in enumerate(self.ListOfFrames):
                Files.to_csv("{0}/SumFrame_{1}.csv".format(Dir, Index))
        elif os.path.isdir(self.Source) is False:
            warnings.warn("No source entered for hourly sum frame exports, passing export")
                
class createLinearEquations(computations):
    def compute(self, InputFile):
        LinearEquationFrame = EDFunctions.computeLinearEquations(InputFile)
        return(LinearEquationFrame)
    
class computeIntegrals(computationsWithExport):
    
    ListOfFrames = []
    
    def __init__(self, ExportFilePath = ""):
        self.Source = ExportFilePath
    
    def compute(self, InputFile):
        IntegralFrame = EDFunctions.computeIntegrals(InputFile)
        self.ListOfFrames.append(IntegralFrame)
        return(IntegralFrame)
    
    def exportFunction(self):
        if os.path.isdir(self.Source) is True:
            Path = "IntegralFrame"
            Dir = os.path.join(self.Source, Path)
            if os.path.exists(Dir) is False:
                os.mkdir(Dir)
            else:
                shutil.rmtree(Dir)
                os.mkdir(Dir)
                print("Integral directory exits, overwriting")
            for Index, Files in enumerate(self.ListOfFrames):
                Files.to_csv("{0}/IntegralFrame_{1}.csv".format(Dir, Index))
        elif os.path.isdir(self.Source) is False:
            warnings.warn("No source entered for hourly Integral Frame exports, passing export")
    
class MainGraphs(ABC):
    @abstractmethod
    def sendToGraph(self, InputFile):
        pass
    
class linePlot(MainGraphs):
    def sendToGraph(self, InputFile):
        Graph = GraphFunctions.lineplot(InputFile)
        return(Graph)

if __name__=="__main__":
    FilePath=r"F:\work\TestVideos_NewNetwork\20191206-20200507T194022Z-001\20191206\RawVids\RawVideos2"
    Class = loadPreprocess(FilePath, PValCutoff = 0.95, FPS=4)
    
    EuclideanDistances = Class.computeEuclideanDistance()
    computeSums = createHourlySum().compute(InputFile = EuclideanDistances)
    Export = createHourlySum(ExportFilePath=r"F:\work\TestVideos_NewNetwork\20191206-20200507T194022Z-001\20191206\RawVids\RawVideos2").exportFunction()
    
    computeLinearEqn = createLinearEquations().compute(InputFile=computeSums)
    computeIntegral = computeIntegrals().compute(InputFile = computeLinearEqn)
    Export2 = computeIntegrals(ExportFilePath=r"F:\work\TestVideos_NewNetwork\20191206-20200507T194022Z-001\20191206\RawVids\RawVideos2").exportFunction()
    
    print(computeSums)
    linePlot().sendToGraph(InputFile = computeSums)
    
    
    
    print(computeIntegral)

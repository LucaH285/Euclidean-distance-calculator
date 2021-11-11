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
import numpy as np
import time

"""
Structure the code such that this program is executed per file Input
Files should be retrieved from another python program that includes the CLI
"""


class loadPreprocess(object):

    BodyPartList = []

    def __init__(self, FilePath, PValCutoff, FPS):
        self.Source = FilePath
        self.PVal = PValCutoff
        self.FramesPerSecond = FPS

    def loadFiles(self):
        Frames = []
        for Index, Files in enumerate(self.Source):
            Frames.append(FileImport.Import(Files))
        return(Frames)

    def preprocess(self):
        PreprocessedFrames = [[] for _ in range(len(self.Source))]
        for Index, Files in enumerate(self.loadFiles()):
            for Frames in Files:
                Preprocess = EDFunctions.preprocessor(Frames)
                PValAdjust = EDFunctions.checkPVals(Preprocess[0], self.PVal)
                PreprocessedFrames[Index].append(PValAdjust)
                self.BodyPartList.append(Preprocess[1])
        return(PreprocessedFrames)

    def checkBodyPartList(self):
        #function automatically called by BodyPartList
        for Lists1, Lists2 in zip(self.BodyPartList[:-1], self.BodyPartList[1:]):
            if (np.array(Lists1) == np.array(Lists2)).all() is False:
                raise(TypeError("Please check your input files, the body parts between {0} and {1} are not equal".format(Lists1, Lists2)))
            else:
                pass
        return("Checked body parts for equivalency")

    def computeEuclideanDistance(self):
        PreprocessedFrames = self.preprocess()
        print(self.checkBodyPartList())
        FrameList = [[EDFunctions.computeEuclideanDistance(Frames, self.BodyPartList[0]) for Frames in File] for File in PreprocessedFrames]
        return(FrameList)

class computations(ABC):
    @abstractmethod
    def compute(self, InputFileList):
        pass

class computationsWithExport(computations):
    @abstractmethod
    def exportFunction(self):
        pass

    @abstractmethod
    def compute(self, InputFileList):
        pass

    @abstractmethod
    def reorientAxis(self, InputFileList):
        pass

class createHourlySum(computationsWithExport):

    ListOfFrames = []

    def __init__(self, ExportFilePath = ""):
        self.Source = ExportFilePath

    def compute(self, InputFileList):
        HourlySumLists = [EDFunctions.computeHourlySums(FileList) for FileList in InputFileList]
        self.ListOfFrames.append(HourlySumLists)
        return(HourlySumLists)

    def reorientAxis(self, InputFileList):
        pass

    def exportFunction(self):
        if os.path.isdir(self.Source) is True:
            #Parent folder creation, this should not change
            Path = "Hourly_Sums"
            Dir = os.path.join(self.Source, Path)
            if os.path.exists(Dir) is False:
                os.mkdir(Dir)
            else:
                shutil.rmtree(Dir)
                os.mkdir(Dir)
                print("Hourly Sum directory exits, overwriting")
            for Index, Files in enumerate(self.ListOfFrames):
                Files[0].to_csv("{0}/SumFrame_{1}.csv".format(Dir, Index))
        elif os.path.isdir(self.Source) is False:
            warnings.warn("No source entered for hourly sum frame exports, passing export")

class createLinearEquations(computations):
    def compute(self, InputFileList):
        LinearEquationFrame = [EDFunctions.computeLinearEquations(Frames) for Frames in InputFileList]
        return(LinearEquationFrame)

class computeIntegrals(computationsWithExport):

    ListOfFrames = []

    def __init__(self, ExportFilePath = ""):
        self.Source = ExportFilePath

    def compute(self, InputFileList):
        IntegralFrame = [EDFunctions.computeIntegrals(Frames) for Frames in InputFileList]
        self.ListOfFrames.append(IntegralFrame)
        return(IntegralFrame)

    def exportFunction(self):
        if os.path.isdir(self.Source) is True:
            #Parent folder creation, this stays constant
            Path = "IntegralFrame"
            Dir = os.path.join(self.Source, Path)
            if os.path.exists(Dir) is False:
                os.mkdir(Dir)
            else:
                shutil.rmtree(Dir)
                os.mkdir(Dir)
                print("Integral directory exits, overwriting")
            for Index, Files in enumerate(self.ListOfFrames):
                Files[0].to_csv("{0}/IntegralFrame_{1}.csv".format(Dir, Index))
        elif os.path.isdir(self.Source) is False:
            warnings.warn("No source entered for hourly Integral Frame exports, passing export")

class MainGraphs(ABC):
    @abstractmethod
    def sendToGraph(self, InputFileList, GenotypeIdentifier, SexIdentifier):
        pass

    @abstractmethod
    def sendToGraph_Generic(self, InputFileList):
        pass

class linePlot(MainGraphs):
    def sendToGraph(self, InputFile, GenotypeIdentifier, SexIdentifier):
        Graph = GraphFunctions.lineplot(InputFile)
        return(Graph)


if __name__=="__main__":
    FilePath=["/Users/lucahategan/Desktop/For work/work files/drive-download-20200528T164242Z-001"]
    Class = loadPreprocess(FilePath, PValCutoff = 0.95, FPS=4)

    EuclideanDistances = Class.computeEuclideanDistance()
    computeSums = createHourlySum().compute(InputFileList = EuclideanDistances)
    Export = createHourlySum(ExportFilePath="/Users/lucahategan/Desktop/For work/work files/drive-download-20200528T164242Z-001").exportFunction()

    computeLinearEqn = createLinearEquations().compute(InputFileList=computeSums)
    computeIntegral = computeIntegrals().compute(InputFileList = computeLinearEqn)
    Export2 = computeIntegrals(ExportFilePath="/Users/lucahategan/Desktop/For work/work files/drive-download-20200528T164242Z-001").exportFunction()

    #linePlot().sendToGraph(InputFile = computeSums, GenotypeIdentifier = ["WT"], SexIdentifier = ["Male"])

# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 14:21:04 2021

@author: Luca Hategan
"""
from ED import FileImport
from ED import EDFunctions
from abc import ABC, abstractmethod
from Graphs import GraphFunctions
from VectorCalculations import VectorFunctions
from ResidualCalculations import ResidualFunctions as RF
from functools import reduce
import math
import matplotlib.pyplot as mp
#import pandas
import os
import warnings
import shutil
import numpy as np
import pandas as pd
import itertools
import time
import click

PREDICTLABEL_PVAL = []

class loadPreprocess(object):

    BodyPartList = []
    PreProcessedFrames = []

    def __init__(self, FilePath, PValCutoff, FPS, Predict,
                 ReferenceLabels, PredictLabel):
        self.Source = FilePath
        self.PVal = PValCutoff
        self.FramesPerSecond = FPS
        self.predict_label = Predict
        self.ReferenceLabel = ReferenceLabels
        self.LabelToPredict = PredictLabel

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
                if self.predict_label == True:
                    Predict = EDFunctions.predictLabel_MidpointAdjacent(Preprocess[0], self.PVal, 
                                                                  LabelsFrom=self.ReferenceLabel, 
                                                                  colNames = Preprocess[1], 
                                                                  PredictLabel=self.LabelToPredict)
                    
                    PREDICTLABEL_PVAL.append(Predict[1])
                elif self.predict_label == False:
                    Predict = [Preprocess[0]]
                PValAdjust = EDFunctions.checkPVals(Predict[0], self.PVal)
                # PValAdjust.to_csv(r"F:\WorkFiles_XCELLeration\Video\Trim2\DF_edited_adjust.csv")
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
        print("Checked body parts for equivalency")
        return(self.BodyPartList)

    def __call__(self):
        PreProcessedFrames = self.preprocess()
        print(self.BodyPartList[0])
        Len = len(PreProcessedFrames[0][0])
        XVal = list(pd.to_numeric(PreProcessedFrames[0][0].loc[Len-10000:Len, 4], downcast="float"))
        YVal = [i for i in range(len(XVal))]#list(pd.to_numeric(PreProcessedFrames[0][0].loc[20000:29000, 2], downcast = "float"))
        #GraphFunctions.genericGraph(YVal, XVal, Xlab="Time (arbitrary)", Ylab="X-position of Head", Title="Time vs. Position")
        BodyParts = self.checkBodyPartList()
        return(PreProcessedFrames, BodyParts)

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

class computationsExportAndReorientAxis(computations):
    @abstractmethod
    def compute(self, InputFileList):
        pass

    @abstractmethod
    def reorientAxis(self, InputFileList, ReIndex = []):
        pass

    @abstractmethod
    def exportFunction(self):
        pass

class computeEuclideanDistance(computationsWithExport):

    ListOfFrames = []

    def __init__(self, ExportFilePath = "", BodyPartList = []):
        self.Source = ExportFilePath
        self.BodyParts = BodyPartList

    def compute(self, InputFileList):
        FrameList = [[EDFunctions.computeEuclideanDistance(Frames, self.BodyParts) for Frames in File] for File in InputFileList]
        self.ListOfFrames = FrameList
        return(FrameList)

    def exportFunction(self):
        if os.path.isdir(self.Source) is True:
            #Parent folder creation, this should not change
            Path = "Euclidean_Distances"
            Dir = os.path.join(self.Source, Path)
            if os.path.exists(Dir) is False:
                os.mkdir(Dir)
            else:
                shutil.rmtree(Dir)
                os.mkdir(Dir)
                print("Euclidean distance directory exits, overwriting")
            for Index, Files in enumerate(self.ListOfFrames):
                Files.to_csv("{0}/EDFrame_{1}.csv".format(Dir, Index))
        elif os.path.isdir(self.Source) is False:
            warnings.warn("No source entered for Euclidean distance frame exports, passing export")

class createHourlySum(computationsExportAndReorientAxis):

    ListOfFrames = []

    def __init__(self, ExportFilePath = ""):
        self.Source = ExportFilePath

    def compute(self, InputFileList):
        HourlySumLists = [EDFunctions.computeHourlySums(FileList) for FileList in InputFileList]
        for Frames in HourlySumLists:
            self.ListOfFrames.append(Frames)
        return(HourlySumLists)

    def reorientAxis(self, InputFileList, ReIndex):
        """
        Function split into 2 parts, first checks that all frames contain 24 hr of footage - raise error if this is not the case
        If the files pass this check, then reorient the axis
        """
        for Index, Frames in enumerate(InputFileList):
            if (len(list(Frames.index.values)) == len(ReIndex)) and (sorted(list(Frames.index.values)) == sorted(ReIndex)):
                Frames = Frames.reindex(ReIndex)
                self.ListOfFrames[Index] = Frames
            elif (len(list(Frames.index.values)) == len(ReIndex)) and (sorted(list(Frames.index.values)) != sorted(ReIndex)):
                print("Assigning custom index values: {}. Reformatting hours of the day".format(ReIndex))
                Frames["Index"] = ReIndex
                Frames = Frames.set_index(["Index"], drop=True)
                self.ListOfFrames[Index] = Frames
            else:
                raise(IndexError("Index length does not match frame index length, either pass a correctly sized index list or leave blank"))
        return(self.ListOfFrames)

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
                Files.to_csv("{0}/SumFrame_{1}.csv".format(Dir, Index))
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
        for Frames in IntegralFrame:
            self.ListOfFrames.append(Frames)
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
                Files.to_csv("{0}/IntegralFrame_{1}.csv".format(Dir, Index))
        elif os.path.isdir(self.Source) is False:
            warnings.warn("No source entered for hourly Integral Frame exports, passing export")

class residualComputations(ABC):
    @abstractmethod
    def residualcomputation(self, InputFileList):
        pass

class residualComputationsWithExport(residualComputations):
    @abstractmethod
    def residualcomputation(self, InputFileList):
        pass

    @abstractmethod
    def exportFunction(self):
        pass
#all the code from this should be moved to a dedicated .py file
#Move it to residual functions.
class computeAverageObjectPosition(residualComputations):
    def __init__(self, LabelsOfInterest = [], AllLabels = []):
        self.ObjectLabels = LabelsOfInterest
        self.Labels = AllLabels

    def residualcomputation(self, InputFileList):
        """
        Input file list should be the positional coordinates of each CSV.
        Should pass the positional data from loadpreprocess class.

        Split into an initial preprocessing step, then passed to functions file where the average
        coordinates of the stationary objects are computed and returned per file.
        """
        if set(self.ObjectLabels).issubset(self.Labels) is True:
            ColsToUse = [Str for StrInit in self.Labels for Str in [StrInit]*2]
            Reset2ColNames = [[] for _ in range(len(InputFileList))]
            for Ind, Files in enumerate(InputFileList):
                for Frames in Files:
                    ColsToDrop = [Cols for Cols in Frames.columns.values if Cols % 3 == 0]
                    Frames = Frames.drop(ColsToDrop, axis = 1)
                    Frames = Frames.rename(columns={Cols:ColsToUse[Ind] for Cols, Ind in zip(Frames.columns.values, range(len(ColsToUse)))})
                    Reset2ColNames[Ind].append(Frames)
            #Rename all columns to x, y
            #Call them in the main function as: for Cols + "_x", Cols + "_y" in objectlabels
            AverageStationaryPosition = [[EDFunctions.computeAveragePositionStationary(Frames, self.ObjectLabels) for Frames in Files] for Files in Reset2ColNames]
            #Return here is a list of lists containing frames with the average coordinate position of stationary objects
            #This will be list of 1 row frames with n columns (depending on how many objects of interest were defined)
            return(AverageStationaryPosition)
        else:
            raise(ValueError("Labels in the labels of interest do not match all of the labels tracked for this experiment, please check your input"))

class createStationaryVectors(residualComputations):
    def __init__(self, drawVectorsFrom = [], drawVectorsTo = []):
        self.startLabels = drawVectorsFrom
        self.endLabels = drawVectorsTo

    def residualcomputation(self, InputFile):
        """
        First create cross cage vectors and find the point of intersection of the 2
        lines.
        Then calculate angular velocity as dTheta/dT

        Whole cage circling behavior
        """
        StationaryValues = list(InputFile[0][0].columns.values)
        if ((set(self.startLabels).issubset(StationaryValues) and set(self.endLabels).issubset(StationaryValues)) and (len(StationaryValues) % 2 == 0)):
            #return 2-tuple acting as direction vector
            ComputedStationaryVecs = [[] for _ in range(len(InputFile))]
            #return a list of vectors containing the position vector and direction vector
            Vectors = lambda StartLabel, EndLabel: [Vj - Vi for Vi, Vj in zip(StartLabel, EndLabel)]
            for Ind, Files in enumerate(InputFile):
                for Frames in Files:
                    for SCols, ECols in zip(self.startLabels, self.endLabels):
                        Vector = Vectors(Frames[SCols], Frames[ECols])
                        ComputedStationaryVecs[Ind].append((list(Frames[SCols]), Vector))
            return(ComputedStationaryVecs)
        else:
            raise(KeyError("Inputted stationary labels are not a part of the stationary values tracked"))

class computePointOfIntersection(residualComputations):
    def residualcomputation(self, InputFile):
        Centroids = []
        POI = lambda Scalar, VecTuple: [(Posi + (Scalar*Coefj)) for Posi, Coefj in zip(VecTuple[0], VecTuple[1])]
        for Files in InputFile:
            for Vecs1, Vecs2 in zip(Files[:-1], Files[1:]):
                """
                prepare the vectors for the augmented matrix:
                    Algorithm
                    - subtract the position vectors of vector list 1 from vector list 2: Vec1[1] - Vec2[1]
                    - set up the coefficient matrix using the directional vectors in the vector lists
                    - augment the matrix such that the values of the positon vectors are equal to the coefficient matrix

                    Coef =[[1, 2]
                           [3, 4]]

                    Position = [[5]
                                [6]]

                """
                PositionVectors = [(V2 - V1) for V1, V2 in zip(Vecs1[0], Vecs2[0])]
                CoefficientMatrix = np.array([Vecs1[1], [-1*Vals for Vals in Vecs2[1]]]).T
                Solution = np.linalg.solve(CoefficientMatrix, PositionVectors)
                Centroid = POI(Solution[0], Vecs1)
                Centroids.append(Centroid)
        CentroidArray = np.array(Centroids)
        AverageX = 0
        AverageY = 0
        for Vals in CentroidArray:
            AverageX += Vals[0]
            AverageY += Vals[1]
        AveragedCentroid = [(AverageX/len(Centroids)).round(3), (AverageY/len(Centroids)).round(3)]
        return(AveragedCentroid)

class computeAngularVelocity(residualComputations):
    def __init__(self, drawToLabel, CentroidCoord, AllLabels, FramesPerSecond):
        self.Label = drawToLabel
        self.Centroid = CentroidCoord
        self.AllLabels = AllLabels
        self.FPS = FramesPerSecond

    def residualcomputation(self, InputFile):
        """
        First draw the vector from the centroid to the label of interest, from the positional data.
        Second compute the change in angle over the change in time.
        """
        if set([self.Label]).issubset(self.AllLabels):
            RenamedCols = RF.renameCols(InputFileList=InputFile, BodyParts=self.AllLabels)
            CreatePositionVectors = lambda XCoords, YCoords: [[x, y] for x,y in zip(XCoords, YCoords)]
            ComputeCentroidLabelVec = lambda CentroidCoord, Coords: [[PosVecs[i] - CentroidCoord[j] for i, j in zip(range(len(PosVecs)), range(len(CentroidCoord)))] for PosVecs in Coords]
            ComputeVectorAngles = lambda Vectors: [math.degrees(np.arccos((np.dot(Vec1, Vec2))/(np.linalg.norm(Vec1)*np.linalg.norm(Vec2)))) for Vec1, Vec2 in zip(Vectors[:-1], Vectors[1:])]
            for Ind, Files in enumerate(RenamedCols):
                for Frames in Files:
                    PosVecs = CreatePositionVectors(XCoords=list(pd.to_numeric(Frames[self.Label+"_x"], downcast="float")), YCoords=list(pd.to_numeric(Frames[self.Label+"_y"], downcast="float")))
                    CentroidLabelVecs = ComputeCentroidLabelVec(CentroidCoord=self.Centroid, Coords=PosVecs)
                    dTheta = ComputeVectorAngles(CentroidLabelVecs)
                    AngularVelocity = [Theta/(1/self.FPS) for Theta in dTheta]
                    mp.plot([i * 1/30 for i in range(len(AngularVelocity))], AngularVelocity)
                    mp.xlabel("Time (seconds)")
                    mp.ylabel("Angular Velocity (dTheta/dT)")
                    mp.show()
                    print(AngularVelocity)
        else:
            raise(KeyError("Label of interest is not a label that has been tracked by DLC!"))

class circlingBehavior(residualComputations):
    def __init__(self, FromLabel, ToLabel, AllLabels, VideoIn, VideoOut, ScreenRes, **kwargs):
        self.LabelsToTrack_From = FromLabel
        self.LabelsToTrack_To = ToLabel
        self.AllLabels = AllLabels
        self.VideoInput = VideoIn
        self.VideoOutput = VideoOut
        self.Resolution = ScreenRes
        self.Labels_To = [kwargs[Labels] for Labels in kwargs]
        
    def residualcomputation(self, InputFileList):
        if set([self.LabelsToTrack_From]).issubset(self.AllLabels) and set([self.LabelsToTrack_From]).issubset(self.AllLabels):
            FileList = RF.renameCols(InputFileList=InputFileList, BodyParts=self.AllLabels)
            
            PositionVectorFunction = lambda CoordsX, CoordsY: [[x, y] for x, y in zip(pd.to_numeric(CoordsX, downcast="float"), pd.to_numeric(CoordsY, downcast="float"))]
            Midpoint = lambda PosVec1, PosVec2: [[(1/2)*(i + j) for i, j in zip(Vals1, Vals2)] for Vals1, Vals2 in zip(PosVec1, PosVec2)]
            MidpointLabelVector = lambda Midpoint, Coordinate: [[(V2 - V1) for V1, V2 in zip(Vals1, Vals2)] for Vals1, Vals2 in zip(Midpoint, Coordinate)]
            
            for Ind, Files in enumerate(FileList):
                for Frames in Files:
                    Coords_From = PositionVectorFunction(Frames[self.LabelsToTrack_From+"_x"], Frames[self.LabelsToTrack_From+"_y"])
                    Coords_To = PositionVectorFunction(Frames[self.LabelsToTrack_To+"_x"], Frames[self.LabelsToTrack_To+"_y"])
                    Midpoints = Midpoint(Coords_To, Coords_From)
                    
                    #MaxY=1080,
                    Vectors = MidpointLabelVector(Midpoints, Coords_To)
                    Quantifier = RF.rotationQuantifier(PositionVecX=Coords_From, PositionVecY=Coords_To, MaxY=self.Resolution[1], 
                                                       MaxX=self.Resolution[0], CriticalAngle=335)
                    
                    RF.TrackOnVideo(Annotations=Quantifier, videoFile=self.VideoInput,
                                    PositionVectorsX=Coords_From, PositionVectorsY=Coords_To, VideoOut = self.VideoOutput,
                                    PVals = PREDICTLABEL_PVAL)

        else:
            raise(KeyError("Label(s) of interest not tracked by DLC"))

class sinusodialRegression(residualComputations):
    def __init__(self, Labels = ""):
        self.LabelToTrack = Labels

    def residualComputations(self, InputFileList):
        pass

class vectorComputations(ABC):
    @abstractmethod
    def vectorCompute(self, Inputs):
        pass

class computeSkeleton(vectorComputations):
    def vectorCompute(self, Inputs):
        """
        Input here should be preprocessed dataframes.
        """
        for Frames in Inputs:
            CreateSkeleton = VectorFunctions.computeLabelVectors(Frames)

class MainGraph_ED(ABC):
    @abstractmethod
    def sendToGraph(self, InputFileList, GenotypeIdentifier, SexIdentifier, BodyPart):
        pass

class linePlot(MainGraph_ED):
    def sendToGraph(self, InputFile, GenotypeIdentifier, SexIdentifier, BodyPart):
        Graph = GraphFunctions.lineplot_forHourlySum(InputFile, GenotypeIdentifier, SexIdentifier, BodyPart)
        return(Graph)

class integralPlot(MainGraph_ED):
    def sendToGraph(self, InputFileList, GenotypeIdentifier, SexIdentifier, BodyPart):
        pass

class graphGeneric(ABC):
    @abstractmethod
    def sendToGraph_Generic(XVals, YVals):
        pass

class linePlot_Generic(graphGeneric):
    def __init__(self, Xlab, Ylab, Title):
        self.Xlab = Xlab
        self.Ylab = Ylab
        self.Title = Title

    def sendToGraph(XVals, YVals):
       # Graphs = GraphFunctions.genericGraph(XVals, YVals, self.Xlab, self.Ylab, self.Title)
        #probably don't need this as a variable.
        #return(Graphs)
        pass

if __name__=="__main__":
    FilePath=[r"F:\WorkFiles_XCELLeration\Video\2minTrim_end\PK-10-CTR_Rotation30_7month_May_30_2021_TrimDLC_resnet50_Parkinsons_RatNov13shuffle1_200000.csv"]
    OutPath = "",
    Class = loadPreprocess(FilePath, PValCutoff = 0.5, FPS=4, Predict = True, ReferenceLabels=["Right_Ear", "Left_Ear"], PredictLabel="Head")
    PreProcessedData = Class.__call__()

    #EuclideanDistances = computeEuclideanDistance(BodyPartList = PreProcessedData[1][0]).compute(InputFileList=PreProcessedData[0])
    #Export = computeEuclideanDistance(ExportFilePath=OutPath).exportFunction()

    #computeSums = createHourlySum().compute(InputFileList = EuclideanDistances)
    #Structure so that if called the computesums variable is replaced with the reindexed frame list.
    #computeSums = createHourlySum().reorientAxis(InputFileList=computeSums, ReIndex=[ 16, 17, 18, 19, 20, 21, 22, 23, 8, 9, 10, 11, 12, 13, 14, 15])
    #Export2 = createHourlySum(ExportFilePath=OutPath).exportFunction()

    #computeLinearEqn = createLinearEquations().compute(InputFileList=computeSums)

    #computeIntegral = computeIntegrals().compute(InputFileList = computeLinearEqn)
    #Export3 = computeIntegrals(ExportFilePath=OutPath).exportFunction()

    #StationaryFrames = computeAverageObjectPosition(LabelsOfInterest = ["TopWall", "RightWall", "BottomWall", "LeftWall"], AllLabels = PreProcessedData[1][0]).residualcomputation(InputFile
    #StationaryVectors = createStationaryVectors(drawVectorsFrom = ["TopWall", "RightWall"], drawVectorsTo = ["BottomWall", "LeftWall"]).residualcomputation(StationaryFrames)
    #POI = computePointOfIntersection().residualcomputation(InputFile = StationaryVectors)
    #computeAngularVelocity(drawToLabel = "Head", CentroidCoord=POI, AllLabels=PreProcessedData[1][0], FramesPerSecond=30).residualcomputation(InputFile = PreProcessedData[0])
    #Vectors = computeSkeleton().vectorCompute(Inputs = Class.returnPreprocessed())


    circling = circlingBehavior(FromLabel="Body", ToLabel="Head", ScreenRes = [1920, 1080],
                                VideoIn = r'F:\WorkFiles_XCELLeration\Video\2minTrim_end\PK-10-CTR_Rotation30_7month_May_30_2021_TrimDLC_resnet50_Parkinsons_RatNov13shuffle1_200000_labeled.mp4',
                                VideoOut = r"", AllLabels=PreProcessedData[1][0],
                                Label_To1 = "Left_Ear", Label_To2 = "Right_Ear").residualcomputation(InputFileList=PreProcessedData[0])
    
                                                                                                      
    

                                                                                                      
    #linePlot().sendToGraph(InputFile = computeSums, GenotypeIdentifier = ["WT", "KO"], SexIdentifier = ["Male", "Male"], BodyPart = "Body")

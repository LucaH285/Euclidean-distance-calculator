# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 00:08:39 2021

@author: Desktop
"""
import matplotlib.pyplot as mp
import numpy as np
import warnings
import seaborn as sns

def lineplot_forHourlySum(DataFrames, Genotype, Sex, BodyPart):
    fig, ax = mp.subplots()
    for Index, Frames in enumerate(DataFrames):
        if BodyPart.lower() == "all":
            for Cols in Frames.columns.values:
                XVals = list(Frames.index.values)
                YVals = list(Frames[Cols])
                ax.bar(XVals, YVals, label = f"{Cols}_{Genotype[Index]}_{Sex[Index]}")
                mp.xticks(list(np.arange(min(XVals), max(XVals) + 1, 1)), XVals)
        elif BodyPart in list(Frames.columns.values):
            XVals = list(Frames.index.values)
            YVals = list(Frames[BodyPart])
            ax.bar(XVals, YVals, label = BodyPart)
            mp.xticks(list(np.arange(min(XVals), max(XVals) + 1, 1)), XVals)
    warnings.filterwarnings("ignore")
    mp.xlabel("Hour of Day")
    mp.ylabel("Total motility per hour")
    mp.legend()
    mp.show()

def AreaUnderCurve(DataFrames, Genotype, Sex, BodyPart):
    pass

def genericGraph(XVals, YVals, Xlab, Ylab, Title):
    mp.plot(XVals, YVals)
    mp.xlabel(Xlab)
    mp.ylabel(Ylab)
    mp.title(Title)
    mp.show()
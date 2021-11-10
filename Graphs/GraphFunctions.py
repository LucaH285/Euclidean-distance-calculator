# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 00:08:39 2021

@author: Desktop
"""
import matplotlib.pyplot as mp


def lineplot(DataFrame):
    for Cols in DataFrame.columns.values:
        mp.plot(DataFrame.index.values, DataFrame[Cols], label=Cols)
    mp.xlabel("Hour of Day")
    mp.ylabel("Total motility per hour")
    mp.legend()
    mp.show()




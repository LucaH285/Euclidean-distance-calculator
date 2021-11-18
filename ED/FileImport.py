# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 14:22:12 2021

@author: Luca Hategan
"""
import pandas as pd
import os

def ImportFunction_IfPath(Path):
    """
    If the filepath is a directory (i.e.: a windows folder), this function will parse
    the directory and read all .csv files in it, with the stipulation that only .csv files
    outputted from DLC are included.
    
    Parameters
    ----------
    The input file path

    Returns
    -------
    The function returns a list of these frames.
    """
    CSVFileList = []
    for files in os.listdir(Path):
        if files.endswith(".csv"):
            CSVFileList.append(pd.read_csv("{0}/{1}".format(Path, files)))
    return(CSVFileList)

def ImportFunction_IfFile(Path):
    """
    If the filepath is an individual file in .csv, this function simply reads it and
    appends it to a list. List will contain a single element, being that one .csv file

    Parameters
    ----------
    The input file path

    Returns
    -------
    List of length 1.
    """
    CSVFileList = [pd.read_csv(Path, low_memory=False)]
    return(CSVFileList)


def Import(File):
    """
    The discriminator function, this function determines if the input file is a directory or an individual file
    """
    if os.path.isdir(File) is True:
        return(ImportFunction_IfPath(Path = File))
    elif os.path.isdir(File) is False:
        return(ImportFunction_IfFile(Path = File))
    else:
        raise(TypeError("The path was neither a directory nor a .csv file, please ensure you are inputting either the path of a directory or the path to a .csv file"))
        
        
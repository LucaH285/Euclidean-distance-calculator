# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 12:17:12 2021

@author: Desktop
"""

#ArchiveFunctions

# def circlingBehaviour2(MidPoints, MidLabelVectors, MaxY):
#     MaxYVectors_Fxn = lambda midpoints, maxY: [[0, maxY - Vectors[1]] for Vectors in midpoints]
#     MinYVectors_Fxn = lambda midpoints, minY: [[0, minY - Vectors[1]] for Vectors in midpoints]
#     MaxYVectors = MaxYVectors_Fxn(MidPoints, maxY=MaxY)
#     MinYVectors = MinYVectors_Fxn(MidPoints, minY = 0)
#     Angle = 0
#     AngleVector = []
#     thetavector = []
#     Rotations = 0
#     for consVector, sampleVector in zip(MaxYVectors, MidLabelVectors):
#         theta = math.degrees(np.arccos((np.dot(consVector, sampleVector))/(np.linalg.norm(consVector)*np.linalg.norm(sampleVector))))
#         thetavector.append(theta)
#         if sampleVector[0] > 0:
#             Angle = theta
#             AngleVector.append(Angle)
#         elif sampleVector[0] <= 0:
#             Angle = 360 - theta
#             AngleVector.append(Angle)
#             if Angle >= 357:
#                 Angle = 0
#                 Rotations += 1
#     print(max(AngleVector))
#     mp.plot(np.array(AngleVector[0:1000]))
#     mp.show()
#     print(Rotations)

"""
def aggregateCircling(Midpoints, VectorList, MaxY, MaxX):
    RotationalHashMap = {"Angle":0, "CWAngle":0, "CCWAngle":0, "CW":0, "CCW":0, "Total":0, "PartialCW":0, "PartialCCW":0}
    AngleVector = []
    for Vec1, Vec2 in zip(VectorList[:-1], VectorList[1:]):
        if Vec1 != Vec2:
            theta = math.degrees(math.acos((np.dot(Vec1, Vec2))/((np.linalg.norm(Vec1))*(np.linalg.norm(Vec2)))))
        else:
            theta = 0.0 
        # ClockwiseRotationMatrix = [[np.cos(theta), np.sin(theta)], 
        #                            [-1*np.sin(theta), np.cos(theta)]]    
        # CounterClockwiseRotationMatrix = [[np.cos(theta), -1*np.sin(theta)], 
        #                                   [np.sin(theta), np.cos(theta)]] 
        #print(np.cross(np.dot(np.array(ClockwiseRotationMatrix), Vec1), Vec2))
        # time.sleep(0.1)
        if RotationalHashMap["Angle"] < 358:
            RotationalHashMap["Angle"] += theta
            RotationalHashMap["PartialCW"] += theta/360
            AngleVector.append(RotationalHashMap["Angle"])
        elif RotationalHashMap["Angle"] >= 358:
            RotationalHashMap["Angle"] = 0
            RotationalHashMap["PartialCW"] = 0
            RotationalHashMap["Total"] += 1
            
    print(RotationalHashMap)
    
    mp.plot(np.array(AngleVector[10000:12000]))
    mp.xlabel("Frame Index")
    mp.ylabel("Angle-theta, Degrees")
    mp.title("Consecutive frame Midpoint-Head vector angles")
    mp.show()
    breakpoint()



def circlingBehaviour(VectorList, CriticalAngle):
    RotationalHashMap = {"Angle":0, "CCWAngle":0, "CW":0, "CCW":0, "PartialCW":0, "PartialCCW":0}
    AngleVector = []
    CCWAngleVector = []
    for Vec1, Vec2 in zip(VectorList[:-1], VectorList[1:]):
        if Vec1 != Vec2:
            theta = math.degrees(math.acos((np.dot(Vec1, Vec2))/((np.linalg.norm(Vec1))*(np.linalg.norm(Vec2)))))
        else:
            theta = 0.0
        #Lagrange Identity for CW/CCW
        Lagrange = np.linalg.norm(Vec1)*np.linalg.norm(Vec2)*np.sin(theta)
        #CrossProduct for CW/CCW
        CrossProduct = np.cross(Vec1, Vec2)
        if (CrossProduct >= 0):
            RotationalHashMap["Angle"] += theta
            AngleVector.append(RotationalHashMap["Angle"])
            CCWAngleVector.append(0)
            if RotationalHashMap["Angle"] >= CriticalAngle:
                RotationalHashMap["Angle"] = 0
                RotationalHashMap["CW"] += 1
        else:
            RotationalHashMap["CCWAngle"] += theta
            CCWAngleVector.append(RotationalHashMap["CCWAngle"])
            AngleVector.append(RotationalHashMap["Angle"])
            if RotationalHashMap["CCWAngle"] >= CriticalAngle:
                RotationalHashMap["CCWAngle"] = 0
                RotationalHashMap["CCW"] += 1
    return(RotationalHashMap, AngleVector, CCWAngleVector)


def circlingBehavior2(VectorList, CriticalAngle):
    RotationalHashMap = {"Angle":0, "CCWAngle":0, "CW":0, "CCW":0, "PartialCW":0, "PartialCCW":0}
    AngleVector = []
    CCWAngleVector = []
    for Vectors in VectorList:
        if Vectors[0] != Vectors[1]:
            theta = math.degrees(math.acos((np.dot(Vectors[0], Vectors[1]))/((np.linalg.norm(Vectors[0]))*(np.linalg.norm(Vectors[1])))))
        else:
            theta = 0.0
        #Lagrange Identity for CW/CCW
        Lagrange = np.linalg.norm(Vectors[0])*np.linalg.norm(Vectors[1])*np.sin(theta)
        #CrossProduct for CW/CCW
        CrossProduct = np.cross(Vectors[0], Vectors[1])
        if (CrossProduct >= 0):
            RotationalHashMap["Angle"] += theta
            AngleVector.append(RotationalHashMap["Angle"])
            CCWAngleVector.append(RotationalHashMap["CCWAngle"])
            if RotationalHashMap["Angle"] >= CriticalAngle:
                RotationalHashMap["Angle"] = 0
                RotationalHashMap["CW"] += 1
        else:
            RotationalHashMap["CCWAngle"] += theta
            CCWAngleVector.append(RotationalHashMap["CCWAngle"])
            AngleVector.append(RotationalHashMap["Angle"])
            if RotationalHashMap["CCWAngle"] >= CriticalAngle:
                RotationalHashMap["CCWAngle"] = 0
                RotationalHashMap["CCW"] += 1
    print(RotationalHashMap)
    
    mp.plot(np.array(AngleVector[10000:12000]))
    mp.xlabel("Frame Index")
    mp.ylabel("Angle-theta, Degrees")
    mp.title("Consecutive frame Midpoint-Head vector angles")
    mp.show()

    mp.plot(np.array(CCWAngleVector[27000:28000]))
    mp.xlabel("Frame Index")
    mp.ylabel("Angle-theta, Degrees")
    mp.title("Consecutive frame Midpoint-Head vector angles")
    mp.show()

    breakpoint()
    FFT = fft(np.array(np.cos(AngleVector[26500:28000])))
    mp.plot(FFT)
    mp.xlabel("Frame Index")
    mp.ylabel("fourier-transformed, frequency")
    mp.title("fourier-transform of cicling data")
    mp.show()
    return(RotationalHashMap, AngleVector, CCWAngleVector)

from scipy import fftpack
def circlingBehaviour3(Midpoints, VectorList, MaxY, MaxX, CriticalAngle):
    VectorFunction_YVal = lambda midpoints, Val: [[0, Val - Vectors[1]] for Vectors in midpoints]
    VectorFunction_XVal = lambda midpoints, Val: [[Val - Vectors[0], 0] for Vectors in midpoints]
    MaxYVectors = VectorFunction_YVal(midpoints=Midpoints, Val=MaxY)
    MinYVectors = VectorFunction_YVal(midpoints=Midpoints, Val=0)
    MaxXVectors = VectorFunction_XVal(midpoints=Midpoints, Val=MaxX)
    MinXVectors = VectorFunction_XVal(midpoints=Midpoints, Val=0)
    RotationalHashMap = {"CWAngle":0, "CCWAngle":0, "CW":0, "CCW":0, "PartialCW":0, "PartialCCW":0}
    
    AngleList = []
    PartialAngles = []
    CWTurns = []
    RealChange = []
    # Counter = 0
    # print(MaxYVectors[27000:27500])
    # print(VectorList[27000:27500])
    # rotationX = 0
    # rotationY = 500
    # mp.plot([i[0] for i in MaxYVectors[rotationX:rotationY]], [i[1] for i in MaxYVectors[rotationX:rotationY]], color="black")
    # # mp.plot([i[0] for i in MinYVectors[rotationX:rotationY]], [i[1] for i in MinYVectors[rotationX:rotationY]], color="green")
    # # mp.plot([i[0] for i in MaxXVectors[rotationX:rotationY]], [i[1] for i in MaxXVectors[rotationX:rotationY]], color="orange")
    # # mp.plot([i[0] for i in MinXVectors[rotationX:rotationY]], [i[1] for i in MinXVectors[rotationX:rotationY]], color="blue")
    # mp.plot([i[0] for i in VectorList[rotationX:rotationY]], [i[1] for i in VectorList[rotationX:rotationY]], color="red")
    # mp.show()
    # breakpoint()
    for Vectors, MaxVec, MinVecY in zip(VectorList, MinYVectors, MinYVectors):
        Cross = np.cross(MaxVec, Vectors)
        Cross_Min = np.cross(MinVecY, Vectors)
        theta = math.degrees(math.acos((np.dot(MaxVec, Vectors))/((np.linalg.norm(MaxVec))*(np.linalg.norm(Vectors)))))
        if ((Cross > 0)):
            # print(Vectors, MaxVec, theta)
            # time.sleep(0.2)
            AngleList.append(theta)
        elif ((Cross < 0)):
            print(Vectors, MaxVec, theta)
            time.sleep(0.2)
            Phi = 360 - theta
            AngleList.append(Phi)
        elif Cross == 0:
            AngleList.append(0)
           
    # print(AngleList[0:50])    
    # breakpoint()   
        
    for Theta1, Theta2 in zip(AngleList[:-1], AngleList[1:]):
        if (((Theta1 >= CriticalAngle) and (Theta2 < CriticalAngle))
            and (np.cross(VectorList[AngleList.index(Theta1)], VectorList[AngleList.index(Theta2)]) > 0)):
            RotationalHashMap["CW"] += 1
            
    for Theta1, Theta2 in zip(AngleList[:-1], AngleList[1:]):
        if (((Theta1 >= CriticalAngle) and (Theta2 < CriticalAngle))
            and (np.cross(VectorList[AngleList.index(Theta1)], VectorList[AngleList.index(Theta2)]) > 0)):
            print(Theta1, Theta2, AngleList.index(Theta1))
        # if Angles >= CriticalAngle:
        #     RotationalHashMap["CW"] += 1
    
    for Theta1, Theta2 in zip(AngleList[:-1], AngleList[1:]):
        #Introduce Angular velocity component, cannot exceed too large d(Theta)/dT
        if ((Theta1 > Theta2) and ((Theta1 - Theta2) < 350)):
            RotationalHashMap["CCWAngle"] += (Theta1 - Theta2)
            if RotationalHashMap["CCWAngle"] >= CriticalAngle:
                print(AngleList.index(Theta2), Theta1, Theta2)
                RotationalHashMap["CCWAngle"] = 0
                RotationalHashMap["CCW"] += 1
        else:
            if RotationalHashMap["CCWAngle"] > 0:
                RotationalHashMap["CCWAngle"] -= (Theta2 - Theta1)
            elif RotationalHashMap["CCWAngle"] < 0:
                RotationalHashMap["CCWAngle"] == 0
                
         
    Angles = [np.cos(math.radians(i)) for i in AngleList]
    Direction_X = [i[0] for i in VectorList[0:5000]]
    Direction_Y = [i[1] for i in VectorList[0:5000]]
    NormalizedDirection_X = [((2*((j-min(Direction_X))/(max(Direction_X) - min(Direction_X)))) - 1) for j in Direction_X]
    NormalizedDirection_Y = [((2*((j-min(Direction_Y))/(max(Direction_Y) - min(Direction_Y)))) - 1) for j in Direction_Y]
    mp.plot(AngleList, label = "Angles")
    # mp.plot(NormalizedDirection_X, color="red", label="X_values")
    # mp.plot(NormalizedDirection_Y, color = "black", label = "Y_values")
    mp.xlabel("Frame Index")
    mp.ylabel("Angle-theta, sin(radians)")
    mp.title("Consecutive frame Midpoint-Head vector & North-vector angles")
    mp.show()

    print(RotationalHashMap)
    breakpoint()
    print(AngleList[27000:28000])
    mp.plot([i for i in range(13450, 13550)], AngleList[13450:13550])
    mp.xlabel("Time (seconds)")
    mp.ylabel("Angles (degrees)")
    mp.show()
    
    mp.plot(Direction_X)
    mp.plot(Direction_Y, color="orange")
    mp.show()
    
    coslist = [np.sin(math.radians(1440 - i)) for i in range(0, 1440)]
    mp.plot(coslist)
    mp.show()


    FFT = fftpack.fft(coslist)
    Freq = fftpack.fftfreq(len(coslist))
    fig, ax = mp.subplots()
    ax.stem(Freq, np.abs(FFT))
    breakpoint() 
    return(RotationalHashMap)
   

    # mp.plot([i[0] for i in VectorList[27000:28000]], [j[1] for j in VectorList[27000:28000]])
    # mp.show()

    # print(RotationalHashMap)
    # mp.plot(np.array(AngleVector[27000:28000]))
    # mp.xlabel("Frame Index")
    # mp.ylabel("Angle-theta, Degrees")
    # mp.title("Consecutive frame Midpoint-Head vector angles")
    # mp.show()

    # mp.plot(np.array(CCWAngleVector))
    # mp.xlabel("Frame Index")
    # mp.ylabel("Angle-theta, Degrees")
    # mp.title("Consecutive frame Midpoint-Head vector angles")
    # mp.show()

    # breakpoint()
    # FFT = fft(np.array(np.cos(AngleVector[26500:28000])))
    # mp.plot(FFT)
    # mp.xlabel("Frame Index")
    # mp.ylabel("fourier-transformed, frequency")
    # mp.title("fourier-transform of cicling data")
    # mp.show()
    
    
    def rotationQuantifier(PositionVecX, PositionVecY, MaxY, MaxX, CriticalAngle):
        DirectionVectorsFxn = lambda PositionVec1, PositionVec2: [[V2 - V1  for V1, V2 in zip(Vectors1, Vectors2)] for Vectors1, Vectors2 in zip(PositionVec1, PositionVec2)]
        MaxVectorFunction = lambda StartPosition, EndPosition: [[EndVecs[0] - StartVecs[0], EndVecs[1] - StartVecs[1]] for StartVecs, EndVecs in zip(StartPosition, EndPosition)]
        DirectionVectors = DirectionVectorsFxn(PositionVecX, PositionVecY)
        RotationalHashMap = {"CWAngle":0, "CCWAngle":0, "CW":0, "CCW":0.000, "PartialCW":0, "PartialCCW":0}
        
        #MaxVectorFunction(PositionVecX, plotMaxVec_YMax)
        plotMaxVec_YMax = [[DVectors[0], 0]for DVectors in PositionVecX]
        plotMaxVec_YMin = [[DVectors[0], MaxY]for DVectors in PositionVecX]
        plotMaxVec_XMax = [[MaxX, DVectors[1]]for DVectors in PositionVecX]
        plotMaxVec_XMin = [[0, DVectors[1]]for DVectors in PositionVecX]
        MaxVectors = MaxVectorFunction(PositionVecX, plotMaxVec_YMax)


        AngleList = []
        RotationalMotionCW = []
        RotationalMotionCCW = []
        for V1, V2 in zip(MaxVectors, DirectionVectors):
            #Overcome the cross product by crossing the body-head vectors with the maximum vector
            Cross = np.cross(V1, V2)
            Theta = math.degrees(math.acos((np.dot(V2, V1))/((np.linalg.norm(V2))*(np.linalg.norm(V1)))))
            if Cross > 0:
                AngleList.append(Theta)
            elif Cross < 0:
                Phi = 360 - Theta
                AngleList.append(Phi)
                
        # for Theta1, Theta2 in zip(AngleList[:-1], AngleList[1:]):
        #     Notice:
        #         that if Theta2 > Theta1 is defined then when theta2 is < Critical angle
        #         it will be treated as a CCW rotation.
                
        #         So you cannot define the CW motion like this, however, it should be enough to 
        #         do it as you were with the various conditions. On the CCW motion you should use Theta1 < Theta2
        #     if (Theta2 > Theta1):
        #         if (Theta1 >= CriticalAngle) and (Theta2 < CriticalAngle):
        #             print(AngleList.index(Theta1))
        #     elif (Theta1 > Theta2):
        #         pass
            
        # for Theta1, Theta2 in zip(AngleList[:-1], AngleList[1:]):
        #     Rotation = Theta1/CriticalAngle
        #     if Rotation < CriticalAngle/360:
        #         RotationalMotionCW.append(RotationalHashMap["CW"] + Rotation)
        #     elif ((Rotation >= CriticalAngle/360) 
        #           and ((np.cross(DirectionVectors[AngleList.index(Theta1)], DirectionVectors[AngleList.index(Theta2)]) > 0))):
        #         RotationalHashMap["CW"] += 1
        #         RotationalMotionCW.append(RotationalHashMap["CW"] + Rotation)
        AngleIndex = 0       
        for Theta1, Theta2 in zip(AngleList[:-1], AngleList[1:]):
            if ((Theta1/CriticalAngle >= CriticalAngle/360) and (Theta2/CriticalAngle < 0.25)
                and ((AngleList.index(Theta1) - AngleIndex) > 30)):
                #and (np.cross(DirectionVectors[AngleList.index(Theta1)], DirectionVectors[AngleList.index(Theta2)])>0)):
                #add clause not within 10 or so frames of each other
                    RotationalHashMap["CW"] += 1
                    print(RotationalHashMap["CW"])
                    RotationalMotionCW.append(RotationalHashMap["CW"])
                    AngleIndex = AngleList.index(Theta1)
            else:
                RotationalMotionCW.append(RotationalHashMap["CW"] + Theta1/CriticalAngle)
            
        # Compute the CW and CCW rotations respectively        
        # for Theta1, Theta2 in zip(AngleList[:-1], AngleList[1:]):
        #     if ((((Theta1 >= CriticalAngle) and (Theta2 < CriticalAngle))
        #         and (np.cross(DirectionVectors[AngleList.index(Theta1)], DirectionVectors[AngleList.index(Theta2)]) > 0)) 
        #         and (Theta1/CriticalAngle >= 0.95)):
        #         # print(Theta1, Theta2, AngleList.index(Theta1))
        #         RotationalHashMap["CW"] += 1
        #         RotationalMotionCW.append(RotationalHashMap["CW"])
        #     elif Theta1 > Theta2 and Theta1 >= CriticalAngle:
        #         print(Theta1, Theta2, AngleList.index(Theta1))
        #     else:
        #         RotationalMotionCW.append(RotationalHashMap["CW"] + Theta1/CriticalAngle)
        # # mp.plot(AngleList)
        # mp.show()
        # print(RotationalHashMap)
        # breakpoint()
        
        for Theta1, Theta2 in zip(AngleList[:-1], AngleList[1:]):
            #Introduce Angular velocity component, cannot exceed too large d(Theta)/dT
            if ((Theta1 > Theta2) and ((Theta1 - Theta2) < CriticalAngle)):
                if RotationalHashMap["CCWAngle"] >= CriticalAngle:
                    RotationalHashMap["CCWAngle"] = 0
                    RotationalHashMap["CCW"] += 1
                    RotationalMotionCCW.append(RotationalHashMap["CCW"])
                else:
                    RotationalHashMap["CCWAngle"] += (Theta1 - Theta2)
                    RotationalMotionCCW.append(RotationalHashMap["CCW"] + (Theta1 - Theta2)/360)
            else:
                RotationalMotionCCW.append(RotationalHashMap["CCW"])
                if RotationalHashMap["CCWAngle"] > 0:
                    RotationalHashMap["CCWAngle"] -= (Theta2 - Theta1)
                elif RotationalHashMap["CCWAngle"] < 0:
                    RotationalHashMap["CCWAngle"] == 0
                    
        print(RotationalHashMap)
        return(RotationalMotionCW, RotationalMotionCCW, DirectionVectors, plotMaxVec_YMax,
               plotMaxVec_YMin, plotMaxVec_XMax, plotMaxVec_XMin)
#####################################################################
#Jan 22, 2022 - EDFunctions predictor function V1
#####################################################################

    #Scale
    ScaleVector = [[] for _ in range(len(DirectionVectors))]
    for Labels1, Labels2, Ind in zip(Displacement, DirectionVectors, range(len(DirectionVectors))):
        for V1, V2 in zip(Labels1, Labels2):
            print(V1, V2)
            time.sleep(0.05)
            Scale = [X + Y for X, Y in zip(V1, V2)]
            ScaleVector[Ind].append(Scale)
    
    DirectionVector_Dict = {LabelsFrom[Ind]:DirectionVectors[Ind] for Ind in range(len(DirectionVectors))}
    DisplacementVector_Dict = {LabelsFrom[Ind]:Displacement[Ind] for Ind in range(len(Displacement))}
    ScaleVector_Dict = {LabelsFrom[Ind]:ScaleVector[Ind] for Ind in range(len(ScaleVector))}

    OldColumns = list(DataFrame.columns.values)
    NewColumns = list(ExtractHighPVal.columns.values)
    DataFrame = DataFrame.rename(columns={OldColumns[Ind]: NewColumns[Ind] for Ind in range(len(OldColumns))})
    LabelToUse = []
    Ind = 0
    CountNone = 0
    while Ind < len(DataFrame.index.values):
        Check = False
        for Labels in LabelsFrom:
            if ((DataFrame[f"{Labels}p-val"][Ind] >= CutOff)
                and (str(ScaleVector_Dict[Labels][Ind][0]) != "nan")):
                LabelToUse.append(Labels)
                Check = True
                break
        if Check == False:
            LabelToUse.append("None")
            CountNone += 1
        Ind += 1
    print(CountNone)
    #Predict the location of the head label by adding the availble surrounding label displacement vector
    #With the last available surrounding label - target label direction vector and scale the last available 
    #surrounding label 
    for Ind in DataFrame.index.values:
        if ((DataFrame[f"{PredictLabel}p-val"][Ind] < CutOff)):
            print(DataFrame[f"{PredictLabel}p-val"][Ind], LabelToUse[Ind])
            time.sleep(0.5)
            

    for Ind in DataFrame.index.values:
        if ((DataFrame[f"{PredictLabel}p-val"][Ind] < CutOff) and (LabelToUse[Ind] != "None")):
            DataFrame[f"{PredictLabel}_x"][Ind] = DataFrame[f"{LabelToUse[Ind]}_x"][Ind] + DirectionVector_Dict[LabelToUse[Ind]][Ind][0]
            DataFrame[f"{PredictLabel}_y"][Ind] = DataFrame[f"{LabelToUse[Ind]}_y"][Ind] + DirectionVector_Dict[LabelToUse[Ind]][Ind][1]
            DataFrame[f"{PredictLabel}p-val"][Ind] = 1.5
    DataFrame = DataFrame.rename(columns={NewColumns[Ind]: OldColumns[Ind] for Ind in range(len(NewColumns))})
    DataFrame.to_csv(r"F:\WorkFiles_XCELLeration\Video\2minTrim_end\Corrected2.csv")
    
    """

"""
    #Get the average geometric vector between ears and head 
    #given the p-val > cutoff
    ExtractHighPVal = copy.copy(DataFrame)
    Cols = 3
    while Cols <= max(ExtractHighPVal.columns.values):
        Query = [i for i in range(Cols-2, Cols+1)]
        ExtractHighPVal[Query] = ExtractHighPVal[Query].mask(pd.to_numeric(ExtractHighPVal[Cols], downcast="float") < CutOff).fillna(np.nan)
        Cols += 3    
    #Convert all values to float from string
    for Cols in ExtractHighPVal.columns.values:
        ExtractHighPVal[Cols] = pd.to_numeric(ExtractHighPVal[Cols], downcast="float")
        DataFrame[Cols] = pd.to_numeric(DataFrame[Cols], downcast="float")
    #RenameCols (optimize later)
    FeatureList = ["_x", "_y", "p-val"]
    Ind = 0
    Ind2 = 0
    for Cols in ExtractHighPVal.columns.values:
        if Ind <= 2:
            ExtractHighPVal = ExtractHighPVal.rename(columns={Cols:f"{colNames[Ind2]}{FeatureList[Ind]}"})
            if Ind == 2:
                Ind = 0
                Ind2 += 1
            else:
                Ind += 1
    PositionVectors = lambda XCoords, YCoords: [[X, Y] for X, Y in zip(XCoords, YCoords)]
    VectorFunction = lambda StartVec, EndVec: [[Y - X for X, Y in zip(Vec1, Vec2) 
                                                if X != np.nan and Y != np.nan] 
                                               for Vec1, Vec2 in zip(StartVec, EndVec)]
    PredictLabelPosVec = PositionVectors(XCoords=ExtractHighPVal[f"{PredictLabel}_x"], YCoords=ExtractHighPVal[f"{PredictLabel}_y"])
    FromLabelPosVec = [PositionVectors(XCoords=ExtractHighPVal[f"{Label}_x"], YCoords=ExtractHighPVal[f"{Label}_y"]) for Label in LabelsFrom]
    #direction vectors from the surrounding labels and the label to predict
    DirectionVectors = [VectorFunction(FromLabels, PredictLabelPosVec) for FromLabels in FromLabelPosVec]
    #displacement vectors of the sorrounding labels as they change per frame
    Displacement = [[] for _ in range(len(DirectionVectors))]
    for Ind, Labels in enumerate(FromLabelPosVec):
        for V1, V2 in zip(Labels[:-1], Labels[1:]):
            if len(Displacement[Ind]) == 0:
                Displacement[Ind].append([np.nan, np.nan])
            Function = [J2 - J1 for J1, J2 in zip(V1, V2)]
            Displacement[Ind].append(Function)
            
    DirectionVectorsDict = {LabelsFrom[Ind]: DirectionVectors[Ind] for Ind in range(len(DirectionVectors))}
    DisplacementVectorsDict = {LabelsFrom[Ind]: Displacement[Ind] for Ind in range(len(Displacement))}
    OldColumns = list(DataFrame.columns.values)
    NewColumns = list(ExtractHighPVal.columns.values)
    DataFrame = DataFrame.rename(columns={OldColumns[Ind]: NewColumns[Ind] for Ind in range(len(OldColumns))})
    LabelToUse = []
    Ind = 0
    while Ind < len(DataFrame.index.values):
        Check = False
        for Labels in LabelsFrom:
            if ((DataFrame[f"{Labels}p-val"][Ind] >= CutOff) 
                and (str(DisplacementVectorsDict[Labels][Ind][0]) != "nan")
                and (str(DirectionVectorsDict[Labels][Ind - 1][0]) != "nan")):
                LabelToUse.append(Labels)
                Check = True
                break
        if Check == False:
            LabelToUse.append("None")
        Ind += 1
    Counter = 0
    """
    Direction vectors at index Ind will always be 0 since the head label itself is absent
    """
    for Ind in DataFrame.index.values:
        if ((DataFrame[f"{PredictLabel}p-val"][Ind] < CutOff) and (LabelToUse[Ind] != "None")):
            Scale = [I1 + I2 for I1, I2 in zip(DisplacementVectorsDict[LabelToUse[Ind]][Ind], DirectionVectorsDict[LabelToUse[Ind]][Ind - 1])]
            DataFrame[f"{PredictLabel}_x"][Ind] = DataFrame[f"{LabelToUse[Ind]}_x"][Ind] + Scale[0]
            DataFrame[f"{PredictLabel}_y"][Ind] = DataFrame[f"{LabelToUse[Ind]}_y"][Ind] + Scale[1]
            DataFrame[f"{PredictLabel}p-val"][Ind] = 1.5
        elif ((DataFrame[f"{PredictLabel}p-val"][Ind] < CutOff) and (LabelToUse[Ind] == "None")):
            print([DataFrame[f"{Label}p-val"][Ind] for Label in LabelsFrom])
            Counter += 1
    #DataFrame.to_csv(r"F:\WorkFiles_XCELLeration\Video\2minTrim_end\Corrected3.csv")
    DataFrame = DataFrame.rename(columns={NewColumns[Ind]: OldColumns[Ind] for Ind in range(len(NewColumns))})



"""


















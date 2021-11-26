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

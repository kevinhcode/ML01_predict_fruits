#!/usr/bin/python
# -*- coding: utf-8 -*-

# Based on Youtube Tutorial
# https://www.youtube.com/watch?v=cKxRvEZd3Mw
# Hello World - Machine Learning Recipes #1
# by Google Developers
#
# Modified by Kevin H for learning use (2019-12-22)

"""
Training data set:
Weight 	Length 	Color	Label
131	63	1	"Orange"
150	60	1	"Orange"
175	59	1	"Orange"
120	70	1	"Orange"
450	280	1	"Papaya"
460	250	1	"Papaya"
500	320	2	"Papaya"
550	325	2	"Papaya"
Note:
1. Weight in "gram"
2. Height in "mm"
3. Color: 1="Orange Color", 2="Green Color"
"""
print(__doc__)
import time
from sklearn import tree

start_time = time.time()
features = [[131, 63, 1],[150, 60, 1], [175, 59, 1], [120, 70, 1],
            [450, 280, 1],[460, 250, 1],[500, 320, 2], [550, 325, 2]]
labels = ['orange','orange','orange','orange','papaya','papaya','papaya','papaya']
print ("running Scikit Learn Decision Tree Classifier")
clfy = tree.DecisionTreeClassifier()
clfy = clfy.fit(features, labels)
print ("new data: 133, 70, 1")
print ("Prediction = ", clfy.predict([[133,70,1]]))
end_time = time.time()
print ("Total running time: {}".format(end_time - start_time))


        



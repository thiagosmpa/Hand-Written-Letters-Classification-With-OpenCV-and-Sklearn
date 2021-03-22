import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
"""
Dictionary:
     0-A    1-B     2-C    3-D    4-E    5-F    6-G    7-H    8- I   9-J 
    10-K   11-L    12-M   13-N   14-O   15-P   16-Q   17-R    18-S  19-T   
    20-U   21-V    22-W   23-X   24-Y   25-Z   
    26 - zero    27 - one    28 - two  29 - three   30 - four
    31 - five    32 - six   33 - seven 34 - eight   35 - nine
"""

train = pd.read_csv('trainingSet.csv')
test = pd.read_csv('testSet.csv')

trainSet = train[['NumPixels', 'ContourArea', 'ContourPerim', 'CircleArea',
       'LetterHeight', 'LetterWeight', 'SpaceLeft', 'SpaceBottom',
       'SpaceRight', 'SpaceTop', 'RotatingRectH', 'RotatingRectW',
       'RotatingRectAngle']]

testSet = test[['NumPixels', 'ContourArea', 'ContourPerim', 'CircleArea',
       'LetterHeight', 'LetterWeight', 'SpaceLeft', 'SpaceBottom',
       'SpaceRight', 'SpaceTop', 'RotatingRectH', 'RotatingRectW',
       'RotatingRectAngle']]


XTr = np.asarray(trainSet)
XTe = np.asarray(testSet)

yTr = np.asarray(train['Class'])
yTe = np.asarray(test['Class'])


# =============================================================================
# SVM
# =============================================================================

classifier = svm.SVC(kernel='linear', gamma = 'auto', C = 2)
classifier.fit(XTr, yTr)

y_predict_svm = classifier.predict(XTe)

print('SVM Classification Report: ')
print(classification_report(yTe,  y_predict_svm))


# =============================================================================
# DECISION TREE
# =============================================================================

clf = tree.DecisionTreeClassifier()
clf = clf.fit(XTr, yTr)

y_predict_tree = clf.predict(XTe)

print('Decision Trees Classification Report: ')
print(classification_report(yTe,  y_predict_tree))

y_predicted = confusion_matrix(yTe, y_predict_svm)

"""
Main step: Extract features from images
Choose the classifier: Decision Forest, SVM
"""
import cv2
import numpy as np
import math
import csv


alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
            'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 
            'eight', 'nine']



data_size = 20                                    # Number of samples per letter
train_size = int(data_size * 0.6)                       # Training set size
test_size = 20 - train_size                             # Test set size







# =============================================================================
# FEATURES
# =============================================================================
def lineAngle(a, cnt):
    rows,cols = a.shape[:2]
    [vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
    lefty = int((-x*vy/vx) + y)
    righty = int(((cols-x)*vy/vx)+y)
    try: 
        line = cv2.line(a,(cols-1,righty),(0,lefty),(255,255,255),2)
    except OverflowError:
        return 90
        
    (_, _, _, maxLoc1) = cv2.minMaxLoc(line[199,:])
    if (maxLoc1[1] > 0):                    # Line on the bottom
        maxLoc1 = np.array(maxLoc1).astype(np.ndarray)
        maxLoc1[0] = 199
        
    else:                                   # Line in the left margin
        (_, _, _, maxLoc1) = cv2.minMaxLoc(line[:,0])
        maxLoc1 = np.array(maxLoc1).astype(np.ndarray)
        maxLoc1[0] = maxLoc1[1]
        maxLoc1[1] = 0
        
    (_, _, _, maxLoc2) = cv2.minMaxLoc(line[0,:])
    if (maxLoc2[1] > 0):                    # Line on the top
        maxLoc2 = np.array(maxLoc2).astype(np.ndarray)
        maxLoc2[0] = 0
        
    else:                                   # Line in the right margin
        (_, _, _, maxLoc2) = cv2.minMaxLoc(line[:,199])
        maxLoc2 = np.array(maxLoc2).astype(np.ndarray)
        maxLoc2[0] = maxLoc2[1]
        maxLoc2[1] = 199
    
    weightL = maxLoc2[1] - maxLoc1[1]
    heightL = maxLoc1[0] - maxLoc2[0]
    
    angle = abs(math.atan2(heightL, weightL) * 180 / math.pi)   
    return angle

def rotatingRect(cnt):
    rect = cv2.minAreaRect(cnt)
    (x,y), (deltax, deltay), theta = rect
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    # cv2.drawContours(a,[box],0,(255,255,255),2)
    return deltax, deltay

def Rectangle(cnt):
    x,y,w,h = cv2.boundingRect(cnt)
    # cv2.rectangle(a,(x,y),(x+w,y+h),(255,255,255),2)
    return w,h

def Circle(cnt):
    (x,y),radius = cv2.minEnclosingCircle(cnt)
    center = (int(x),int(y))
    radius = int(radius)
    # cv2.circle(a,center,radius,(255,255,255),2)
    return center, radius

def Profile(rectx, recty, weightRect, heightRect, a):
    profile = np.zeros(4)
    for j in range(recty, recty + heightRect):
        for i in range(rectx, rectx + weightRect):
            try:
                if (a[j,i] != 0):
                    break
                profile[0] = profile[0] + 1
            except IndexError:
                continue
    for i in range(rectx, rectx + weightRect):
        for j in range(recty + heightRect, recty, -1):
            try:
                if (a[j,i] != 0):
                    break
                profile[1] = profile[1] + 1
            except IndexError:
                continue
    for j in range(recty + weightRect, recty, -1):
        for i in range(rectx + weightRect, rectx, -1):
            try:
                if (a[j,i] != 0):
                    break
                profile[2] = profile[2] + 1
            except IndexError:
                continue
    for i in range(rectx + weightRect, rectx, -1):
        for j in range(recty, recty + heightRect):
            try:
                if (a[j,i] != 0):
                    break
                profile[3] = profile[3] + 1
            except IndexError:
                continue
    return profile

# =============================================================================
# NORMALIZING DATA
# =============================================================================

def normalizeTrain(train):
    for j in range(13):
        min_value.append(min(train[:,j]))
        max_value.append(max(train[:,j]))
        for i in range(len(train)):
            actual_value = train[i,j]
            train[i,j] = (actual_value - min_value[j]) / (max_value[j] - min_value[j])
    
    return train, max_value, min_value

def normalizeTest(test, max_value, mix_value):
    for j in range(13):
        for i in range(len(test)):
            actual_value = test[i,j]
            test[i,j] = (actual_value - min_value[j]) / (max_value[j] - min_value[j])
    return test


# =============================================================================
# MAIN - TRAIN SET
# =============================================================================

alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
            'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 
            'eight', 'nine']


kernel = np.ones((3,3),np.uint8)
# letter = 'f'
features = []


numberPixels = []
areaContours = []
perimeterContours = []
areaCircle = []
heightRect = []
weightRect = []
heightRotatingRect = []
weightRotatingRect = []
angle = []
profileLeft = []
profileBottom = []
profileRight = []
profileTop = []
letterClass = []


for letter in alphabet: 
    for i in range(12):# For the train set
        exec(f'{letter}{i} = cv2.imread("train/" + letter + "{i}.jpg")')
        exec(f'{letter}{i} = cv2.cvtColor({letter}{i}, cv2.COLOR_BGR2GRAY)')
        exec(f'_, {letter}{i} = cv2.threshold({letter}{i}, 240, 255, cv2.THRESH_BINARY)')    
        exec(f'contours, _ = cv2.findContours({letter}{i}, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)') 
        cnt = contours[len(contours) - 1]
        
        # First Feature: Count the number of black Pixels 
        exec(f'numberPixels.append(np.sum({letter}{i} == 255))')
        # Area of the contour
        areaContours.append(cv2.contourArea(cnt))
        # Perimeter of the contour
        perimeterContours.append(cv2.arcLength(cnt, True))
        # Area of the minimum Circle
        _, radius = Circle(cnt)
        areaCircle.append(math.pi * np.square(radius))
        # Rectangle
        rectx,recty,weightRect_,heightRect_ = cv2.boundingRect(cnt)
        heightRect.append(heightRect_)
        weightRect.append(weightRect_)
        # Profile (Black spaces between the bounding rectangle and the letter)
        # profile[0] = left spaces, profile[1] = bottom, profile[2] = right, profile[3] = top (clockwise)
        exec(f'profile = Profile(rectx, recty, weightRect_, heightRect_, {letter}{i})')
        profileLeft.append(profile[0])
        profileBottom.append(profile[1])
        profileRight.append(profile[2])
        profileTop.append(profile[3])
        # Rotating Rectangle
        weightRotatingRect_, heightRotatingRect_ = rotatingRect(cnt)
        heightRotatingRect.append(heightRotatingRect_)
        weightRotatingRect.append(weightRotatingRect_)
        # Angle of the baseline of the letter with the horizontal axis
        exec(f'angle_ = lineAngle({letter}{i}, cnt)')
        angle.append(angle_)
        # Class
        letterClass_ = alphabet.index(letter)
        letterClass.append(letterClass_)
    
    
numberPixels = np.array(numberPixels).T
features.append(numberPixels)

areaContours = np.array(areaContours).T
features.append(areaContours)

perimeterContours = np.array(perimeterContours).T
features.append(perimeterContours)

areaCircle = np.array(areaCircle).T
features.append(areaCircle)

heightRect = np.array(heightRect).T
weightRect = np.array(weightRect).T
features.append(heightRect)
features.append(weightRect)

profileLeft = np.array(profileLeft).T
profileBottom = np.array(profileBottom).T
profileRight = np.array(profileRight).T
profileTop = np.array(profileTop).T
features.append(profileLeft)
features.append(profileBottom)
features.append(profileRight)
features.append(profileTop)

heightRotatingRect = np.array(heightRotatingRect).T
weightRotatingRect = np.array(weightRotatingRect).T
features.append(heightRotatingRect)
features.append(weightRotatingRect)

angle = np.array(angle).T
features.append(angle)

letterClass = np.array(letterClass).T
features.append(letterClass)

features = np.array(features).T






trainSet = np.array(features)
min_value = []
max_value = []
trainSet, max_value, min_value = normalizeTrain(trainSet)

# =============================================================================
# WRITING CSV
# =============================================================================

with open('trainingSet.csv', mode='w') as train_file:
    train_writer = csv.writer(train_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    train_writer.writerow((['NumPixels', 'ContourArea', 'ContourPerim', 'CircleArea','LetterHeight', 
                            'LetterWeight', 'SpaceLeft', 'SpaceBottom', 'SpaceRight','SpaceTop', 'RotatingRectH',
                            'RotatingRectW', 'RotatingRectAngle', 'Class']))
    for i in range(len(trainSet)):
            train_writer.writerow(([trainSet[i,0], trainSet[i,1], trainSet[i,2], trainSet[i,3],
                                    trainSet[i,4], trainSet[i,5], trainSet[i,6], trainSet[i,7],
                                    trainSet[i,8], trainSet[i,9], trainSet[i,10],trainSet[i,11], trainSet[i,12], trainSet[i,13]]))
            
            

# =============================================================================
# MAIN - TEST SET
# =============================================================================
            
            
features = []


numberPixels = []
areaContours = []
perimeterContours = []
areaCircle = []
heightRect = []
weightRect = []
heightRotatingRect = []
weightRotatingRect = []
angle = []
profileLeft = []
profileBottom = []
profileRight = []
profileTop = []
letterClass = []
            
for letter in alphabet: 
    for i in range(test_size, data_size):# For the test set
        exec(f'{letter}{i} = cv2.imread("test/" + letter + "{i}.jpg")')
        exec(f'{letter}{i} = cv2.cvtColor({letter}{i}, cv2.COLOR_BGR2GRAY)')
        exec(f'_, {letter}{i} = cv2.threshold({letter}{i}, 240, 255, cv2.THRESH_BINARY)')    
        exec(f'contours, _ = cv2.findContours({letter}{i}, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)') 
        cnt = contours[len(contours) - 1]
        
        # First Feature: Count the number of black Pixels 
        exec(f'numberPixels.append(np.sum({letter}{i} == 255))')
        # Area of the contour
        areaContours.append(cv2.contourArea(cnt))
        # Perimeter of the contour
        perimeterContours.append(cv2.arcLength(cnt, True))
        # Area of the minimum Circle
        _, radius = Circle(cnt)
        areaCircle.append(math.pi * np.square(radius))
        # Rectangle
        rectx,recty,weightRect_,heightRect_ = cv2.boundingRect(cnt)
        heightRect.append(heightRect_)
        weightRect.append(weightRect_)
        # Profile (Black spaces between the bounding rectangle and the letter)
        # profile[0] = left spaces, profile[1] = bottom, profile[2] = right, profile[3] = top (clockwise)
        exec(f'profile = Profile(rectx, recty, weightRect_, heightRect_, {letter}{i})')
        profileLeft.append(profile[0])
        profileBottom.append(profile[1])
        profileRight.append(profile[2])
        profileTop.append(profile[3])
        # Rotating Rectangle
        weightRotatingRect_, heightRotatingRect_ = rotatingRect(cnt)
        heightRotatingRect.append(heightRotatingRect_)
        weightRotatingRect.append(weightRotatingRect_)
        # Angle of the baseline of the letter with the horizontal axis
        exec(f'angle_ = lineAngle({letter}{i}, cnt)')
        angle.append(angle_)
        # Class
        letterClass_ = alphabet.index(letter)
        letterClass.append(letterClass_)
    
numberPixels = np.array(numberPixels).T
features.append(numberPixels)

areaContours = np.array(areaContours).T
features.append(areaContours)

perimeterContours = np.array(perimeterContours).T
features.append(perimeterContours)

areaCircle = np.array(areaCircle).T
features.append(areaCircle)

heightRect = np.array(heightRect).T
weightRect = np.array(weightRect).T
features.append(heightRect)
features.append(weightRect)

profileLeft = np.array(profileLeft).T
profileBottom = np.array(profileBottom).T
profileRight = np.array(profileRight).T
profileTop = np.array(profileTop).T
features.append(profileLeft)
features.append(profileBottom)
features.append(profileRight)
features.append(profileTop)

heightRotatingRect = np.array(heightRotatingRect).T
weightRotatingRect = np.array(weightRotatingRect).T
features.append(heightRotatingRect)
features.append(weightRotatingRect)

angle = np.array(angle).T
features.append(angle)

letterClass = np.array(letterClass).T
features.append(letterClass)
features = np.array(features).T




testSet = np.array(features)
testSet = normalizeTest(testSet, max_value, min_value)


# =============================================================================
# WRITING CSV
# =============================================================================


with open('testSet.csv', mode='w') as test_file:
    test_writer = csv.writer(test_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    test_writer.writerow((['NumPixels', 'ContourArea', 'ContourPerim', 'CircleArea', 'LetterHeight', 
                            'LetterWeight', 'SpaceLeft', 'SpaceBottom', 'SpaceRight','SpaceTop', 'RotatingRectH',
                            'RotatingRectW', 'RotatingRectAngle', 'Class']))
    for i in range(len(testSet)):
            test_writer.writerow(([testSet[i,0], testSet[i,1], testSet[i,2], testSet[i,3],
                                    testSet[i,4], testSet[i,5], testSet[i,6], testSet[i,7],
                                    testSet[i,8], testSet[i,9], testSet[i,10],testSet[i,11], testSet[i,12], testSet[i,13]]))
            
            
with open('testSet.csv', mode='w') as test_file:
    test_writer = csv.writer(test_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    test_writer.writerow((['NumPixels', 'ContourArea', 'ContourPerim', 'CircleArea', 'LetterHeight', 
                            'LetterWeight', 'SpaceLeft', 'SpaceBottom', 'SpaceRight','SpaceTop', 'RotatingRectH',
                            'RotatingRectW', 'RotatingRectAngle', 'Class']))
    for i in range(len(testSet)):
            test_writer.writerow(([testSet[i,0], testSet[i,1], testSet[i,2], testSet[i,3],
                                    testSet[i,4], testSet[i,5], testSet[i,6], testSet[i,7],
                                    testSet[i,8], testSet[i,9], testSet[i,10],testSet[i,11], testSet[i,12], testSet[i,13]]))


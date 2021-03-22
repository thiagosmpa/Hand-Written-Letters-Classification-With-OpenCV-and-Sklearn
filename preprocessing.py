import cv2
import numpy as np
import os
import csv


kernel = np.ones((3,3),np.uint8)
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
            'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 
            'eight', 'nine']



data_size = 20                                    # Number of samples per letter
train_size = int(data_size * 0.6)                       # Training set size
test_size = 20 - train_size                             # Test set size





# =============================================================================
# TRAIN SET
# =============================================================================

if (os.path.isdir('train') == False):
    os.mkdir('train')

    
for letter in alphabet: 
    for i in range(train_size):
        exec(f'{letter}{i} = cv2.imread("alphabet/" + letter + "/" + letter + "{i}.jpg")')
        exec(f'{letter}{i} = cv2.cvtColor({letter}{i}, cv2.COLOR_BGR2GRAY)')
        exec(f'th, {letter}{i} = cv2.threshold({letter}{i}, 200, 255,cv2.THRESH_OTSU)')
        exec(f'_, {letter}{i} = cv2.threshold({letter}{i}, th, 255, cv2.THRESH_BINARY_INV)')
        exec(f'{letter}{i} = cv2.dilate({letter}{i}, kernel, iterations = 1)')
        exec(f'{letter}{i} = cv2.erode({letter}{i}, kernel, iterations = 1)')
        number = str(i)
        exec(f'cv2.imwrite("train/" + letter + number + ".jpg", {letter}{i})')


# =============================================================================
# TEST SET
# =============================================================================

if (os.path.isdir('test') == False):
    os.mkdir('test')
    
for letter in alphabet: 
    for i in range(test_size, data_size):
        exec(f'{letter}{i} = cv2.imread("alphabet/" + letter + "/" + letter + "{i}.jpg")')
        exec(f'{letter}{i} = cv2.cvtColor({letter}{i}, cv2.COLOR_BGR2GRAY)')
        exec(f'th, {letter}{i} = cv2.threshold({letter}{i}, 200, 255,cv2.THRESH_OTSU)')
        exec(f'_, {letter}{i} = cv2.threshold({letter}{i}, th, 255, cv2.THRESH_BINARY_INV)')
        exec(f'{letter}{i} = cv2.dilate({letter}{i}, kernel, iterations = 1)')
        exec(f'{letter}{i} = cv2.erode({letter}{i}, kernel, iterations = 1)')
        number = str(i)
        exec(f'cv2.imwrite("test/" + letter + number + ".jpg", {letter}{i})')


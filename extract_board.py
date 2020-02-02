import cv2
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D


mask = 255 * np.ones((28,28), 'uint8')
mask[:, :3] = 0
mask[:3, :] = 0
mask[25:, :] = 0
mask[:, 25:] = 0

digit_mask = 255* np.ones((28,28), 'uint8')
digit_mask[:, :5] = 0
digit_mask[:5, :] = 0
digit_mask[23:, :] = 0
digit_mask[:, 23:] = 0



def load_model(wts_file):
    img_rows, img_cols = 28, 28
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu', input_shape=(img_rows, img_cols, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.load_weights(wts_file)
    
    return model


def extract_board(img_filepath, wts_file='model/model_wts.h5', remove_lines=False, he=2, hd=2, ve=2, vd=2):
    
    # Read image
    try:
        orig_img = cv2.imread(img_filepath, cv2.IMREAD_GRAYSCALE)
    except Exception as e:
        print("Image not found. Check file path")
        return
    
    aspect_ratio = orig_img.shape[0]/orig_img.shape[1]
    if(aspect_ratio>1 and orig_img.shape[0]>1000):
        orig_img = cv2.resize(orig_img, (1000, int(1000//aspect_ratio)))
    
    elif(aspect_ratio<=1 and orig_img.shape[1]>1000):
        orig_img = cv2.resize(orig_img, (int(1000//aspect_ratio), 1000))
    
    # Convert to grayscale
    orig_gray = cv2.GaussianBlur(orig_img, (7, 7), 0)
    orig_gray = cv2.adaptiveThreshold(orig_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C | cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)
    orig_gray = cv2.bitwise_not(orig_gray)


    (contours, _) = cv2.findContours(orig_gray.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea , reverse = True)

    # Board image
    x, y, w, h = cv2.boundingRect(contours[0])
    img = orig_img[y:y+h, x:x+w]
    img = cv2.resize(img, (252, 252))
    
    gray = cv2.GaussianBlur(img, (7, 7), 0)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C | cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)
    gray = cv2.bitwise_not(gray)

    # Remove grid lines from the board
    s=28
    if(eval(remove_lines)):
        hor = np.ones((1,7))
        ver = np.ones((7,1))

        h_lines = cv2.erode(gray, hor, iterations=he)
        h_lines = cv2.dilate(h_lines, hor, iterations=hd)

        v_lines = cv2.erode(gray, ver, iterations=ve)
        v_lines = cv2.dilate(v_lines, ver, iterations=vd)

        grid_lines = cv2.add(h_lines, v_lines)
        gray_no_grid = cv2.subtract(gray, grid_lines)
    
    else:
        gray_no_grid = gray.copy()
    
    '''
    plt.figure(figsize=(12, 12))
    plt.subplot(221)
    plt.imshow(orig_img)
    plt.axis('off')
    plt.subplot(222)
    plt.imshow(orig_gray, cmap='gray')
    plt.axis('off')
    plt.subplot(223)
    plt.imshow(gray, cmap='gray')
    plt.axis('off')
    plt.subplot(224)
    plt.imshow(gray_no_grid, cmap='gray')
    plt.axis('off')
    plt.show()
    '''
    
    # Detect digits
    model = load_model(wts_file)
    board = np.zeros((9,9), np.uint)
    for i in range(9):
        for j in range(9):
            sub_img = gray_no_grid[s*i:s*(i+1), s*j:s*(j+1)]
            sub_img = cv2.bitwise_and(sub_img, mask)
            
            #Check if not empty
            cen_img = cv2.bitwise_and(sub_img, digit_mask)
            if(np.sum(cen_img)> 0.1*18*18*255):    
                sub_img = sub_img.astype('float32')
                sub_img = sub_img/255
                pred = model.predict(sub_img.reshape(1, 28, 28, 1))
                label = np.argmax(pred)
                board[i,j] = label
    return board





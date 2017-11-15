import numpy as np
import cv2
import glob
import pickle


# Prepare object points, like (0, 0, 0), (1, 0, 0), (2, 0, 0) ...
objp = np.zeros((6*9, 3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

# Array to store object points and image points from all the images
# 3D points in real world space
objpoints = []
# 2D points in image plane
imgpoints = []

# Make a list of calibration images
images = glob.glob('./calibration*.jpg')

# Step through the list and search for chessboard
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    # Convert 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Search for corners in the grayscaled image
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    # if found, add object points, image points
    if ret == True:
        print('working on', fname)
        objpoints.append(objp)
        imgpoints.append(corners)

        cv2.drawChessboardCorners(img, (9, 6), corners, ret)
        write_name = 'corners_found' + str(idx) + '.jpg'
        cv2.imwrite(write_name, img)


img = cv2.imread('./calibration1.jpg')
img_size = (img.shape[1], img.shape[0])

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist

pickle.dump(dist_pickle, open("calibration_pickle.p", "wb"))

exit()

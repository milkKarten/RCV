import numpy as np
import cv2

def calibrate():
    try:
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((7*6,3), np.float32)
        objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        filename = "checkerboard.jpg"
        # filename = "checkerboards/Image6.tif"
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (7,6),None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)

            cv2.imshow('img',img)
            cv2.waitKey(1000)
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
            rot = cv2.Rodrigues(np.array(rvecs))[0]
            tvecs = np.array(tvecs)
            rot = np.append(rot, tvecs[0], axis=1)
            M = np.dot(mtx,rot)
            print("Camera matrix for image", filename,"\n",M)
            # Apply QR factorization to find M's components
            M = np.delete(M, 3, axis=1)
            Q, R = np.linalg.qr(np.linalg.inv(M))
            Rwc = np.linalg.inv(Q)  # Rotation
            K = np.linalg.inv(R)    # intrinsic
            # Apply SVD to M to find twc
            u, s, v = np.linalg.svd(mtx)
            twc = v[len(v)-1]
            Twc = np.c_[Rwc, twc]
            print("Twc:", Twc)
            print("K:", K)
            print()

        cv2.destroyAllWindows()
    except:
        print("File not found/Bad input data")
        return

if __name__ == "__main__":
    calibrate()

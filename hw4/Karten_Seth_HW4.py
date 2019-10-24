import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cv2

def draw2D(pts):
    xx = [pt[0,0] for pt in pts]
    xx.insert(0,0)
    yy = [pt[1,0] for pt in pts]
    yy.insert(0,0)
    lines = []
    lines.append(Line2D([xx[1], xx[2]], [yy[1], yy[2]]))
    lines.append(Line2D([xx[2], xx[3]], [yy[2], yy[3]]))
    lines.append(Line2D([xx[3], xx[4]], [yy[3], yy[4]]))
    lines.append(Line2D([xx[4], xx[1]], [yy[4], yy[1]]))
    lines.append(Line2D([xx[5], xx[6]], [yy[5], yy[6]]))
    lines.append(Line2D([xx[6], xx[7]], [yy[6], yy[7]]))
    lines.append(Line2D([xx[7], xx[8]], [yy[7], yy[8]]))
    lines.append(Line2D([xx[8], xx[5]], [yy[8], yy[5]]))
    lines.append(Line2D([xx[1], xx[5]], [yy[1], yy[5]]))
    lines.append(Line2D([xx[2], xx[6]], [yy[2], yy[6]]))
    lines.append(Line2D([xx[3], xx[7]], [yy[3], yy[7]]))
    lines.append(Line2D([xx[4], xx[8]], [yy[4], yy[8]]))

    lines.append(Line2D([xx[5],xx[9]], [yy[5], yy[9]]))
    lines.append(Line2D([xx[6],xx[9]], [yy[6], yy[9]]))
    lines.append(Line2D([xx[7],xx[9]], [yy[7], yy[9]]))
    lines.append(Line2D([xx[8],xx[9]], [yy[8], yy[9]]))
    return lines

def draw3D(pts, ax):
    xx = np.squeeze(np.asarray(pts.T[0]))
    xx=np.insert(xx,0,0)
    yy = np.squeeze(np.asarray(pts.T[1]))
    yy=np.insert(yy,0,0)
    zz = np.squeeze(np.asarray(pts.T[2]))
    zz=np.insert(zz,0,0)
    ax.plot([xx[1], xx[2]], [yy[1], yy[2]], [zz[1], zz[2]])
    ax.plot([xx[2], xx[3]], [yy[2], yy[3]], [zz[2], zz[3]])
    ax.plot([xx[3],xx[4]], [yy[3], yy[4]], [zz[3], zz[4]])
    ax.plot([xx[4],xx[1]], [yy[4], yy[1]], [zz[4], zz[1]])
    ax.plot([xx[5],xx[6]], [yy[5], yy[6]], [zz[5], zz[6]])
    ax.plot([xx[6],xx[7]], [yy[6], yy[7]], [zz[6], zz[7]])
    ax.plot([xx[7],xx[8]], [yy[7], yy[8]], [zz[7], zz[8]])
    ax.plot([xx[8],xx[5]], [yy[8], yy[5]], [zz[8], zz[5]])
    ax.plot([xx[1],xx[5]], [yy[1], yy[5]], [zz[1], zz[5]])
    ax.plot([xx[2],xx[6]], [yy[2], yy[6]], [zz[2], zz[6]])
    ax.plot([xx[3],xx[7]], [yy[3], yy[7]], [zz[3], zz[7]])
    ax.plot([xx[4],xx[8]], [yy[4], yy[8]], [zz[4], zz[8]])
    # Connect roof to 4,5,6,7
    ax.plot([xx[5],xx[9]], [yy[5], yy[9]], [zz[5], zz[9]])
    ax.plot([xx[6],xx[9]], [yy[6], yy[9]], [zz[6], zz[9]])
    ax.plot([xx[7],xx[9]], [yy[7], yy[9]], [zz[7], zz[9]])
    ax.plot([xx[8],xx[9]], [yy[8], yy[9]], [zz[8], zz[9]])
    return

def _triangulate_midpoint(pl,pr,Rlr,tlr):
    plt=pl
    prt=pr
    # print("plt", plt)
    # print("prt", prt)
    q = np.cross(np.squeeze(np.asarray(plt)),np.squeeze(np.asarray(Rlr*prt)));
    q = np.divide(q, np.linalg.norm(q)) # normalize it
    # Find the scalars a,b,c from this equation
    # a (plt  + c (q) = b ( Rlr prt ) + Tlr
    # Solve 3 equations, 3 unknowns, exact solution
    A = np.matrix([ np.squeeze(np.asarray(plt)).tolist(),
                    np.squeeze(np.asarray((-Rlr*prt))).tolist(),
                    np.squeeze(np.asarray(q)).tolist() ]).T
    _A = np.linalg.inv(A)*tlr;
    a = _A[0,0]
    b = _A[1,0]
    c = _A[2,0]
    # 3D point is a*plt + c*0.5*q
    outpoint = (np.squeeze(np.asarray(a*plt)).tolist() + c*0.5*q).tolist()
    return outpoint

def triangulate_midpoint(pl,pr,Rlr,tlr,Twl):
    outpoints = []
    for _pl, _pr in zip(pl,pr):
        wrt_left = _triangulate_midpoint(_pl,_pr,Rlr,tlr)
        wrt_left.append(1)
        three_d_point = Twl*np.matrix(wrt_left).T
        outpoints.append(np.squeeze(np.asarray(three_d_point.T)).tolist())
    return outpoints

def euclidean_stereo_reconstruction():
    K = np.matrix([ [-100, 0, 200],
                    [0, -100, 200],
                    [0, 0, 1]])
    Mextleft = np.matrix([[0.707, 0.707, 0, -3+1.414],
                        [-0.707, 0.707, 0, -0.5],
                        [0, 0,  1, 3]])
    Mextright = np.matrix([[0.866, -0.5, 0, -3+(0.866-0.5)],
                        [0.5, 0.866, 0, -1.5+(0.5+0.866)],
                        [0, 0, 1, 3]])
    Ps_w = [  [1, -1, 0],
              [2, -1, 0],
              [2, 0, 0],
              [1, 0, 0],
              [1, -1, 1],
              [2, -1, 1],
              [2, 0, 1],
              [1, 0, 1],
              [1.5,-0.5,1.5]]    #
    leftpix = []
    rightpix = []
    for p in Ps_w:
        p.append(1)
        pixels = K*Mextleft*np.matrix(p).T
        divisor = 1
        if pixels[2] != 0:
            divisor = pixels[2]
        leftpix.append(pixels/divisor)
        pixels = K*Mextright*np.matrix(p).T
        divisor = 1
        if pixels[2] != 0:
            divisor = pixels[2]
        rightpix.append(pixels/divisor)
    # convert from pixels to rays
    rightray = []
    leftray = []
    for pix in rightpix:
        rightray.append(np.linalg.inv(K)*np.matrix([pix[0,0], pix[1,0], pix[2,0]]).T)
    for pix in leftpix:
        leftray.append(np.linalg.inv(K)*np.matrix([pix[0,0], pix[1,0], pix[2,0]]).T)

    # 3D reconstruction with known camera Matrix
    Trw = np.matrix([np.squeeze(np.asarray(Mextright[0])),
                    np.squeeze(np.asarray(Mextright[1])),
                    np.squeeze(np.asarray(Mextright[2])),
                    [0,0,0,1]])
    Tlw = np.matrix([np.squeeze(np.asarray(Mextleft[0])),
                    np.squeeze(np.asarray(Mextleft[1])),
                    np.squeeze(np.asarray(Mextleft[2])),
                    [0,0,0,1]])
    Twr = np.linalg.inv(Trw); # can be done using transpose
    Twl = np.linalg.inv(Tlw); # can be done using transpose

    Tlr = Tlw*Twr;
    # Rotation from right to left coordinate frame
    Rlr = Tlr[0:3,0:3]
    # translation
    tlr = Tlr[0:3,3:4]
    # Triangulate
    out = []
    for _lr, _rr in zip(leftray, rightray):
        # print(_lr,"\n",_rr,"\n",Rlr,"\n",tlr,"\n",Twl)
        wrt_left = _triangulate_midpoint(_lr,_rr,Rlr,tlr)
        # print(wrt_left)
        wrt_left.append(1)
        three_d_point = Twl*np.matrix(wrt_left).T
        out.append(np.squeeze(np.asarray(three_d_point.T)).tolist())
        # print(np.round(np.squeeze(np.asarray(three_d_point.T)).tolist()[0:3],5))
    _K = np.matrix([[K[0,0], K[0,1], K[0,2], 0],
                    [K[1,0], K[1,1], K[1,2], 0],
                    [K[2,0], K[2,1], K[2,2], 0],
                    [0,0,0,1]])
    # print(_K)
    # Reprojection error
    M_RW = _K*Trw
    # print(M_RW)
    M_LW = _K * Tlw
    # print(M_LW)
    total_error = 0
    for actual_l, actual_r, projected in zip(leftpix, rightpix, out):
        projected = np.matrix(projected).T
        # print(M_LW)
        left_projected = M_LW * projected
        if left_projected[2] != 0:
            left_projected = left_projected[0:2] / left_projected[2]
        else:
            left_projected = left_projected[0:2] / 1

        # print(actual_l[0:2].T)
        actual_l = actual_l[0:2]
        actual_r = actual_r[0:2]
        total_error += np.square(left_projected-actual_l).sum()
        right_projected = M_RW * projected
        if right_projected[2] != 0:
            right_projected = right_projected[0:2] / right_projected[2]
        else:
            right_projected = right_projected[0:2] / 1
        total_error += np.square(right_projected-actual_r).sum()
        # print(total_error)
        # print(actual_l, "\n", left_projected, "\n", actual_r, "\n", right_projected, "\n")

    mean_error = total_error / (2*len(out))
    print("Problem 1: The mean squared reprojection error is", mean_error)

    fig2 = plt.figure("Problem 8c: 3D World")
    ax2 = fig2.gca(projection='3d')
    draw3D(np.matrix(out), ax2)
    plt.show()
    return

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

            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
            rot = cv2.Rodrigues(np.array(rvecs))[0]
            tvecs = np.array(tvecs)
            rot = np.append(rot, tvecs[0], axis=1)
            M = np.dot(mtx,rot)
            print("\nCamera matrix for image", filename,"\n",M)
            # Apply QR factorization to find M's components
            _M = np.delete(M, 3, axis=1)
            Q, R = np.linalg.qr(np.linalg.inv(_M))
            Rwc = np.linalg.inv(Q)  # Rotation
            K = np.linalg.inv(R)    # intrinsic
            D = np.matrix([[np.sign(K[0,0]),0,0],
                            [0,np.sign(K[1,1]),0],
                            [0,0,np.sign(K[2,2])]])

            # print("D",D)
            K = K*D
            Rwc = D*Rwc
            K = K/K[2,2]
            # Apply SVD to M to find tcw
            u, s, v = np.linalg.svd(M)
            tcw = v[len(v)-1][0:3] / v[len(v)-1][3]
            tcw = -Rwc*np.matrix(tcw).T
            Tcw = np.c_[Rwc, tcw]
            print("\nTcw:\n", Tcw)
            print("\nK:\n", K)
            # _K=np.matrix([[K[0,0],K[0,1],K[0,2],0],
            #               [K[1,0],K[1,1],K[1,2],0],
            #               [K[2,0],K[2,1],K[2,2],0],
            #               [0,0,0,1]])
            # _Tcw=np.matrix([[Tcw[0,0],Tcw[0,1],Tcw[0,2],Tcw[0,3]],
            #               [Tcw[1,0],Tcw[1,1],Tcw[1,2],Tcw[1,3]],
            #               [Tcw[2,0],Tcw[2,1],Tcw[2,2],Tcw[2,3]],
            #               [0,0,0,1]])
            #
            # print("M at end\n",_K*_Tcw)
            # print()
            cv2.imshow('img',img)
            cv2.waitKey(5000)

        cv2.destroyAllWindows()
    except:
        print("File not found/Bad input data")
        return

def main():
    K = np.matrix([ [-100, 0, 200],
                    [0, -100, 200],
                    [0, 0, 1]])
    Mextleft = np.matrix([[0.707, 0.707, 0, -3+1.414],
                        [-0.707, 0.707, 0, -0.5],
                        [0, 0,  1, 3]])
    Mextright = np.matrix([[0.866, -0.5, 0, -3+(0.866-0.5)],
                        [0.5, 0.866, 0, -1.5+(0.5+0.866)],
                        [0, 0, 1, 3]])

    Ps_w = [[1, -1, 0],
            [2, -1, 0],
            [2, 0, 0],
            [1, 0, 0],
            [1, -1, 1],
            [2, -1, 1],
            [2, 0, 1],
            [1, 0, 1],
            [1.5,-0.5,1.5]]    #
    leftpix  = []        # Get points in left camera view
    rightpix = []       # Get points in right camera view
    # print(K*Mextleft)
    for p in Ps_w:
        p.append(1)
        pixels = K*Mextleft*np.matrix(p).T
        divisor = 1
        if pixels[2] != 0:
            divisor = pixels[2]
        leftpix.append(pixels/divisor)
        pixels = K*Mextright*np.matrix(p).T
        divisor = 1
        if pixels[2] != 0:
            divisor = pixels[2]
        rightpix.append(pixels/divisor)
    # print(leftpix)

    # Do eight point algorithm
    # Create A
    tt = [left * right.T for left, right in zip(leftpix, rightpix)]
    # print(tt)
    A = []
    for _tt in tt:
        # _tt = _tt.T
        # print(np.reshape(_tt, (9,1)))
        _tt = np.reshape(_tt, (9,1))
        _tt = np.squeeze(np.asarray(_tt))
        # print(_tt)
        A.append(_tt)
    A = np.matrix(A)
    # print(A)
    # for p in Pw
    # Compute SVD(A)
    u, s, v = np.linalg.svd(A)
    # print(v[len(v)-1])
    F = np.reshape(v[len(v)-1],(3,3))
    print("The Fundamental Matrix:\n", F)

    # Estimate the Essential Matrix
    E = np.linalg.inv(K) * F * K
    print("The Essential Matrix:\n", E)

    W = [[0,-1,0],
         [1,0,0],
         [0,0,1]]
    Z = [[0,1,0],
         [-1,0,0],
         [0,0,0]]

    u, s, v = np.linalg.svd(E)
    S1 = -u * Z * np.linalg.inv(u)
    S2 = u * Z * np.linalg.inv(u)
    R1 = u * np.linalg.inv(W) * np.linalg.inv(v)
    R2 = u * W * np.linalg.inv(v)

    # convert from pixels to rays
    rightray = []
    leftray = []
    for pix in rightpix:
        rightray.append(np.linalg.inv(K)*np.matrix([pix[0,0], pix[1,0], pix[2,0]]).T)
    for pix in leftpix:
        leftray.append(np.linalg.inv(K)*np.matrix([pix[0,0], pix[1,0], pix[2,0]]).T)

    foundit = False # foundit flags that the +z reconstruction is found
    points3D=[]
    # case 1
    if not foundit:
        S = S1;R=R1;
        tlr = np.matrix([ S[2,1], S[0,2], -S[0,1]]).T
        reconpts = triangulate_midpoint(leftray,rightray,R,tlr,np.identity(4))
        if np.min(np.matrix(reconpts).T[2]) > 0:
            foundit = True

    # case 2
    if not foundit:
        S = S2;R=R1;
        tlr = np.matrix([ S[2,1], S[0,2], -S[0,1]]).T;
        reconpts = triangulate_midpoint(leftray,rightray,R,tlr,np.identity(4))
        if np.min(np.matrix(reconpts).T[2]) > 0:
            foundit = True

    # case 3
    if not foundit:
        S = S1;R=R2;
        tlr = np.matrix([ S[2,1], S[0,2], -S[0,1]]).T;
        reconpts = triangulate_midpoint(leftray,rightray,R,tlr,np.identity(4))
        if np.min(np.matrix(reconpts).T[2]) > 0:
            foundit = True

    # case 4
    if not foundit:
        S = S2;R=R2;
        tlr = np.matrix([ S[2,1], S[0,2], -S[0,1]]).T;
        reconpts = triangulate_midpoint(leftray,rightray,R,tlr,np.identity(4))
        if np.min(np.matrix(reconpts).T[2]) > 0:
            foundit = True

    # Draw 2D left camera
    leftLines = draw2D(leftpix)
    fig = plt.figure("Left Camera")
    ax = fig.add_subplot(111)
    for line in leftLines:
        ax.add_line(line)
    minx = min([pt[0,0] for pt in leftpix])
    maxx = max([pt[0,0] for pt in leftpix])
    miny = min([pt[1,0] for pt in leftpix])
    maxy = max([pt[1,0] for pt in leftpix])
    axis=[minx-2,maxx+2,miny-2,maxy+2]
    ax.axis(axis)
    # Draw 2D right camera
    rightLines = draw2D(rightpix)
    fig1 = plt.figure("Right Camera")
    ax1 = fig1.add_subplot(111)
    for line in rightLines:
        ax1.add_line(line)
    minx = min([pt[0,0] for pt in rightpix])
    maxx = max([pt[0,0] for pt in rightpix])
    miny = min([pt[1,0] for pt in rightpix])
    maxy = max([pt[1,0] for pt in rightpix])
    axis1=[minx-2,maxx+2,miny-2,maxy+2]
    ax1.axis(axis1)
    # Draw 3D plot
    fig2 = plt.figure("3D World")
    ax2 = fig2.gca(projection='3d')
    draw3D(np.matrix(reconpts), ax2)
    plt.show()

if __name__ == "__main__":
    print("\nProblems 1 and 8a next\n")
    euclidean_stereo_reconstruction()
    print("\nProblems 5 to 8b next\n")
    main()
    print("\nProblems 2 and 3 next\n")
    calibrate()

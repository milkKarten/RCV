import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pprint import pprint

def draw2D(pts):
    xx = [pt[0,0] for pt in pts]
    xx.insert(0,0)
    yy = [pt[1,0] for pt in pts]
    yy.insert(0,0)
    # print(xx)
    lines = []
    # lines.append(Line2D([xx[1],xx[2]], [yy[1],yy[2]]))
    # lines.append(Line2D([xx[1],xx[3]], [yy[1],yy[3]]))
    # lines.append(Line2D([xx[1],xx[4]], [yy[1],yy[4]]))
    # lines.append(Line2D([xx[2],xx[5]], [yy[2],yy[5]]))
    # lines.append(Line2D([xx[2],xx[6]], [yy[2],yy[6]]))
    # lines.append(Line2D([xx[3],xx[5]], [yy[3],yy[5]]))
    # lines.append(Line2D([xx[3],xx[7]], [yy[3],yy[7]]))
    # lines.append(Line2D([xx[4],xx[6]], [yy[4],yy[6]]))
    # lines.append(Line2D([xx[4],xx[7]], [yy[4],yy[7]]))
    # lines.append(Line2D([xx[4],xx[9]], [yy[4],yy[9]]))
    # lines.append(Line2D([xx[5],xx[8]], [yy[5],yy[8]]))
    # lines.append(Line2D([xx[6],xx[8]], [yy[6],yy[8]]))
    # lines.append(Line2D([xx[6],xx[9]], [yy[6],yy[9]]))
    # lines.append(Line2D([xx[7],xx[8]], [yy[7],yy[8]]))
    # lines.append(Line2D([xx[7],xx[9]], [yy[7],yy[9]]))
    # lines.append(Line2D([xx[8],xx[9]], [yy[8],yy[9]]))
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
    return lines

def draw3D(pts, ax):
    xx = np.squeeze(np.asarray(pts.T[0]))
    xx=np.insert(xx,0,0)
    yy = np.squeeze(np.asarray(pts.T[1]))
    yy=np.insert(yy,0,0)
    zz = np.squeeze(np.asarray(pts.T[2]))
    zz=np.insert(zz,0,0)
    # [xx[1],xx[2]], [yy[1],yy[2]]
    # [xx[1],xx[3]], [yy[1],yy[3]]
    # [xx[1],xx[4]], [yy[1],yy[4]]
    # [xx[2],xx[5]], [yy[2],yy[5]]
    # [xx[2],xx[6]], [yy[2],yy[6]]
    # [xx[3],xx[5]], [yy[3],yy[5]]
    # [xx[3],xx[7]], [yy[3],yy[7]]
    # [xx[4],xx[6]], [yy[4],yy[6]]
    # [xx[4],xx[7]], [yy[4],yy[7]]
    # [xx[4],xx[9]], [yy[4],yy[9]]
    # [xx[5],xx[8]], [yy[5],yy[8]]
    # [xx[6],xx[8]], [yy[6],yy[8]]
    # [xx[6],xx[9]], [yy[6],yy[9]]
    # [xx[7],xx[8]], [yy[7],yy[8]]
    # [xx[7],xx[9]], [yy[7],yy[9]]
    # [xx[8],xx[9]], [yy[8],yy[9]]
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
    return

def _triangulate_midpoint(pl,pr,Rlr,tlr):
    plt=pl
    prt=pr
    # print("plt", plt)
    # print("prt", prt)
    # print(Rlr)
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

def main():
    K = np.matrix([ [-100, 0, 200],
                    [0, -100, 200],
                    [0, 0, 1]])
    # Mextleft = np.matrix([  [0,0,-1, 10],
    #                         [1,0,0, 0],
    #                         [0,-1,0, 0]])
    # Mextright = np.matrix([ [-1.0/np.sqrt(2),0,0, np.sqrt(50)],
    #                         [1.0/np.sqrt(2),0,-1.0/np.sqrt(2), np.sqrt(50)],
    #                         [0,-1,0, 0] ])
    # # Load a simple polygonal house
    # Ps_w = [[0-0.5,0-0.5,0-0.5],#1: 2,3,4
    #      [1-0.5,0-0.5,0-0.5],#2: 5,6
    #      [0-0.5,1-0.5,0-0.5],#3: 5,7
    #      [0-0.5,0-0.5,1-0.5],#4: 6,7,9
    #      [1-0.5,1-0.5,0-0.5],#5: 8
    #      [1-0.5,0-0.5,1-0.5],#6: 8,9
    #      [0-0.5,1-0.5,1-0.5],#7: 8,9
    #      [1-0.5,1-0.5,1-0.5],#8: 9
    #      [0.5-0.5,0.5-0.5,1.5-0.5]]  #9:
    Mextleft = np.matrix([[0.707, 0.707, 0, -3],
                        [-0.707, 0.707, 0, -0.5],
                        [0, 0,  1, 3]])
    Mextright = np.matrix([[0.866, -0.5, 0, -3],
                        [0.5, 0.866, 0, -0.5],
                        [0, 0, 1, 3]])
    Ps_w = [  [2, 0, 0],
              [3, 0, 0],
              [3, 1, 0],
              [2, 1, 0],
              [2, 0, 1],
              [3, 0, 1],
              [3, 1, 1],
              [2, 1, 1],
              [2.5, 0.5, 2]]
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
    print(leftpix)

    # Do eight point algorithm
    # Create A
    tt = [left * right.T for left, right in zip(leftpix, rightpix)]
    print(tt)
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
    print()
    pprint(reconpts)
    print("\n\n\n")

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


    '''Euclidean Reconstruction'''

if __name__ == "__main__":
    main()

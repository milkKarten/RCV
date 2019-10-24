import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

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

def rotate(_Ps, R, t, Ps_new, M_in=None):
    Ps = [p[:] for p in _Ps.copy()]
    K = np.matrix([ [-100, 0, 200],
                    [0, -100, 200],
                    [0, 0, 1]])
    M = K * R.T
    M_t = K * -R.T*t
    M = np.c_[M, M_t]
    if Min is not None:
        M=M_in
    # print(M)
    _ps = []
    for P in Ps:
        P.append(1)
        P = np.matrix(P)
        _ps.append(M*P.T)
    for p in _ps:
        z = p[2]
        if z == 0:
            z = 1
        x_new = (p[0]/z)[0,0]
        y_new = (p[1]/z)[0,0]
        # Ps_new.append([(p[0])[0,0], (p[1])[0,0], z[0,0]])
        Ps_new.append([x_new, y_new, 100])
    return M

def triangulate_midpoint(pl,pr,Rlr,tlr,Twl):
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
    Mextleft = np.matrix([[0.707, 0.707, 0, -3],
                        [-0.707, 0.707, 0, -0.5],
                        [0, 0,  1, 3]])
    Mextright = np.matrix([[0.866, -0.5, 0, -3],
                        [0.5, 0.866, 0, -0.5],
                        [0, 0, 1, 3]])
    # p1_w = [2,0,0]
    # p2_w = [3,0,0]
    # p3_w = [3,1,0]
    # Ps_w = [[0-0.5,0-0.5,0-0.5],#1: 2,3,4
    #      [1-0.5,0-0.5,0-0.5],#2: 5,6
    #      [0-0.5,1-0.5,0-0.5],#3: 5,7
    #      [0-0.5,0-0.5,1-0.5],#4: 6,7,9
    #      [1-0.5,1-0.5,0-0.5],#5: 8
    #      [1-0.5,0-0.5,1-0.5],#6: 8,9
    #      [0-0.5,1-0.5,1-0.5],#7: 8,9
    #      [1-0.5,1-0.5,1-0.5],#8: 9
    #      [0.5-0.5,0.5-0.5,1.5-0.5]]  #9:
    Ps_w = [  [2, 0, 0],
              [3, 0, 0],
              [3, 1, 0],
              [2, 1, 0],
              [2, 0, 1],
              [3, 0, 1],
              [3, 1, 1],
              [2, 1, 1],
              [2.5, 0.5, 2]]
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
        wrt_left = triangulate_midpoint(_lr,_rr,Rlr,tlr,Twl)
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
    print("The mean squared reprojection error is", mean_error)

    fig2 = plt.figure("3D World")
    ax2 = fig2.gca(projection='3d')
    draw3D(np.matrix(out), ax2)
    plt.show()
    return

if __name__ == "__main__":
    main()

import numpy as np

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

def main2():
    K = np.matrix([ [-100, 0, 200],
                    [0, -100, 200],
                    [0, 0, 1]])
    Mextleft = np.matrix([[0.707, 0.707, 0, -3],
                        [-0.707, 0.707, 0, -0.5],
                        [0, 0,  1, 3]])
    Mextright = np.matrix([[0.866, -0.5, 0, -3],
                        [0.5, 0.866, 0, -0.5],
                        [0, 0, 1, 3]])
    p1_w = [2,0,0]
    p2_w = [3,0,0]
    p3_w = [3,1,0]
    Ps_w = [p1_w, p2_w, p3_w]
    leftpix = []
    rightpix = []
    for p in Ps_w:
        p.append(1)
        pixels = K*Mextleft*np.matrix(p).T
        leftpix.append(pixels/pixels[2])
        pixels = K*Mextright*np.matrix(p).T
        rightpix.append(pixels/pixels[2])
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
        print(np.round(np.squeeze(np.asarray(three_d_point.T)).tolist()[0:3],5))
    return

def main():
    p1_w = [-0.5, -0.5, -0.5]
    p2_w = [0.5, -0.5, -0.5]
    p3_w = [-0.5, 0.5, -0.5]
    Ps_w = [p1_w, p2_w, p3_w]
    # world frame R, t
    # A: RWA, tWA
    R = np.matrix([[0,0,-1],[1,0,0],[0,-1,0]])
    t = np.matrix([10.,0,0]).T
    Ps_A = []
    M_A = rotate(Ps_w, R, t, Ps_A)

    # B: RWB, tWB
    R2 = np.matrix([[-1.0/np.sqrt(2),0,0],[1.0/np.sqrt(2),0,-1.0/np.sqrt(2)],[0,-1,0]])
    t2 = np.matrix([np.sqrt(50),np.sqrt(50),0]).T
    Ps_B = []
    M_B = rotate(Ps_w, R2, t2, Ps_B)

    K = np.matrix([ [-100, 0, 200],
                    [0, -100, 200],
                    [0, 0, 1]])
    Rlr = R.T * R2
    tlr = R.T * t2
    print(tlr)
    sx=1;sy=1
    ox=200;oy=200
    f=100
    for pA, pB in zip(Ps_A, Ps_B):
        print(pA, pB)
        pA = np.linalg.inv(K)*np.matrix(pA).T
        pB = np.linalg.inv(K)*np.matrix(pB).T
        pA = np.squeeze(np.asarray(pA.T))
        pB = np.squeeze(np.asarray(pB.T))
        pl = [(pA[0]-ox)*-1*sx, (pA[1]-oy)*-1*sy, f]
        pr = [(pB[0]-ox)*-1*sx, (pB[1]-oy)*-1*sy, f]

        # init pl, plr, Rlr, tlr
        plt = np.matrix(pl).T
        prt = np.matrix(pr).T
        # print(np.squeeze(np.asarray(Rlr*prt)), np.squeeze(np.asarray(plt)))
        q = np.cross(np.squeeze(np.asarray(plt)),np.squeeze(np.asarray(Rlr*prt)));
        q = np.divide(q, np.linalg.norm(q)) # normalize it
        # Find the scalars a,b,c from this equation
        # a (plt  + c (q) = b ( Rlr prt ) + Tlr
        # Solve 3 equations, 3 unknowns, exact solution
        A = [np.squeeze(np.asarray(plt)).tolist(),  np.squeeze(np.asarray((-Rlr*prt))).tolist(), np.squeeze(np.asarray(q)).tolist()]
        _A = np.linalg.inv(A)*tlr;
        a = _A[0,0]
        b = _A[1,0]
        c = _A[2,0]
        # 3D point is a*plt + c*0.5*q

        # print(np.squeeze(np.asarray(a*plt)).tolist())
        # print(c*0.5*q)
        outpoint = np.squeeze(np.asarray(a*plt)).tolist() + c*0.5*q
        print(outpoint)
        pL = np.matrix(np.append(outpoint, 1))
        # print(pL)
        # print(M_A)
        M_A = np.matrix([np.squeeze(np.asarray(M_A[0])),
                        np.squeeze(np.asarray(M_A[1])),
                        np.squeeze(np.asarray(M_A[2])),
                        [0,0,0,1]])
        # print(M_A)
        print(M_A * pL.T)
        exit(0)

    return

if __name__ == "__main__":
    main2()

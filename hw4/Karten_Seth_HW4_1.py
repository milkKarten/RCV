import numpy as np

def rotate(_Ps, R, t, Ps_new):
    Ps = [p[:] for p in _Ps.copy()]
    K = np.matrix([ [-100, 0, 200],
                    [0, -100, 200],
                    [0, 0, 1]])
    M = K * R.T
    M_t = K * -R.T*t
    M = np.c_[M, M_t]
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
    main()

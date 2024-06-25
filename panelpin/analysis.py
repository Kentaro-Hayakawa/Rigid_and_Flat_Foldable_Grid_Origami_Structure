##################################################
# import libraries
##################################################
import numpy as np
import math
import os
import shutil



##################################################
# hyperparameters  units: kN, m ,s
##################################################
height = 0.03              # half length of the longest edge of a triangular panel
thick = 0.003              # thickness of panels
ni = 3                     # number of grids in i-direction
nj = 3                     # number of grids in j-direction
wfa = 0.02*thick           # face weight per unit area
kkc0 = 5.e-5               # rotation stiffness of hinge-rotational springs per unit length of crease lines
kkf = 2.e+7*thick          # stiffness of vertex-connecting springs
dtol = 1.e-8               # threshold of zero singular value for mechanism analysis
nstep = 73                 # number of analysis steps
nprint = 12                # number of displayed eigenvalues
theta_min = 0.001*math.pi  # min. value of theta
theta_max = 0.999*math.pi  # max. value of theta



##################################################
# functions for model construction
##################################################
### construction of unit A
def unitA(theta,  # angle between xy-plane and face 014 (float)
          height, # height of triangular panel (float)
          i,      # grid index in i-direction (int)
          j       # grid index in j-direction (int)
          ):
    # parameters
    if theta < 1.e-12:
        theta = 0.
        phi = math.pi
    elif theta > math.pi - 1.e-12:
        theta = math.pi
        phi = 0.
    else:
        phi = 2.*math.atan((math.sqrt(2)-1)/math.tan(theta/2))
    hh = height/2
    # reference vectors
    xa = np.array([hh*(1.+math.cos(theta)), 0., hh*math.sin(theta)])
    xb = np.array([0., hh*(1.+math.cos(phi)), hh*math.sin(phi)])
    vai = np.array([hh, -hh, 0.])
    vbi = np.array([-xa[0]+xb[0], xa[1]+xb[1], -xa[2]-xb[2]])/2.
    vi = vai-vbi
    vaj = np.array([hh, hh, 0.])
    vbj = np.array([-xa[0]+xb[0], xa[1]-xb[1], -xa[2]-xb[2]])/2.
    vj = vaj-vbj
    # vertices
    vert = np.array([[hh, -hh, 0.],
                     [hh, hh, 0.],
                     [-hh, hh, 0.],
                     [-hh, -hh, 0.],
                     [xa[0], xa[1], xa[2]],
                     [xb[0], xb[1], xb[2]],
                     [-xa[0], xa[1], xa[2]],
                     [xb[0], -xb[1], xb[2]]])
    vert += np.array([[i*vi[0]+j*vj[0], i*vi[1]+j*vj[1], 0.]])
    # faces
    face = np.array([[0, 1, 4],
                     [1, 5, 4],
                     [1, 2, 5],
                     [2, 6, 5],
                     [2, 3, 6],
                     [3, 7, 6],
                     [3, 0, 7],
                     [0, 4, 7]], dtype=int)
    # crease lines and adjacent faces
    crease = np.array([[0, 4, 7, 0],
                       [0, 7, 6, 7],
                       [1, 4, 0, 1],
                       [1, 5, 1, 2],
                       [2, 5, 2, 3],
                       [2, 6, 3, 4],
                       [3, 6, 4, 5],
                       [3, 7, 5, 6]], dtype=int)
    return vert.T, face.T, crease.T

### construction of unit B
def unitB(theta,  # angle between xy-plane and face 014 (float)
          height, # height of triangular panel (float)
          i,      # grid index in i-direction (int)
          j       # grid index in j-direction (int)
          ):
    # parameters
    if theta < 1.e-12:
        theta = 0.
        phi = math.pi
    elif theta > math.pi - 1.e-12:
        theta = math.pi
        phi = 0.
    else:
        phi = 2.*math.atan((math.sqrt(2)-1)/math.tan(theta/2))
    hh = height/2
    # reference vectors
    xa = np.array([hh*(1.+math.cos(theta)), 0., -hh*math.sin(theta)])
    xb = np.array([0., hh*(1.+math.cos(phi)), -hh*math.sin(phi)])
    vai = np.array([hh, -hh, 0.])
    vbi = np.array([-xa[0]+xb[0], xa[1]+xb[1], xa[2]+xb[2]])/2.
    vi = vai-vbi
    vaj = np.array([hh, hh, 0.])
    vbj = np.array([-xa[0]+xb[0], xa[1]-xb[1], xa[2]+xb[2]])/2.
    vj = vaj-vbj
    # vertices
    vert = np.array([[hh, -hh, 0.],
                     [hh, hh, 0.],
                     [-hh, hh, 0.],
                     [-hh, -hh, 0.],
                     [xa[0], xa[1], xa[2]],
                     [xb[0], xb[1], xb[2]],
                     [-xa[0], xa[1], xa[2]],
                     [xb[0], -xb[1], xb[2]]])
    vert += np.array([[i*vi[0]+j*vj[0], i*vi[1]+j*vj[1], vi[2]]])
    # faces
    face = np.array([[0, 1, 4],
                     [1, 5, 4],
                     [1, 2, 5],
                     [2, 6, 5],
                     [2, 3, 6],
                     [3, 7, 6],
                     [3, 0, 7],
                     [0, 4, 7]], dtype=int)
    # crease lines and adjacent faces
    crease = np.array([[0, 4, 7, 0],
                       [0, 7, 6, 7],
                       [1, 4, 0, 1],
                       [1, 5, 1, 2],
                       [2, 5, 2, 3],
                       [2, 6, 3, 4],
                       [3, 6, 4, 5],
                       [3, 7, 5, 6]], dtype=int)

    return vert.T, face.T, crease.T

### construction of entire model
def modelling(theta,  # angle between xy-plane and face 014 (float)
              height, # height of triangular panel (float)
              ni,     # number of grids in i-direction (int)
              nj      # number of grids in j-direction (int)
              ):
    vert = np.zeros((3,0), float)
    face = np.zeros((3,0), int)
    crease = np.zeros((4,0), int)
    glued = np.zeros((2,0), int)
    for j in range(nj):
        for i in range(ni):
            kij = i+ni*j
            if (i+j)%2 == 0:
                vert_new, face_new, crease_new = unitA(theta, height, i, j)
            else:
                vert_new, face_new, crease_new = unitB(theta, height, i, j)
            vert = np.append(vert, vert_new, axis=1)
            face = np.append(face, face_new+8*(i+ni*j), axis=1)
            crease = np.append(crease, crease_new++8*(i+ni*j), axis=1)
            if j == 0:
                if i > 0:
                    glued = np.append(glued, np.array([[8*((i-1)+ni*j)+7], [8*(i+ni*j)+3]]), axis=1)
            else:
                if i == 0:
                    glued = np.append(glued, np.array([[8*(i+ni*(j-1))+1], [8*(i+ni*j)+5]]), axis=1)
                else:
                    glued = np.append(glued, np.array([[8*((i-1)+ni*j)+7], [8*(i+ni*j)+3]]), axis=1)
                    glued = np.append(glued, np.array([[8*(i+ni*(j-1))+1], [8*(i+ni*j)+5]]), axis=1)
    return vert, face, crease, glued



##################################################
# functions for mechanism analysis
##################################################
def rref(arr, subarr=np.zeros(0), tol=1e-12):
    arr_new = np.copy(arr)
    arr = arr_new
    rows, cols = arr.shape
    pivot_pos = np.zeros((0,2),dtype=int)
    rows_pos = np.arange(rows)
    if subarr.shape[0] == 0:
        subarr = np.identity(rows)
    else:
        subarr_new = np.copy(subarr)
        subarr = subarr_new
    r = 0
    for c in range(cols):
        # find the pivot point
        pivot = np.argmax(np.abs(arr[r:rows,c]))+r
        maxc = np.abs(arr[pivot,c])
        # skip column c if maxc <= tol
        if maxc <= tol:
            ## make column c accurately zero
            arr[r:rows,c] = 0.
        else:
            pivot_pos = np.append(pivot_pos, [[r,c]], axis=0)
            # swap current row and pivot row
            if pivot != r:
                arr[[r,pivot],:] = arr[[pivot,r],:]
                rows_pos[[r,pivot]] = rows_pos[[pivot,r]]
                subarr[[r,pivot],:] = subarr[[pivot,r],:]
            # normalize pivot row
            div = np.copy(arr[r,c])
            arr[r,c:cols] = arr[r,c:cols]/div
            subarr[r,:] = subarr[r,:]/div
            v = arr[r,c:cols]
            subv = subarr[r,:]
            # eliminate the current column
            # above r
            if r > 0:
                dif = np.copy(arr[0:r,c])
                arr[0:r,c:cols] = arr[0:r,c:cols] - np.outer(dif,v)
                subarr[0:r,:] = subarr[0:r,:] - np.outer(dif,subv)
            # below r
            if r < rows:
                dif = np.copy(arr[r+1:rows,c])
                arr[r+1:rows,c:cols] = arr[r+1:rows,c:cols] - np.outer(dif,v)
                subarr[r+1:rows,:] = subarr[r+1:rows,:] - np.outer(dif,subv)
            r += 1
            # check if done
        if r == rows or c == cols:
            # eliminate nearly zero element
            for i in range(rows):
                for j in range(cols):
                    if abs(arr[i,j]) <= tol:
                        arr[i,j] = 0.
            break
    pivot_pos = pivot_pos[np.argsort(pivot_pos, axis=0)[:,0]]
    return arr, pivot_pos, rows_pos, subarr

### cross product matrix
def crossmatrix(vec  # vector(s) ([3,*] array, float)
                ):
    # construct outer product matrix
    mat = np.zeros((3,3,vec.shape[1]),float)
    mat[0,1,:] = -vec[2,:]
    mat[0,2,:] =  vec[1,:]
    mat[1,0,:] =  vec[2,:]
    mat[1,2,:] = -vec[0,:]
    mat[2,0,:] = -vec[1,:]
    mat[2,1,:] =  vec[0,:]
    return mat

### 3D rotation matrix around rotation vector
def rot3D(vec  # rotation vector(s) ([3,*] array, float)
          ):
    # construct rotation matrix
    angle = np.linalg.norm(vec, axis=0)
    rot = np.zeros((3,3,vec.shape[1]),float)
    # angle > 0
    jj = np.where(angle > 0.)[0]
    if len(jj) > 0:
        nn = vec[:,jj]/angle[jj]
        nx = crossmatrix(nn)
        nnn = np.stack([nn,nn,nn],axis=1)*nn
        cos = np.cos(angle[jj])
        sin = np.sin(angle[jj])
        ee = np.zeros((3,3,len(jj)))
        ee[:,:,:] = np.identity(3)[:,:,None]
        rot[:,:,jj] = cos*ee+(1.-cos)*nnn+sin*nx
    # angle = 0
    rot[:,:,np.where(angle <= 0.)[0]] = np.identity(3)[:,:,None]
    return rot

### first-order derivative of rotation matrix with respect to vec[ii]
def drot3D(vec,  # rotation vector ([3,*] array, float)
           ii    # index of DOF to compute derivatives (int)
           ):
    # construct derivative of rotation matrix
    angle = np.linalg.norm(vec, axis=0)
    drot = np.zeros((3,3,vec.shape[1]),float)
    # angle > 0
    jj = np.where(angle > 0.)[0]
    if len(jj) > 0:
        nn = vec[:,jj]/angle[jj]
        nx = crossmatrix(nn)
        nnn = np.stack([nn,nn,nn],axis=1)*nn
        cos = np.cos(angle[jj])
        sin = np.sin(angle[jj])
        ee = np.zeros((3,3,len(jj)),float)
        ee[:,:,:] = np.identity(3)[:,:,None]
        ev = np.zeros((3,len(jj)),float)
        ev[ii,:] = 1.
        ex = crossmatrix(ev)
        ne = np.stack([nn,nn,nn],axis=1)*ev
        en = np.stack([ev,ev,ev],axis=1)*nn
        drot[:,:,jj] += -nn[ii,:]*(sin*ee - (sin-2.*(1.-cos)/angle[jj])*nnn - (cos-sin/angle[jj])*nx)
        drot[:,:,jj] += (1.-cos)/angle[jj]*(ne+en) + sin/angle[jj]*ex
    # angle = 0
    jj = np.where(angle <= 0.)[0]
    if len(jj) > 0:
        ev = np.zeros((3,len(jj)),float)
        ev[ii,:] = 1.
        ex = crossmatrix(ev)
        drot[:,:,jj] += ex
    return drot

### second-order derivative of rotation matrix with respect to vec[i1] and vec[i2]
def ddrot3D(vec,  # rotation vector ([3,*] array, float)
            i1,   # index of DOF to compute derivatives (int)
            i2    # index of DOF to compute derivatives (int)
            ):
    # construct derivative of rotation matrix
    angle = np.linalg.norm(vec, axis=0)
    ddrot = np.zeros((3,3,vec.shape[1]),float)
    # angle > 0
    jj = np.where(angle > 0.)[0]
    if len(jj) > 0:
        nn = vec[:,jj]/angle[jj]
        nx = crossmatrix(nn)
        nnn = np.stack([nn,nn,nn],axis=1)*nn
        cos = np.cos(angle[jj])
        sin = np.sin(angle[jj])
        ee = np.zeros((3,3,len(jj)),float)
        ee[:,:,:] = np.identity(3)[:,:,None]
        ev1 = np.zeros((3,len(jj)),float)
        ev1[i1,:] = 1.
        ex1 = crossmatrix(ev1)
        ne1 = np.stack([nn,nn,nn],axis=1)*ev1
        en1 = np.stack([ev1,ev1,ev1],axis=1)*nn
        ev2 = np.zeros((3,len(jj)),float)
        ev2[i2,:] = 1.
        ex2 = crossmatrix(ev2)
        ne2 = np.stack([nn,nn,nn],axis=1)*ev2
        en2 = np.stack([ev2,ev2,ev2],axis=1)*nn
        ee12 = np.stack([ev1,ev1,ev1],axis=1)*ev2
        ee21 = np.stack([ev2,ev2,ev2],axis=1)*ev1
        nn12 = nn[i1,:]*nn[i2,:]
        if i1 == i2:
            del12 = np.ones(len(jj),float)
        else:
            del12 = np.zeros(len(jj),float)
        ddrot[:,:,jj] += ( -nn12*cos - (del12-nn12)*sin/angle[jj] )*ee
        ddrot[:,:,jj] += ( nn12*cos + (del12-5.*nn12)*sin/angle[jj] - 2.*(del12-4.*nn12)*(1.-cos)/angle[jj]**2 )*nnn
        ddrot[:,:,jj] += ( -nn12*sin + (del12-3.*nn12)*(cos/angle[jj]-sin/angle[jj]**2) )*nx
        ddrot[:,:,jj] += ( sin/angle[jj] - 2.*(1.-cos)/angle[jj]**2 )*( nn[i2,:]*(en1+ne1) + nn[i1,:]*(en2+ne2) )
        ddrot[:,:,jj] += ( (1.-cos)/angle[jj]**2 )*( ee12 + ee21 )
        ddrot[:,:,jj] += ( cos/angle[jj] - sin/angle[jj]**2 )*( nn[i1,:]*ex2 + nn[i2,:]*ex1 )
    # angle = 0
    jj = np.where(angle <= 0.)[0]
    if len(jj) > 0:
        ee = np.zeros((3,3,len(jj)),float)
        ee[:,:,:] = np.identity(3)[:,:,None]
        ev1 = np.zeros((3,len(jj)),float)
        ev1[i1,:] = 1.
        ev2 = np.zeros((3,len(jj)),float)
        ev2[i2,:] = 1.
        ee12 = np.stack([ev1,ev1,ev1],axis=1)*ev2
        ee21 = np.stack([ev2,ev2,ev2],axis=1)*ev1
        if i1 == i2:
            ddrot[:,:,jj] += -ee + (ee12 + ee21)/2.
        else:
            ddrot[:,:,jj] += (ee12 + ee21)/2.
    return ddrot


##################################################
### incompatibility vector for face displacement
def compatibility(disp,   # generalized displacement ([6nf] array, float)
                  nf,     # number of faces (int)
                  dd,     # vectors from barycenters to vertices of faces ([3,*] array, float)
                  jkcomp  # face and ref. vector indices adjacent to connecter vertices ([4,#connect] array, int)
                  ):
    dispf = disp.reshape([6,nf])
    # rotation matrices
    rot0 = rot3D(dispf[3:6,jkcomp[0,:]])
    rot1 = rot3D(dispf[3:6,jkcomp[1,:]])
    # displacement of vertices
    vv0 = dispf[0:3,jkcomp[0,:]] + np.sum(rot0*dd[:,jkcomp[2,:]], axis=1) - dd[:,jkcomp[2,:]]
    vv1 = dispf[0:3,jkcomp[1,:]] + np.sum(rot1*dd[:,jkcomp[3,:]], axis=1) - dd[:,jkcomp[3,:]]
    # incompatibility vector
    comp = np.ravel(vv0 - vv1)
    return comp

### first-order derivative of incompatibility vector for face displacement
def diffcompatibility(disp,   # generalized displacement ([6nf] array, float)
                      nf,     # number of faces (int)
                      dd,     # vectors from barycenters to vertices of faces ([3,*] array, float)
                      jkcomp  # face and ref. vector indices adjacent to connecter vertices ([4,#connect] array, int)
                      ):
    ncmp = jkcomp.shape[1]
    dispf = disp.reshape([6,nf])
    dcomp = np.zeros((3*ncmp,6*nf))
    ii = np.arange(ncmp)
    # first-order derivative of incompatibility vector
    for i in range(3):
        # w.r.t. translation
        dcomp[ii+ncmp*i ,jkcomp[0,:]+nf*i] = 1.
        dcomp[ii+ncmp*i ,jkcomp[1,:]+nf*i] = -1.
        # w.r.t. rotation
        drot0 = drot3D(dispf[3:6,jkcomp[0,:]],i)
        drot1 = drot3D(dispf[3:6,jkcomp[1,:]],i)
        ddd0 = np.sum(drot0*dd[:,jkcomp[2,:]], axis=1)
        ddd1 = np.sum(drot1*dd[:,jkcomp[3,:]], axis=1)
        dcomp[ii       ,jkcomp[0,:]+nf*i+3*nf] = ddd0[0,:]
        dcomp[ii+ncmp  ,jkcomp[0,:]+nf*i+3*nf] = ddd0[1,:]
        dcomp[ii+2*ncmp,jkcomp[0,:]+nf*i+3*nf] = ddd0[2,:]
        dcomp[ii       ,jkcomp[1,:]+nf*i+3*nf] = -ddd1[0,:]
        dcomp[ii+ncmp  ,jkcomp[1,:]+nf*i+3*nf] = -ddd1[1,:]
        dcomp[ii+2*ncmp,jkcomp[1,:]+nf*i+3*nf] = -ddd1[2,:]
    return dcomp

### second-order derivative of incompatibility vector for face displacement
def diff2compatibility(disp,   # generalized displacement ([6nf] array, float)
                       nf,     # number of faces (int)
                       dd,     # vectors from barycenters to vertices of faces ([3,*] array, float)
                       jkcomp  # face and ref. vector indices adjacent to connecter vertices ([4,#connect] array, int)
                       ):
    ncmp = jkcomp.shape[1]
    dispf = disp.reshape([6,nf])
    ddcomp = np.zeros((3*ncmp,6*nf,6*nf))
    ii = np.arange(ncmp)
    # second-order derivative of incompatibility vector
    for i in range(3):
        for j in range(3):
            # w.r.t. rotation (translation = 0)
            ddrot0 = ddrot3D(dispf[3:6,jkcomp[0,:]],i,j)
            ddrot1 = ddrot3D(dispf[3:6,jkcomp[1,:]],i,j)
            dddd0 = np.sum(ddrot0*dd[:,jkcomp[2,:]], axis=1)
            dddd1 = np.sum(ddrot1*dd[:,jkcomp[3,:]], axis=1)
            ddcomp[ii       ,jkcomp[0,:]+nf*i+3*nf,jkcomp[0,:]+nf*j+3*nf] = dddd0[0,:]
            ddcomp[ii+ncmp  ,jkcomp[0,:]+nf*i+3*nf,jkcomp[0,:]+nf*j+3*nf] = dddd0[1,:]
            ddcomp[ii+2*ncmp,jkcomp[0,:]+nf*i+3*nf,jkcomp[0,:]+nf*j+3*nf] = dddd0[2,:]
            ddcomp[ii       ,jkcomp[1,:]+nf*i+3*nf,jkcomp[1,:]+nf*j+3*nf] = -dddd1[0,:]
            ddcomp[ii+ncmp  ,jkcomp[1,:]+nf*i+3*nf,jkcomp[1,:]+nf*j+3*nf] = -dddd1[1,:]
            ddcomp[ii+2*ncmp,jkcomp[1,:]+nf*i+3*nf,jkcomp[1,:]+nf*j+3*nf] = -dddd1[2,:]
    return ddcomp

### folding angle
def foldangle(disp,   # generalized displacement ([6nf] array, float)
              nf,     # number of faces (int)
              nc,     # number of crease lines (int)
              nn,     # unit normal vectors of faces ([3,nf] array, float)
              bb,     # cross products of crease line vectors and face normals ([3,nc] array, float)
              jkfold  # face indices adjacent to crease lines ([2,nc] array, int)
              ):
    dispf = disp.reshape([6,nf])
    # rotation matrices
    rot0 = rot3D(dispf[3:6,jkfold[0,:]])
    rot1 = rot3D(dispf[3:6,jkfold[1,:]])
    # rotated vectors
    bb0 = np.sum(rot0*bb, axis=1)
    nn0 = np.sum(rot0*nn[:,jkfold[0,:]], axis=1)
    nn1 = np.sum(rot1*nn[:,jkfold[1,:]], axis=1)
    # cosine and sine of folding angles
    cos = np.sum(nn1*nn0, axis=0)
    sin = np.sum(nn1*bb0, axis=0)
    # folding angles
    rho = np.arctan2(sin,cos)
    return rho

### first-order derivative of folding angle
def difffoldangle(disp,   # generalized displacement ([6nf] array, float)
                  nf,     # number of faces (int)
                  nc,     # number of crease lines (int)
                  nn,     # unit normal vectors of faces ([3,nf] array, float)
                  bb,     # cross products of crease line vectors and face normals ([3,nc] array, float)
                  jkfold  # face indices adjacent to crease lines ([2,nc] array, int)
                  ):
    dispf = disp.reshape([6,nf])
    drho = np.zeros((nc,6*nf))
    ii = np.arange(nc)
    # rotation matrices
    rot0 = rot3D(dispf[3:6,jkfold[0,:]])
    rot1 = rot3D(dispf[3:6,jkfold[1,:]])
    # rotated vectors
    bb0 = np.sum(rot0*bb, axis=1)
    nn0 = np.sum(rot0*nn[:,jkfold[0,:]], axis=1)
    nn1 = np.sum(rot1*nn[:,jkfold[1,:]], axis=1)
    # cosine and sine of folding angles
    cos = np.sum(nn1*nn0, axis=0)
    sin = np.sum(nn1*bb0, axis=0)
    ll2 = sin**2 + cos**2
    for i in range(3):
        # derivatives of rotation matrices
        drot0 = drot3D(dispf[3:6,jkfold[0,:]],i)
        drot1 = drot3D(dispf[3:6,jkfold[1,:]],i)
        # derivatives of rotated vectors
        dbb0 = np.sum(drot0*bb, axis=1)
        dnn0 = np.sum(drot0*nn[:,jkfold[0,:]], axis=1)
        dnn1 = np.sum(drot1*nn[:,jkfold[1,:]], axis=1)
        # derivatives of cosine and sine of folding angles
        dcos_0 = np.sum(nn1*dnn0, axis=0)
        dsin_0 = np.sum(nn1*dbb0, axis=0)
        dcos_1 = np.sum(dnn1*nn0, axis=0)
        dsin_1 = np.sum(dnn1*bb0, axis=0)
        # derivatives of folding angles
        drho[ii,jkfold[0,:]+nf*i+3*nf] = (-sin*dcos_0 + cos*dsin_0)/ll2
        drho[ii,jkfold[1,:]+nf*i+3*nf] = (-sin*dcos_1 + cos*dsin_1)/ll2
    return drho

### second-order derivative of folding angle
def diff2foldangle(disp,   # generalized displacement ([6nf] array, float)
                   nf,     # number of faces (int)
                   nc,     # number of crease lines (int)
                   nn,     # unit normal vectors of faces ([3,nf] array, float)
                   bb,     # cross products of crease line vectors and face normals ([3,nc] array, float)
                   jkfold  # face indices adjacent to crease lines ([2,nc] array, int)
                   ):
    dispf = disp.reshape([6,nf])
    ddrho = np.zeros((nc,6*nf,6*nf))
    ii = np.arange(nc)
    # rotation matrices
    rot0 = rot3D(dispf[3:6,jkfold[0,:]])
    rot1 = rot3D(dispf[3:6,jkfold[1,:]])
    # rotated vectors
    bb0 = np.sum(rot0*bb, axis=1)
    nn0 = np.sum(rot0*nn[:,jkfold[0,:]], axis=1)
    nn1 = np.sum(rot1*nn[:,jkfold[1,:]], axis=1)
    # cosine and sine of folding angles
    cos = np.sum(nn1*nn0, axis=0)
    sin = np.sum(nn1*bb0, axis=0)
    ll2 = sin**2 + cos**2
    for i in range(3):
        # first-order derivatives of rotation matrices
        drot0_i = drot3D(dispf[3:6,jkfold[0,:]],i)
        drot1_i = drot3D(dispf[3:6,jkfold[1,:]],i)
        # first-order derivatives of rotated vectors
        dbb0_i = np.sum(drot0_i*bb, axis=1)
        dnn0_i = np.sum(drot0_i*nn[:,jkfold[0,:]], axis=1)
        dnn1_i = np.sum(drot1_i*nn[:,jkfold[1,:]], axis=1)
        # first-order derivatives of cosine and sine of folding angles
        dcos_0i = np.sum(nn1*dnn0_i, axis=0)
        dsin_0i = np.sum(nn1*dbb0_i, axis=0)
        dcos_1i = np.sum(dnn1_i*nn0, axis=0)
        dsin_1i = np.sum(dnn1_i*bb0, axis=0)
        for j in range(3):
            # first-order derivatives of rotation matrices
            drot0_j = drot3D(dispf[3:6,jkfold[0,:]],j)
            drot1_j = drot3D(dispf[3:6,jkfold[1,:]],j)
            # first-order derivatives of rotated vectors
            dbb0_j = np.sum(drot0_j*bb, axis=1)
            dnn0_j = np.sum(drot0_j*nn[:,jkfold[0,:]], axis=1)
            dnn1_j = np.sum(drot1_j*nn[:,jkfold[1,:]], axis=1)
            # first-order derivatives of cosine and sine of folding angles
            dcos_0j = np.sum(nn1*dnn0_j, axis=0)
            dsin_0j = np.sum(nn1*dbb0_j, axis=0)
            dcos_1j = np.sum(dnn1_j*nn0, axis=0)
            dsin_1j = np.sum(dnn1_j*bb0, axis=0)
            # second-order derivatives of rotation matrices
            ddrot0_ij = ddrot3D(dispf[3:6,jkfold[0,:]],i,j)
            ddrot1_ij = ddrot3D(dispf[3:6,jkfold[1,:]],i,j)
            # second-order derivatives of rotated vectors
            ddbb0_ij = np.sum(ddrot0_ij*bb, axis=1)
            ddnn0_ij = np.sum(ddrot0_ij*nn[:,jkfold[0,:]], axis=1)
            ddnn1_ij = np.sum(ddrot1_ij*nn[:,jkfold[1,:]], axis=1)
            # second-order derivatives of cosine and sine of folding angles
            ddcos_0i0j = np.sum(nn1*ddnn0_ij, axis=0)
            ddcos_0i1j = np.sum(dnn1_j*dnn0_i, axis=0)
            ddcos_0j1i = np.sum(dnn1_i*dnn0_j, axis=0)
            ddcos_1i1j = np.sum(ddnn1_ij*nn0, axis=0)
            ddsin_0i0j = np.sum(nn1*ddbb0_ij, axis=0)
            ddsin_0i1j = np.sum(dnn1_j*dbb0_i, axis=0)
            ddsin_0j1i = np.sum(dnn1_i*dbb0_j, axis=0)
            ddsin_1i1j = np.sum(ddnn1_ij*bb0, axis=0)
            # second-order derivatives of folding angles
            ddrho1_0i0j = ((sin**2 - cos**2) * (dcos_0i*dsin_0j + dcos_0j*dsin_0i) + 2*sin*cos * (dcos_0i*dcos_0j - dsin_0j*dsin_0i)) / (ll2**2)
            ddrho1_0i1j = ((sin**2 - cos**2) * (dcos_0i*dsin_1j + dcos_1j*dsin_0i) + 2*sin*cos * (dcos_0i*dcos_1j - dsin_1j*dsin_0i)) / (ll2**2)
            ddrho1_0j1i = ((sin**2 - cos**2) * (dcos_0j*dsin_1i + dcos_1i*dsin_0j) + 2*sin*cos * (dcos_0j*dcos_1i - dsin_1i*dsin_0j)) / (ll2**2)
            ddrho1_1i1j = ((sin**2 - cos**2) * (dcos_1i*dsin_1j + dcos_1j*dsin_1i) + 2*sin*cos * (dcos_1i*dcos_1j - dsin_1j*dsin_1i)) / (ll2**2)
            ddrho2_0i0j = (-sin * ddcos_0i0j + cos * ddsin_0i0j) / ll2
            ddrho2_0i1j = (-sin * ddcos_0i1j + cos * ddsin_0i1j) / ll2
            ddrho2_0j1i = (-sin * ddcos_0j1i + cos * ddsin_0j1i) / ll2
            ddrho2_1i1j = (-sin * ddcos_1i1j + cos * ddsin_1i1j) / ll2
            ddrho[ii,jkfold[0,:]+nf*i+3*nf,jkfold[0,:]+nf*j+3*nf] = ddrho1_0i0j + ddrho2_0i0j
            ddrho[ii,jkfold[0,:]+nf*i+3*nf,jkfold[1,:]+nf*j+3*nf] = ddrho1_0i1j + ddrho2_0i1j
            ddrho[ii,jkfold[1,:]+nf*j+3*nf,jkfold[0,:]+nf*i+3*nf] = ddrho1_0i1j + ddrho2_0i1j
            ddrho[ii,jkfold[0,:]+nf*j+3*nf,jkfold[1,:]+nf*i+3*nf] = ddrho1_0j1i + ddrho2_0j1i
            ddrho[ii,jkfold[1,:]+nf*i+3*nf,jkfold[0,:]+nf*j+3*nf] = ddrho1_0j1i + ddrho2_0j1i
            ddrho[ii,jkfold[1,:]+nf*i+3*nf,jkfold[1,:]+nf*j+3*nf] = ddrho1_1i1j + ddrho2_1i1j
    return ddrho

### first-order derivative of vertex displacement
def diffvertexdisp(disp,  # generalized displacement ([6nf] array, float)
                   nf,    # number of faces (int)
                   nv,    # number of vertices (int)
                   aa,    # inner angles around vertices ([*,nv] array, float)
                   saa,   # sum of inner angles around vertices ([nv] array, float)
                   dd,    # vectors from barycenters to vertices of faces ([3,*] array, float)
                   jkvf,  # face indices around vertices ([*,nv] array, int)
                   jkvad  # indices of aa and dd around vertices ([*,nv] array, int)
                   ):
    dispf = disp.reshape([6,nf])
    ddispv = np.zeros((3*nv,6*nf))
    ii = np.arange(nv)
    # first-order derivative of vertex displacement
    for i in range(jkvf.shape[0]):
        aav = aa[jkvad[i,:]]
        ddv = dd[:,jkvad[i,:]]
        for j in range(3):
            # w.r.t. translation
            ddispv[ii+nv*j,jkvf[i,:]+nf*j] += aav/saa
            # w.r.t. rotation
            drot = drot3D(dispf[3:6,jkvf[i,:]],j)
            dddv = np.sum(drot*ddv, axis=1)
            ddispv[ii     ,jkvf[i,:]+nf*j+3*nf] += (aav/saa)*dddv[0,:]
            ddispv[ii+nv  ,jkvf[i,:]+nf*j+3*nf] += (aav/saa)*dddv[1,:]
            ddispv[ii+2*nv,jkvf[i,:]+nf*j+3*nf] += (aav/saa)*dddv[2,:]
    return ddispv

### second-order derivative of vertex displacement
def diff2vertexdisp(disp,  # generalized displacement ([6nf] array, float)
                    nf,    # number of faces (int)
                    nv,    # number of vertices (int)
                    aa,    # inner angles around vertices ([*,nv] array, float)
                    saa,   # sum of inner angles around vertices ([nv] array, float)
                    dd,    # vectors from barycenters to vertices of faces ([3,*] array, float)
                    jkvf,  # face indices around vertices ([*,nv] array, int)
                    jkvad  # indices of aa and dd around vertices ([*,nv] array, int)
                    ):
    dispf = disp.reshape([6,nf])
    dddispv = np.zeros((3*nv,6*nf,6*nf))
    ii = np.arange(nv)
    # second-order derivative of vertex displacement
    for i in range(jkvf.shape[0]):
        aav = aa[jkvad[i,:]]
        ddv = dd[:,jkvad[i,:]]
        for j in range(3):
            for k in range(3):
                # w.r.t. rotation (translation = 0)
                ddrot = ddrot3D(dispf[3:6,jkvf[i,:]],j,k)
                ddddv = np.sum(ddrot*ddv, axis=1)
                dddispv[ii     ,jkvf[i,:]+nf*j+3*nf,jkvf[i,:]+nf*k+3*nf] += (aav/saa)*ddddv[0,:]
                dddispv[ii+nv  ,jkvf[i,:]+nf*j+3*nf,jkvf[i,:]+nf*k+3*nf] += (aav/saa)*ddddv[1,:]
                dddispv[ii+2*nv,jkvf[i,:]+nf*j+3*nf,jkvf[i,:]+nf*k+3*nf] += (aav/saa)*ddddv[2,:]
    return dddispv

### total potential energy
def potential(disp,    # generalized displacement ([6nf] array, float)
              nf,      # number of faces (int)
              nc,      # number of crease lines (int)
              nn,      # unit normal vectors of faces ([3,nf] array, float)
              bb,      # cross products of crease line vectors and face normals ([3,nc] array, float)
              dd,      # vectors from barycenters to vertices of faces ([3,*] array, float)
              jkfold,  # face indices adjacent to crease lines ([2,nc] array, int)
              jkcomp,  # face and ref. vector indices adjacent to connecter vertices ([4,#connect] array, int)
              kkc,     # rotation stiffness of crease lines ([nc] array, float)
              rho0,    # initial angles of springs at crease lines ([nc] array, float)
              zz0,     # reference height of gravity potential ([nf] array, float)
              wf,      # face weights ([nf] array, float)
              kkf      # stiffness of face connecting springs (float)
              ):
    # compute displacements
    dispf = disp.reshape([6,nf])
    rho = foldangle(disp, nf, nc, nn, bb, jkfold)
    comp = compatibility(disp, nf, dd, jkcomp)
    # compute total potential energy
    energy_h = np.sum(kkc*(rho-rho0)**2)/2.
    energy_c = kkf*np.sum(comp**2)/2.
    energy_g = np.sum(wf*(dispf[2,:]+zz0))
    return energy_h, energy_c, energy_g

### first-order derivative of total potential energy
def diffpotential(disp,    # generalized displacement ([6nf] array, float)
                  nf,      # number of faces (int)
                  nc,      # number of crease lines (int)
                  nn,      # unit normal vectors of faces ([3,nf] array, float)
                  bb,      # cross products of crease line vectors and face normals ([3,nc] array, float)
                  dd,      # vectors from barycenters to vertices of faces ([3,*] array, float)
                  jkfold,  # face indices adjacent to crease lines ([2,nc] array, int)
                  jkcomp,  # face and ref. vector indices adjacent to connecter vertices ([4,#connect] array, int)
                  kkc,     # rotation stiffness of crease lines ([nc] array, float)
                  rho0,    # initial angles of springs at crease lines ([nc] array, float)
                  zz0,     # reference height of gravity potential ([nf] array, float)
                  wf,      # face weights ([nf] array, float)
                  kkf      # stiffness of face connecting springs (float)
                  ):
    # compute displacements
    dispf = disp.reshape([6,nf])
    rho = foldangle(disp, nf, nc, nn, bb, jkfold)
    drho = difffoldangle(disp, nf, nc, nn, bb, jkfold)
    comp = compatibility(disp, nf, dd, jkcomp)
    dcomp = diffcompatibility(disp, nf, dd, jkcomp)
    # compute first-order derivative of total potential energy
    denergy_h = np.sum((kkc*(rho-rho0)).reshape([nc,1])*drho, axis=0)
    denergy_c = kkf*np.sum(comp.reshape([len(comp),1])*dcomp, axis=0)
    denergy_g = np.zeros_like(denergy_h)
    denergy_g[2*nf:3*nf] += wf
    return denergy_h, denergy_c, denergy_g

### second-order derivative of total potential energy
def diff2potential(disp,    # generalized displacement ([6nf] array, float)
                   nf,      # number of faces (int)
                   nc,      # number of crease lines (int)
                   nn,      # unit normal vectors of faces ([3,nf] array, float)
                   bb,      # cross products of crease line vectors and face normals ([3,nc] array, float)
                   dd,      # vectors from barycenters to vertices of faces ([3,*] array, float)
                   jkfold,  # face indices adjacent to crease lines ([2,nc] array, int)
                   jkcomp,  # face and ref. vector indices adjacent to connecter vertices ([4,#connect] array, int)
                   kkc,     # rotation stiffness of crease lines ([nc] array, float)
                   rho0,    # initial angles of springs at crease lines ([nc] array, float)
                   zz0,     # reference height of gravity potential ([nf] array, float)
                   wf,      # face weights ([nf] array, float)
                   kkf      # stiffness of face connecting springs (float)
                   ):
    # compute displacements
    dispf = disp.reshape([6,nf])
    rho = foldangle(disp, nf, nc, nn, bb, jkfold)
    drho = difffoldangle(disp, nf, nc, nn, bb, jkfold)
    ddrho = diff2foldangle(disp, nf, nc, nn, bb, jkfold)
    comp = compatibility(disp, nf, dd, jkcomp)
    dcomp = diffcompatibility(disp, nf, dd, jkcomp)
    ddcomp = diff2compatibility(disp, nf, dd, jkcomp)
    # compute second-order derivative of total potential energy
    ddenergy_h = np.dot(drho.T,(kkc.reshape([nc,1])*drho))
    ddenergy_h += np.sum((kkc*(rho-rho0)).reshape([nc,1,1])*ddrho, axis=0)
    ddenergy_c = kkf*np.dot(dcomp.T,dcomp)
    ddenergy_c += kkf*np.sum(comp.reshape([len(comp),1,1])*ddcomp, axis=0)
    ddenergy_g = np.zeros_like(ddenergy_h)
    return ddenergy_h, ddenergy_c, ddenergy_g


##################################################
### construction of panel-pin model
def panelpin(vert, # vertex positions ([3,nv] array, float)
             nv,   # number of vertices
             face, # list of face vertices ([list of list], int)
             nf,   # number of faces
             crs,  # endpoints of crease lines and adjacent faces ([4, nc] array, int)
             glu,  # glued faces ([2, ng] array, int)
             wfa,  # face weight per unit area
             kkc0  # rotation stiffness of crease line per unit length (float)
             ):
    # list of connectivity of panels
    connect = np.zeros((3,0), int)
    for i in range(nv):
        include = np.zeros(0, int)
        for j in range(nf):
            if i in face[j]:
                include = np.append(include, j)
        include = np.unique(include)
        if len(include) == 2:
            connect = np.append(connect, np.array([[i],[include[0]],[include[1]]]), axis=1)
        elif len(include) >= 3:
            for j in range(len(include)-1):
                for k in range(j+1,len(include)):
                    connect = np.append(connect, np.array([[i],[include[j]],[include[k]]]), axis=1)
    #  rotation stiffness of crease lines and unit vectors along crease lines
    ee = vert[:,crs[1,:]] - vert[:,crs[0,:]]
    kkc = kkc0*np.linalg.norm(ee, axis=0)
    ee = ee/np.linalg.norm(ee, axis=0)
    # area, barycenter and unit normal of face
    area = np.zeros(nf)
    bf = np.zeros((3,nf))
    nn = np.zeros((3,nf))
    for i in range(nf):
        for j in range(1,len(face[i])-1):
            v1 = vert[:,face[i][j]] - vert[:,face[i][0]]
            v2 = vert[:,face[i][j+1]] - vert[:,face[i][0]]
            nn[:,i] += np.cross(v1,v2)/2.
        area[i] = np.linalg.norm(nn[:,i])
        nn[:,i] /= area[i]
        for j in range(1,len(face[i])-1):
            v1 = vert[:,face[i][j]] - vert[:,face[i][0]]
            v2 = vert[:,face[i][j+1]] - vert[:,face[i][0]]
            bfi = (vert[:,face[i][0]] + vert[:,face[i][j]] + vert[:,face[i][j+1]])/3.
            bfi *= np.dot(nn[:,i],np.cross(v1,v2)/2.)
            bf[:,i] += bfi
        bf[:,i] /= area[i]
    # update barycenter if panels are glued
    for i in range(glu.shape[1]):
        bf0 = np.copy(bf[:,glu[0,i]])
        bf1 = np.copy(bf[:,glu[1,i]])
        bf[:,glu[0,i]] = (bf0 + bf1)/2.
        bf[:,glu[1,i]] = (bf0 + bf1)/2.
    # reference height for computation of gravity potential
    zz0 = bf[2,:]
    # face weight
    wf = wfa*area
    # inner angles and vectors from barycenters to vertices of faces
    aa = np.zeros(0)
    dd = np.zeros((3,0))
    jk = np.full((nf,nv), -1, dtype=int)
    k = 0
    for i in range(nf):
        nvf = len(face[i])
        for j in range(nvf):
            r0 = vert[:,face[i][j]]
            r1 = vert[:,face[i][(j+1)%nvf]]
            r2 = vert[:,face[i][(j-1)%nvf]]
            v1 = (r1-r0)/np.linalg.norm(r1-r0)
            v2 = (r2-r0)/np.linalg.norm(r2-r0)
            aa = np.append(aa, np.arccos(np.clip(np.dot(v1,v2), -1, 1)))
            dd = np.append(dd, (r0-bf[:,i]).reshape([3,1]), axis=1)
            jk[i,face[i][j]] = k
            k += 1
    aa = np.append(aa, 0.)
    dd = np.append(dd, np.zeros((3,1)), axis=1)
    # index list for computation of folding angle and compatibility condition
    jkfold = crs[2:4,:]
    # index list for computation of compatibility condition
    jkcomp = np.zeros((4,connect.shape[1]), int)
    for i in range(connect.shape[1]):
        jkcomp[0,i] = connect[1,i]
        jkcomp[1,i] = connect[2,i]
        jkcomp[2,i] = jk[connect[1,i],connect[0,i]]
        jkcomp[3,i] = jk[connect[2,i],connect[0,i]]
    # index list for computation of vertex displacement
    iii = np.zeros(nv, int)
    jkvf = np.full((1,nv), -1, dtype=int)
    jkvad = np.full((1,nv), -1, dtype=int)
    for i in range(nf):
        for j in range(len(face[i])):
            iii[face[i][j]] += 1
            if jkvf.shape[0] < iii[face[i][j]]:
                jkvf = np.append(jkvf, np.full((1,nv), -1, dtype=int), axis=0)
                jkvad = np.append(jkvad, np.full((1,nv), -1, dtype=int), axis=0)
            jkvf[iii[face[i][j]]-1,face[i][j]] = i
            jkvad[iii[face[i][j]]-1,face[i][j]] = jk[i,face[i][j]]
    # sum of face inner angles around vertices
    saa = np.sum(aa[jkvad], axis=0)
    # cross products of crease line vectors and face normals
    bb = np.cross(ee, nn[:,jkfold[0,:]], axis=0)
    # indices of independent face displacement
    ivar = np.zeros((6,0), int)
    iadd = np.zeros((6,0), int)
    iall = np.zeros((1,nf), int)
    for i in range(nf):
        if i in glu[0,:]:
            j = glu[1,np.where(glu[0,:]==i)[0][0]]
            ivar = np.append(ivar, np.array([[i],[i+nf],[i+2*nf],[i+3*nf],[i+4*nf],[i+5*nf]]), axis=1)
            iadd = np.append(iadd, np.array([[j],[j+nf],[j+2*nf],[j+3*nf],[j+4*nf],[j+5*nf]]), axis=1)
            iall[0,i] = ivar.shape[1]-1
            iall[0,j] = ivar.shape[1]-1
        elif i in glu[1,:]:
            continue
        else:
            ivar = np.append(ivar, np.array([[i],[i+nf],[i+2*nf],[i+3*nf],[i+4*nf],[i+5*nf]]), axis=1)
            iadd = np.append(iadd, np.array([[-1],[-1],[-1],[-1],[-1],[-1]]), axis=1)
            iall[0,i] = ivar.shape[1]-1
    nvar = ivar.shape[1]
    ivar = np.ravel(ivar)
    iadd = np.ravel(iadd)
    iall = np.hstack((iall, iall+nvar, iall+2*nvar, iall+3*nvar, iall+4*nvar, iall+5*nvar))
    iall = np.ravel(iall)
    return dd, nn, bb, aa, saa, kkc, wf, zz0, jkcomp, jkfold, jkvf, jkvad, ivar, iadd, iall



##################################################
# initial modeling
##################################################
alpha = 2. * math.atan(math.sqrt(math.sqrt(2)-1))
vert, face, crs, glu = modelling(alpha, height, ni, nj)
nv = vert.shape[1]
nf = face.shape[1]
nc = crs.shape[1]
ng = glu.shape[1]

### initialize output directory
if os.path.isdir('./result'):
    shutil.rmtree('./result')
os.makedirs('./result')
np.savetxt('./result/vertex.dat', vert.T)
np.savetxt('./result/face.dat', face.T, fmt='%i')
np.savetxt('./result/crease.dat', crs.T, fmt='%i')
np.savetxt('./result/glued.dat', glu.T, fmt='%i')

print("== Model info. ==")
print("# vertices:     %i"%(nv))
print("# faces:        %i"%(nf))
print("# crease lines: %i"%(nc))
print("# glued face:   %i"%(ng))
print()

### rearrangement of face data
face_arr = np.copy(face)
face = []
for i in range(nf):
    face.append([face_arr[0,i], face_arr[1,i], face_arr[2,i]])

### initialize face displacement
disp = np.zeros(6*nf)

### initial folding angles
dd, nn, bb, aa, saa, kkc, wf, zz0,\
jkcomp, jkfold, jkvf, jkvad, ivar, iadd, iall =\
panelpin(vert, nv, face, nf, crs, glu, wfa, kkc0)
rho0 = foldangle(disp, nf, nc, nn, bb, jkfold)
#rho0 = np.zeros(nc)



##################################################
# analysis for each path parameter
##################################################
### initialize list of results
vert_all = []
dof_all = []
singular_all = []
eigen_m_all = []
dispf_m_all = []
dispv_m_all = []
coef_all = []
quaderr_all = []
eigen_e_all = []
dispf_e_all = []
dispv_e_all = []
energy_all = []

### specify evaluated theta
theta_all = (np.arange(nstep)/(nstep-1))*(theta_max-theta_min) + theta_min
iii = np.argmin(np.abs(theta_all-alpha))
theta_all[iii] = alpha

### start analysis
for itr in range(nstep):
    ### update parameter
    print("== Analysis %i/%i; theta = %.3f deg. =="%(itr+1,nstep,np.rad2deg(theta_all[itr])))

    ### update model
    vert, _, _, _ = modelling(theta_all[itr], height, ni, nj)
    vert_all.append(vert)

    ### construct panel-pin model
    dd, nn, bb, aa, saa, kkc, wf, zz0,\
    jkcomp, jkfold, jkvf, jkvad, ivar, iadd, iall =\
    panelpin(vert, nv, face, nf, crs, glu, wfa, kkc0)

    ##################################################
    ### compatibility matrix
    dcomp0 = diffcompatibility(disp, nf, dd, jkcomp)
    dcomp0 = np.append(dcomp0, np.zeros((dcomp0.shape[0],1)), axis=1)
    dcomp = dcomp0[:,ivar] + dcomp0[:,iadd]

    # infinitesimal mechanism analysis
    fmode, sigma, dmodeT = np.linalg.svd(dcomp)
    dmode = dmodeT.T
    dor = dcomp.shape[0] - np.count_nonzero(sigma > dtol)
    dof = dcomp.shape[1] - np.count_nonzero(sigma > dtol)

    ### update infinitesimal mechanism modes
    dmodef_m = dmode[:,dmode.shape[1]-dof:dmode.shape[1]]
    dmodef_m = dmodef_m[iall,:]
    drho = difffoldangle(disp, nf, nc, nn, bb, jkfold)
    dmoder = np.dot(drho,dmodef_m)
    stiff_m = np.dot(np.dot(dmoder.T,np.diag(kkc)),dmoder)
    w_m, v_m = np.linalg.eigh(stiff_m)
    arg = np.argsort(w_m)[::-1]
    w_m = w_m[arg]
    v_m = v_m[:,arg]
    dmodef_m = np.dot(dmodef_m,v_m)
    denergy_h, denergy_c, denergy_g = diffpotential(disp, nf, nc, nn, bb, dd, jkfold, jkcomp, kkc, rho0, zz0, wf, kkf)
    dpotential = denergy_h + denergy_g
    denergy = np.dot(dmodef_m.T,dpotential)
    for i in range(len(denergy)):
        if denergy[i] > 0:
            dmodef_m[:,i] *= -1

    ### indices of rigid-body motion
    irigid = np.where(w_m < dtol)[0]

    ### mechanism modes with respect to vertex displacement
    ddispv = diffvertexdisp(disp, nf, nv, aa, saa, dd, jkvf, jkvad)
    dmodev_m = np.dot(ddispv,dmodef_m)
    for i in range(dof):
        dmodev_m[:,i] /= np.linalg.norm(dmodev_m[:,i])

    ### output
    print("== 1st-order infinitesimal mechanism ==")
    print("# rigid-body motion modes:       %i"%(len(irigid)))
    print("# infinitesimal mechanism modes: %i"%(dof-len(irigid)))
    print("# self-equilibrium force modes:  %i"%(dor))

    ### add results to lists
    dof_all.append(dof-len(irigid))
    singular_all.append(np.sort(sigma))
    eigen_m_all.append(w_m)
    dispf_m_all.append(dmodef_m)
    dispv_m_all.append(dmodev_m)

    ##################################################
    ### Hessian of incompatibility vector
    ddcomp0 = diff2compatibility(disp, nf, dd, jkcomp)

    ### matrix of quadratic form for coefficients of infinitesimal mechanism modes
    ddcomp0 = np.append(ddcomp0, np.zeros((ddcomp0.shape[0],ddcomp0.shape[1],1)), axis=2)
    ddcomp0 = np.append(ddcomp0, np.zeros((ddcomp0.shape[0],1,ddcomp0.shape[2])), axis=1)
    ddcomp = ddcomp0[:,:,ivar] + ddcomp0[:,:,iadd]
    ddcomp = ddcomp[:,ivar,:] + ddcomp[:,iadd,:]
    quad = np.zeros((dor,dof,dof))
    dmodef_m2 = dmodef_m[ivar,:]
    for i in range(dor):
        quad_i = ddcomp * fmode[:,fmode.shape[1]-dor+i].reshape([fmode.shape[0],1,1])
        quad_i = np.sum(quad_i, axis=0)
        quad[i,:,:] = np.dot(np.dot(dmodef_m2.T,quad_i),dmodef_m2)

    coef0 = np.zeros((dor,int(dof*(dof+1)/2+0.1)))
    indices = np.zeros((dof,dof), int)
    for i in range(dor):
        ii = 0
        for j in range(dof):
            for k in range(j,dof):
                if k == j:
                    coef0[i,ii] = quad[i,j,j]
                    indices[j,j] = ii
                else:
                    coef0[i,ii] = quad[i,j,k] + quad[i,k,j]
                    indices[j,k] = ii
                    indices[k,j] = ii
                ii += 1

    ### rearrangement of coef
    coef, _, _, _ = rref(coef0)
    cnorm = np.linalg.norm(coef, axis=1)
    ii = np.where(cnorm > dtol)[0]
    coef = coef[ii,:]
    nquad = np.count_nonzero(cnorm > dtol)

    ### infinitesimal mechanism modes satisfies quadratic eqns or not
    quaderr = np.zeros((dor,dof))
    for i in range(dor):
        for j in range(dof):
            aa = np.zeros(dof)
            aa[j] = 1.
            quaderr[i,j] = np.dot(np.dot(quad[i], aa), aa)

    ### output
    print("== 2nd-order infinitesimal mechanism ==")
    print("# existence condition of modes:  %i"%(nquad))
    print("Error in infinitesimal modes:")
    print('Mode  Error      Eigen')
    for i in range(dof):
        print('%i     %.3e  %.3e'%(i,np.max(np.abs(quaderr[:,i])),w_m[i]))

    ### add results to lists
    coef_all.append(coef)
    quaderr_all.append(np.max(np.abs(quaderr),axis=0))

    ##################################################
    ### stiffness matrix
    ddenergy_h, ddenergy_c, ddenergy_g = diff2potential(disp, nf, nc, nn, bb, dd, jkfold, jkcomp, kkc, rho0, zz0, wf, kkf)
    ddenergy = ddenergy_h + ddenergy_c
    ddenergy = np.append(ddenergy, np.zeros((ddenergy.shape[0],1)), axis=1)
    ddenergy = np.append(ddenergy, np.zeros((1,ddenergy.shape[1])), axis=0)
    stiffness = ddenergy[:,ivar] + ddenergy[:,iadd]
    stiffness = stiffness[ivar,:] + stiffness[iadd,:]

    ### eigenvalue analysis
    w_e, v_e = np.linalg.eigh(stiffness)
    arg = np.argsort(w_e)
    w_e = w_e[arg]
    v_e = v_e[:,arg]
    dmodef_e = v_e[iall,:]

    ### energy for hinge and face springs
    energy_h = np.zeros(len(w_e))
    energy_c = np.zeros(len(w_e))
    for i in range(len(w_e)):
        energy_h[i] = np.dot(dmodef_e[:,i], np.dot(ddenergy_h, dmodef_e[:,i]))/2.
        energy_c[i] = np.dot(dmodef_e[:,i], np.dot(ddenergy_c, dmodef_e[:,i]))/2.
    energy = np.hstack((energy_h.reshape([len(w_e),1]), energy_c.reshape([len(w_e),1])))

    ### vertex displacement modes
    dmodev_e = np.dot(ddispv,dmodef_e)
    for i in range(dmodev_e.shape[1]):
        dmodev_e[:,i] /= np.linalg.norm(dmodev_e[:,i])

    ### output
    print("== Elastic stiffness analysis ==")
    print("Smallest %i eigenvalues:"%(nprint))
    for i in range(nprint):
        print('%i  %.3e'%(i,w_e[i]))
    print()

    ### add results to list
    eigen_e_all.append(w_e)
    dispf_e_all.append(dmodef_e)
    dispv_e_all.append(dmodev_e)
    energy_all.append(energy)



##################################################
# output analysis results
##################################################
### vertex positions
with open('./result/vertices.dat', 'w') as f:
    f.write('theta')
    for i in range(nv):
        f.write(' x%i y%i z%i'%(i,i,i))
    f.write('\n')
    for itr in range(nstep):
        f.write('%.16e'%(theta_all[itr]))
        for i in range(nv):
            f.write(' %.16e %.16e %.16e'%(vert_all[itr][0,i],vert_all[itr][1,i],vert_all[itr][2,i]))
        f.write('\n')

### singular values of compatibility matrices
with open('./result/singular_mech.dat', 'w') as f:
    f.write('theta')
    for i in range(len(singular_all[0])):
        f.write(' S%i'%(i+1))
    f.write('\n')
    for itr in range(nstep):
        f.write('%.16e'%(theta_all[itr]))
        for i in range(len(singular_all[itr])):
            f.write(' %.16e'%(singular_all[itr][i]))
        f.write('\n')

### eigenvalues of stiffness matrices for mechanism analysis
maxdof = 0
for i in range(nstep):
    dofi = len(eigen_m_all[i])
    if dofi > maxdof:
        maxdof = dofi
with open('./result/eigen_mech.dat', 'w') as f:
    f.write('theta dof')
    for i in range(maxdof):
        f.write(' E%i'%(i+1))
    f.write('\n')
    for itr in range(nstep):
        f.write('%.16e %i'%(theta_all[itr], dof_all[itr]))
        for i in range(len(eigen_m_all[itr])):
            f.write(' %.16e'%(eigen_m_all[itr][i]))
        f.write('\n')

### displacement modes for face panels in mechaism analysis
with open('./result/modeface_mech.dat', 'w') as f:
    f.write('index theta eigenvalue')
    for i in range(nf):
        f.write(' Ux%i Uy%i Uz%i Rx%i Ry%i Rz%i'%(i,i,i,i,i,i))
    f.write('\n')
    for itr in range(nstep):
        for i in range(dispf_m_all[itr].shape[1]):
            f.write('%i %.8e %.8e'%(itr,theta_all[itr],eigen_m_all[itr][i]))
            dispf = dispf_m_all[itr][:,i].reshape([6,nf])
            for j in range(nf):
                f.write(' %.8e %.8e %.8e %.8e %.8e %.8e'%(dispf[0,j],dispf[1,j],dispf[2,j],dispf[3,j],dispf[4,j],dispf[5,j]))
            f.write('\n')

### displacement modes for vertices in mechanisn analysis
with open('./result/modevert_mech.dat', 'w') as f:
    f.write('index theta eigenvalue')
    for i in range(nv):
        f.write(' Ux%i Uy%i Uz%i'%(i,i,i))
    f.write('\n')
    for itr in range(nstep):
        for i in range(dispv_m_all[itr].shape[1]):
            f.write('%i %.8e %.8e'%(itr,theta_all[itr],eigen_m_all[itr][i]))
            dispv = dispv_m_all[itr][:,i].reshape([3,nv])
            for j in range(nv):
                f.write(' %.8e %.8e %.8e'%(dispv[0,j],dispv[1,j],dispv[2,j]))
            f.write('\n')

### existence condition of second-order mechanism
with open('./result/secondorder.dat', 'w') as f:
    f.write('theta Neqs Nblk')
    for i in range(maxdof):
        f.write(' Err%i'%(i+1))
    f.write('\n')
    for itr in range(nstep):
        f.write('%.16e %i'%(theta_all[itr],coef_all[itr].shape[0]))
        f.write(' %i'%(np.count_nonzero(quaderr_all[itr] > dtol)))
        for i in range(len(quaderr_all[itr])):
            f.write(' %.16e'%(quaderr_all[itr][i]))
        f.write('\n')

### eigenvalues of stiffness matrices for elastic analysis
with open('./result/eigen_elastic.dat', 'w') as f:
    f.write('theta')
    for i in range(len(eigen_e_all[0])):
        f.write(' E%i'%(i+1))
    f.write('\n')
    for itr in range(nstep):
        f.write('%.16e'%(theta_all[itr]))
        for i in range(len(eigen_e_all[itr])):
            f.write(' %.16e'%(eigen_e_all[itr][i]))
        f.write('\n')

### displacement modes for face panels in elastic analysis
with open('./result/modeface_elastic.dat', 'w') as f:
    f.write('index theta eigenvalue')
    for i in range(nf):
        f.write(' Ux%i Uy%i Uz%i Rx%i Ry%i Rz%i'%(i,i,i,i,i,i))
    f.write('\n')
    for itr in range(nstep):
        for i in range(dispf_e_all[itr].shape[1]):
            f.write('%i %.8e %.8e'%(itr,theta_all[itr],eigen_e_all[itr][i]))
            dispf = dispf_e_all[itr][:,i].reshape([6,nf])
            for j in range(nf):
                f.write(' %.8e %.8e %.8e %.8e %.8e %.8e'%(dispf[0,j],dispf[1,j],dispf[2,j],dispf[3,j],dispf[4,j],dispf[5,j]))
            f.write('\n')

### displacement modes for vertices in elastic analysis
with open('./result/modevert_elastic.dat', 'w') as f:
    f.write('index theta eigenvalue')
    for i in range(nv):
        f.write(' Ux%i Uy%i Uz%i'%(i,i,i))
    f.write('\n')
    for itr in range(nstep):
        for i in range(dispv_e_all[itr].shape[1]):
            f.write('%i %.8e %.8e'%(itr,theta_all[itr],eigen_e_all[itr][i]))
            dispv = dispv_e_all[itr][:,i].reshape([3,nv])
            for j in range(nv):
                f.write(' %.8e %.8e %.8e'%(dispv[0,j],dispv[1,j],dispv[2,j]))
            f.write('\n')

### potential energy for each modes in elastic analysis
with open('./result/energy.dat', 'w') as f:
    f.write('index theta eigenvalue hinge face\n')
    for itr in range(nstep):
        for i in range(energy_all[itr].shape[0]):
            f.write('%i %.8e %.8e %.8e %.8e\n'%(itr,theta_all[itr],eigen_e_all[itr][i],energy_all[itr][i,0],energy_all[itr][i,1]))

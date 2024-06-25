import numpy as np
import math
import os
import shutil
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d import art3d
from matplotlib.ticker import ScalarFormatter


##################################################
# functions for plotting model
##################################################
### convert to orthogonal projection
def orthogonal_transformation(zfront, zback):
    a = 2 / (zfront - zback)
    b = -1 * (zfront + zback)
    c = zback
    return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, a, b], [0, 0, 0, c]])

### plot surface
def plotshape(vert,                  # vertex coordinates ([nv,3] array, float)
              face,                  # list of face vertex ([nf,3] array, int)
              fname,                 # file name (str)
              lims=None,             # range of plot area ([3,2] array, float)
              size=120,              # figure size (mm, float)
              edgewidth=0.3,         # width of edge line (float)
              edgecolor='k',         # edge color (str)
              facecolor='lightgray', # face color (str)
              iso=True,              # plot isometric view (bool)
              pln=True,              # plot plan view (bool)
              elv=False,             # plot elevation view (bool)
              eps=True,              # save .eps file (bool)
              png=True,              # save .png file (bool)
              show = False           # show plot result (bool)
              ):
    ### set figure size
    fsize = size * 0.0393701
    ### set range of plot area
    if lims is None:
        span = np.max(np.max(vert, axis=0) - np.min(vert, axis=0))
        center = (np.max(vert, axis=0) + np.min(vert, axis=0))/2.
        lims = np.zeros((3,2))
        lims[:,0] = center - span/2.
        lims[:,1] = center + span/2.
    ### convert to orthogonal projection
    proj3d.persp_transformation = orthogonal_transformation
    ### set figure
    fig = plt.figure(figsize=(fsize,fsize))
    ax = fig.add_subplot(111, projection='3d')
    ### create axis
    #ax.set_axis_off()
    ax.set_xlim(lims[0,0],lims[0,1])
    ax.set_ylim(lims[1,0],lims[1,1])
    ax.set_zlim(lims[2,0],lims[2,1])
    #ax.grid(False)
    ax.w_xaxis.set_pane_color((0., 0., 0., 0.))
    ax.w_yaxis.set_pane_color((0., 0., 0., 0.))
    ax.w_zaxis.set_pane_color((0., 0., 0., 0.))
    ### plot triangular mesh
    suf = art3d.Poly3DCollection(vert[face])
    suf.set(linewidth=edgewidth, edgecolor=edgecolor, facecolor=facecolor)
    ax.add_collection3d(suf)
    ### output
    if iso:
        ax.view_init(elev = 40, azim = -112)
        if eps:
            plt.savefig(fname+'_iso.eps', format='eps')
        if png:
            plt.savefig(fname+'_iso.png', format='png', dpi=450)
    if pln:
        ax.view_init(elev = 90 + 1.0e-10, azim = -90 + 1.0e-10)
        if eps:
            plt.savefig(fname+'_pln.eps', format='eps')
        if png:
            plt.savefig(fname+'_pln.png', format='png', dpi=450)
    if elv:
        ax.view_init(elev = 0 + 1.0e-10, azim = -90 + 1.0e-10)
        if eps:
            plt.savefig(fname+'_elvx.eps', format='eps')
        if png:
            plt.savefig(fname+'_elvx.png', format='png', dpi=450)
        ax.view_init(elev = 0 + 1.0e-10, azim = 0 + 1.0e-10)
        if eps:
            plt.savefig(fname+'_elvy.eps', format='eps')
        if png:
            plt.savefig(fname+'_elvy.png', format='png', dpi=450)
    if show:
        plt.show()
    plt.cla()
    plt.clf()
    plt.close()

### plot displacement
def plotdisp(vert,                  # initial vertex coordinates ([nv,3] array, float)
             dispv,                 # vertex displacements ([nv,3] array, float)
             face,                  # list of face vertex ([nf,3] array, int)
             fname,                 # file name (str)
             lims=None,             # range of plot area ([3,2] array, float)
             size=120,              # figure size (mm, float)
             edgewidth=0.3,         # width of deformed edge line (float)
             edgecolor='k',         # deformed edge color (str)
             facecolor='lightgray', # face color (str)
             linewidth=0.2,         # width of initial edge line (float, not plotted if zero)
             linecolor='dimgray',   # initial edge color (str)
             dpattern=[1.0,1.5],    # dash pattern of initial edge (list of float)
             iso=True,              # plot isometric view (bool)
             pln=True,              # plot plan view (bool)
             elv=False,             # plot elevation view (bool)
             eps=True,              # save .eps file (bool)
             png=True,              # save .png file (bool)
             show = False           # show plot result (bool)
             ):
    ### set figure size
    fsize = size * 0.0393701
    ### set range of plot area
    if lims is None:
        span = np.max(np.max(vert, axis=0) - np.min(vert, axis=0))
        center = (np.max(vert, axis=0) + np.min(vert, axis=0))/2.
        lims = np.zeros((3,2))
        lims[:,0] = center - span/2.
        lims[:,1] = center + span/2.
    ### convert to orthogonal projection
    proj3d.persp_transformation = orthogonal_transformation
    ### set figure
    fig = plt.figure(figsize=(fsize,fsize))
    ax = fig.add_subplot(111, projection='3d')
    ### create axis
    ax.set_axis_off()
    ax.set_xlim(lims[0,0],lims[0,1])
    ax.set_ylim(lims[1,0],lims[1,1])
    ax.set_zlim(lims[2,0],lims[2,1])
    ax.grid(False)
    ax.w_xaxis.set_pane_color((0., 0., 0., 0.))
    ax.w_yaxis.set_pane_color((0., 0., 0., 0.))
    ax.w_zaxis.set_pane_color((0., 0., 0., 0.))
    ### plot deformed triangular mesh
    suf = art3d.Poly3DCollection(vert[face]+dispv[face])
    suf.set(linewidth=edgewidth, edgecolor=edgecolor, facecolor=facecolor)
    ax.add_collection3d(suf)
    ### list of edges
    if linewidth > 1.e-16:
        edge = np.zeros((0,2), int)
        for i in range(face.shape[0]):
            for j in range(3):
                v0 = min([face[i,-3+j], face[i,-2+j]])
                v1 = max([face[i,-3+j], face[i,-2+j]])
                ee = np.array([np.sort(np.array([v0, v1]))])
                edge = np.append(edge, ee, axis=0)
        edge = np.unique(edge, axis=0)
    ### plot edges of initial shape
    if linewidth > 1.e-16:
        for i in range(edge.shape[0]):
            x = np.array([vert[edge[i,0],0],vert[edge[i,1],0]])
            y = np.array([vert[edge[i,0],1],vert[edge[i,1],1]])
            z = np.array([vert[edge[i,0],2],vert[edge[i,1],2]])
            ax.plot(x, y, z, linewidth=linewidth, color=linecolor,
                    linestyle='dashed', dashes=dpattern, zorder=10000)
    ### output
    if iso:
        ax.view_init(elev = 40, azim = -112)
        if eps:
            plt.savefig(fname+'_iso.eps', format='eps')
        if png:
            plt.savefig(fname+'_iso.png', format='png', dpi=450)
    if pln:
        ax.view_init(elev = 90 + 1.0e-10, azim = -90 + 1.0e-10)
        if eps:
            plt.savefig(fname+'_pln.eps', format='eps')
        if png:
            plt.savefig(fname+'_pln.png', format='png', dpi=450)
    if elv:
        ax.view_init(elev = 0 + 1.0e-10, azim = -90 + 1.0e-10)
        if eps:
            plt.savefig(fname+'_elvx.eps', format='eps')
        if png:
            plt.savefig(fname+'_elvx.png', format='png', dpi=450)
        ax.view_init(elev = 0 + 1.0e-10, azim = 0 + 1.0e-10)
        if eps:
            plt.savefig(fname+'_elvy.eps', format='eps')
        if png:
            plt.savefig(fname+'_elvy.png', format='png', dpi=450)
    if show:
        plt.show()
    plt.cla()
    plt.clf()
    plt.close()



##################################################
# read data files
##################################################
### vertex positions for each theta
vert_all0 = np.loadtxt('./result/vertices.dat', skiprows=1)
theta_all = vert_all0[:,0]
nstep = len(theta_all)
nv = int((vert_all0.shape[1]-1)/3+0.01)
vert_all = np.zeros((nstep,nv,3))
for i in range(nstep):
    vert_all[i] = (vert_all0[i,1:]).reshape([nv,3])


### face data
face = np.loadtxt('./result/face.dat', dtype=int)


### eigenvalues of stiffness matrix in mechanism analysis
eigen_m_all = []
lineno = 1
with open('./result/eigen_mech.dat') as f:
    for line in f:
        if lineno > 1:
            eigen_m_add = np.zeros(0)
            l = line.split(' ')
            for i in range(2,len(l)):
                eigen_m_add = np.append(eigen_m_add, float(l[i].strip()))
            eigen_m_all.append(eigen_m_add)
        lineno += 1


### vertex displacement modes in mechanism analysis
dispv_m_all = []
lineno = 0
index_old = 0
dispv_m_add = np.zeros((0,nv,3))
nline = np.loadtxt('./result/modevert_mech.dat',skiprows=1).shape[0]
with open('./result/modevert_mech.dat') as f:
    for line in f:
        if lineno > 0:
            l = line.split(' ')
            index = int(l[0].strip())
            if (index != index_old) or (lineno == nline):
                dispv_m_all.append(dispv_m_add)
                dispv_m_add = np.zeros((0,nv,3))
            dispv_line = np.zeros(0)
            for i in range(3,len(l)):
                dispv_line = np.append(dispv_line, float(l[i].strip()))
            dispv_m_add = np.append(dispv_m_add, dispv_line.reshape([1,nv,3]), axis=0)
            index_old = np.copy(index)
        lineno += 1


### eigenvalues of stiffness matrix in elastic analysis
eigen_e_all = []
lineno = 1
with open('./result/eigen_elastic.dat') as f:
    for line in f:
        if lineno > 1:
            eigen_e_add = np.zeros(0)
            l = line.split(' ')
            for i in range(1,len(l)):
                eigen_e_add = np.append(eigen_e_add, float(l[i].strip()))
            eigen_e_all.append(eigen_e_add)
        lineno += 1


### vertex displacement modes in elastic analysis
dispv_e_all = []
lineno = 0
index_old = 0
dispv_e_add = np.zeros((0,nv,3))
nline = np.loadtxt('./result/modevert_elastic.dat',skiprows=1).shape[0]
with open('./result/modevert_elastic.dat') as f:
    for line in f:
        if lineno > 0:
            l = line.split(' ')
            index = int(l[0].strip())
            if (index != index_old) or (lineno == nline):
                dispv_e_all.append(dispv_e_add)
                dispv_e_add = np.zeros((0,nv,3))
            dispv_line = np.zeros(0)
            for i in range(3,len(l)):
                dispv_line = np.append(dispv_line, float(l[i].strip()))
            dispv_e_add = np.append(dispv_e_add, dispv_line.reshape([1,nv,3]), axis=0)
            index_old = np.copy(index)
        lineno += 1



##################################################
# plot figures
##################################################
### initialize output directory
dirname = './result/mode'
if os.path.isdir(dirname):
    shutil.rmtree(dirname)
os.makedirs(dirname)

### specify step index for plotting figures
theta_bar = 2. * math.atan(math.sqrt(math.sqrt(2)-1))
iplotb = np.argmin(np.abs(theta_all - theta_bar))
iplots = np.array([iplotb-1, iplotb, iplotb+1])

### set plot area
xmax = np.max(vert_all[:,:,0])
xmin = np.min(vert_all[:,:,0])
ymax = np.max(vert_all[:,:,1])
ymin = np.min(vert_all[:,:,1])
zmax = np.max(vert_all[:,:,2])
zmin = np.min(vert_all[:,:,2])
span = max([xmax-xmin, ymax-ymin, zmax-zmin])
center = np.array([xmax+xmin, ymax+ymin, zmax+zmin])/2.
lims = np.zeros((3,2))
lims[:,0] = center - span/2. + 0.01
lims[:,1] = center + span/2. - 0.01

### plot figures
for iplot in iplots:
    ### plot shape
    plotshape(vert_all[iplot], face, dirname+'/shape%i'%(iplot), lims=lims, iso=True, pln=True, eps=True)

    ### plot displacement mode in mechanism analysis
    for imode in range(dispv_m_all[iplot].shape[0]-6):
        dispv_m_plot = dispv_m_all[iplot][imode]
        dmax = np.max(np.linalg.norm(dispv_m_plot, axis=1))
        plotdisp(vert_all[iplot], dispv_m_plot*0.01/dmax, face, dirname+'/mech%i_mode%i'%(iplot,imode+1),
                 lims=lims, iso=True, pln=True, eps=True)

    ### plot displacement mode in elestic analysis
    imodes = np.array([6,7,8,9,10,11])
    for imode in imodes:
        dispv_e_plot = dispv_e_all[iplot][imode]
        dmax = np.max(np.linalg.norm(dispv_e_plot, axis=1))
        plotdisp(vert_all[iplot], dispv_e_plot*0.01/dmax, face, dirname+'/elastic%i_mode%i'%(iplot,imode+1),
                 lims=lims, iso=True, pln=True, eps=True)

    ### output corresponding eigenvalues
    np.savetxt(dirname+'/eigen_mech%i.dat'%(iplot), eigen_m_all[iplot])
    np.savetxt(dirname+'/eigen_elastic%i.dat'%(iplot), eigen_e_all[iplot][imodes])

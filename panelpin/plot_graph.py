import numpy as np
import math
import os
import shutil
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


##################################################
# function for plotting graphs
##################################################
def plotgraph(data,            # data list to plot (list of [#data,2] array, float)
              xlabel,          # label of x axis (str)
              ylabel,          # label of y axis (str)
              xticks,          # ticks of x axis ([#ticks] array, float)
              yticks,          # ticks of y axis ([#ticks] array, float)
              xlim,            # range of x axis ([min, max] array, float)
              ylim,            # range of y axis ([min, max] array, float)
              fname,           # file name (str)
              figsize=[60,40], # figure size (list of [width,height], mm, float)
              cols=['k'],      # line colors (list, str)
              ptns=[[0.]],     # dash patterns (list of list, float)
              lwidth=0.3,      # line width (float)
              mtype=['o'],     # marker type (list, str)
              msize=1.2,       # marker size (float)
              mwidth=0.4,      # marker line width (float)
              fwidth=0.4,      # frame line width (float)
              gwidth=0.2,      # grid line width (float)
              gcolor='gray',   # grid line color (str)
              adjl=0.22,       # adjuster of left blank area (float)
              adjr=0.92,       # adjuster of right blank area (float)
              adjb=0.2,        # adjuster of bottom blank area (float)
              adjt=0.9,        # adjuster of top blank area (float)
              fontsize=10,     # font size for axis labels (float)
              valuesize=8,     # font size for tick values (float)
              logscale=False,  # plot in log-scale y-axis (bool)
              eps=True,        # save .eps file (bool)
              png=True,        # save .png file (bool)
              show = False     # show plot result (bool)
              ):
    fsizex = figsize[0] * 0.0393701
    fsizey = figsize[1] * 0.0393701
    plt.rcParams['font.family'] ='Times New Roman'
    plt.rcParams['font.size'] = fontsize
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['mathtext.fontset'] = 'stix'
    fig = plt.figure(figsize=(fsizex,fsizey))
    fig.subplots_adjust(left=adjl, right=adjr, bottom=adjb, top=adjt)
    ax = fig.add_subplot(111)
    ax.spines['top'].set_linewidth(fwidth)
    ax.spines['bottom'].set_linewidth(fwidth)
    ax.spines['left'].set_linewidth(fwidth)
    ax.spines['right'].set_linewidth(fwidth)
    ### setings for plotting
    ax.set_xlabel(xlabel, labelpad = 1.)
    ax.set_ylabel(ylabel, labelpad = 1.)
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.tick_params(length=0, labelsize=valuesize)
    ax.grid(axis="both", color=gcolor, linewidth=gwidth)
    ### rearrange color, marker, and dash pattern data
    ndata = len(data)
    ncol = len(cols)
    if ncol < ndata:
        for i in range(ndata-ncol):
            cols.append(cols[i%ncol])
    ndash = len(ptns)
    if ndash < ndata:
        for i in range(ndata-ndash):
            ptns.append(ptns[i%ndash])
    nmkr = len(mtype)
    if nmkr < ndata:
        for i in range(ndata-nmkr):
            mtype.append(mtype[i%nmkr])
    ### plot graph
    for i in range(ndata):
        x = data[i][:,0]
        y = data[i][:,1]
        if len(ptns[i]) < 2:
            plt.plot(x, y, linewidth=lwidth, color=cols[i], marker=mtype[i], markersize=msize,
                     fillstyle='none', markeredgewidth=mwidth, markeredgecolor=cols[i])
        else:
            plt.plot(x, y, linewidth=lwidth, color=cols[i], linestyle='dashed', dashes=ptns[i],
                     marker=mtype[i], markersize=msize, fillstyle='none', markeredgewidth=mwidth,
                     markeredgecolor=cols[i])
    if logscale:
        plt.yscale('log')
    ### output
    if png:
        plt.savefig(fname+'.png', dpi=600)
    if eps:
        plt.savefig(fname+'.eps')
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


### singularvalues of compatibility matrix in mechanism analysis
singular_all = []
lineno = 1
with open('./result/singular_mech.dat') as f:
    for line in f:
        if lineno > 1:
            singular_add = np.zeros(0)
            l = line.split(' ')
            for i in range(1,len(l)):
                singular_add = np.append(singular_add, float(l[i].strip()))
            singular_all.append(singular_add)
        lineno += 1


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


### potential energy in directions of eigenmodes of elastic analysis
energy_all = []
lineno = 0
index_old = 0
energy_add = np.zeros((0,2))
nline = np.loadtxt('./result/energy.dat',skiprows=1).shape[0]
with open('./result/energy.dat') as f:
    for line in f:
        if lineno > 0:
            l = line.split(' ')
            index = int(l[0].strip())
            if (index != index_old) or (lineno == nline):
                energy_all.append(energy_add)
                energy_add = np.zeros((0,2))
            energy_add = np.append(energy_add, np.array([[float(l[3].strip()), float(l[4].strip())]]), axis=0)
            index_old = np.copy(index)
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
# plot graphs
##################################################
### initialize output directory
dirname = './result/graph'
if os.path.isdir(dirname):
    shutil.rmtree(dirname)
os.makedirs(dirname)


### featured angles
theta_bar = 2. * math.atan(math.sqrt(math.sqrt(2)-1))
#itheta1 = np.argmin(np.abs(theta_all - math.pi/6))
#itheta2 = np.argmin(np.abs(theta_all - theta_bar))
#itheta3 = np.argmin(np.abs(theta_all - math.pi/2))
#ithetas = np.array([itheta1, itheta2, itheta3])
ithetas = np.array([np.argmin(np.abs(theta_all - theta_bar))])


### plot graph of singular value of compatibility matrix in mechanism analysis
# arrange data
nplot = 11
xs = np.rad2deg(theta_all)
ys = np.zeros((nplot,nstep))
for i in range(nstep):
    ys[:,i] = np.sort(singular_all[i])[0:nplot]
plotdata = []
for i in range(nplot):
    adddata = np.zeros((nstep,2))
    adddata[:,0] += xs
    adddata[:,1] += ys[i]*10**3
    plotdata.append(adddata)
# plot settings
ptns = []
cols = []
mtype = []
for i in range(6):
    ptns.append([2,3])
    cols.append('g')
    mtype.append('x')
if nplot > 6:
    ptns.append([2,3])
    cols.append('r')
    mtype.append('s')
for i in range(max([0,nplot-7])):
    ptns.append([2,3])
    cols.append('k')
    mtype.append('o')
xlabel = r"$\theta$ [deg.]"
ylabel = r"Singular value [$\times 10^{-3}$]"
xticks = np.array([0, 30, 60, 90, 120, 150, 180])
yticks = np.array([0, 1, 2, 3, 4])
xlim = np.array([np.min(xticks),np.max(xticks)])
ylim = np.array([-(np.max(yticks)-np.min(yticks))/20, np.max(yticks)])
# featured angles
for i in range(len(ithetas)):
    adddata = np.zeros((2,2))
    adddata[0,0] += xs[ithetas[i]]
    adddata[1,0] += xs[ithetas[i]]
    adddata[0,1] += ylim[0]
    adddata[1,1] += ylim[-1]
    plotdata.append(adddata)
    ptns.append([4,2])
    cols.append('k')
    mtype.append(' ')
# plot graph in overall view
plotgraph(plotdata, xlabel, ylabel, xticks, yticks, xlim, ylim,
          dirname+'/singular1', ptns=ptns, cols=cols, mtype=mtype, eps=True)

# rearrange data for detailed view
plotdata = []
for i in range(nplot):
    adddata = np.zeros((nstep,2))
    adddata[:,0] += xs
    adddata[:,1] += ys[i]
    plotdata.append(adddata)
# plot settings
ptns = []
cols = []
mtype = []
for i in range(6):
    ptns.append([2,3])
    cols.append('g')
    mtype.append('x')
if nplot > 6:
    ptns.append([2,3])
    cols.append('r')
    mtype.append('s')
for i in range(max([0,nplot-7])):
    ptns.append([2,3])
    cols.append('k')
    mtype.append('o')
xlabel = r"$\theta$ [deg.]"
ylabel = "Singular value \n (Log scale)"
xticks = np.array([0, 30, 60, 90, 120, 150, 180])
yticks = np.array([1.e-17, 1.e-15, 1.e-13, 1.e-11])
xlim = np.array([np.min(xticks)-6,np.max(xticks)+6])
ylim = np.array([np.min(yticks), np.max(yticks)])
# featured angles
for i in range(len(ithetas)):
    adddata = np.zeros((2,2))
    adddata[0,0] += xs[ithetas[i]]
    adddata[1,0] += xs[ithetas[i]]
    adddata[0,1] += ylim[0]
    adddata[1,1] += ylim[-1]
    plotdata.append(adddata)
    ptns.append([4,2])
    cols.append('k')
    mtype.append(' ')
# plot graph in detailed view
plotgraph(plotdata, xlabel, ylabel, xticks, yticks, xlim, ylim,
          dirname+'/singular2', figsize=[63,40], adjl=0.25, ptns=ptns, cols=cols, mtype=mtype, logscale=True, eps=True)


### plot graph of eigenvalue of stiffness matrix in elastic analysis
# arrange data
nplot = 11
xs = np.rad2deg(theta_all)
ys = np.zeros((nplot,nstep))
for i in range(nstep):
    ys[:,i] = np.sort(eigen_e_all[i])[0:nplot]
plotdata = []
for i in range(nplot):
    adddata = np.zeros((nstep,2))
    adddata[:,0] += xs
    adddata[:,1] += ys[i]
    plotdata.append(adddata)
# plot settings
ptns = []
cols = []
mtype = []
for i in range(6):
    ptns.append([2,3])
    cols.append('g')
    mtype.append('x')
if nplot > 6:
    ptns.append([2,3])
    cols.append('r')
    mtype.append('s')
for i in range(max([0,nplot-7])):
    ptns.append([2,3])
    cols.append('k')
    mtype.append('o')
xlabel = r"$\theta$ [deg.]"
ylabel = "Eigenvalues"
xticks = np.array([0, 30, 60, 90, 120, 150, 180])
yticks = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
xlim = np.array([np.min(xticks),np.max(xticks)])
ylim = np.array([-(np.max(yticks)-np.min(yticks))/20, np.max(yticks)])
# featured angles
for i in range(len(ithetas)):
    adddata = np.zeros((2,2))
    adddata[0,0] += xs[ithetas[i]]
    adddata[1,0] += xs[ithetas[i]]
    adddata[0,1] += ylim[0]
    adddata[1,1] += ylim[-1]
    plotdata.append(adddata)
    ptns.append([4,2])
    cols.append('k')
    mtype.append(' ')
# plot graph in overall view
plotgraph(plotdata, xlabel, ylabel, xticks, yticks, xlim, ylim, dirname+'/eigen_elastic1',
          ptns=ptns, cols=cols, mtype=mtype, eps=True)

# arrange data for detailed view
nplot = 11
xs = np.rad2deg(theta_all)
ys = np.zeros((nplot,nstep))
for i in range(nstep):
    ys[:,i] = np.sort(eigen_e_all[i])[0:nplot]
plotdata = []
for i in range(nplot):
    adddata = np.zeros((nstep,2))
    adddata[:,0] += xs
    adddata[:,1] += np.abs(ys[i])
    plotdata.append(adddata)
# plot settings
ptns = []
cols = []
mtype = []
for i in range(6):
    ptns.append([2,3])
    cols.append('g')
    mtype.append('x')
if nplot > 6:
    ptns.append([2,3])
    cols.append('r')
    mtype.append('s')
for i in range(max([0,nplot-7])):
    ptns.append([2,3])
    cols.append('k')
    mtype.append('o')
xlabel = r"$\theta$ [deg.]"
ylabel = "Abs. Eigenvalues \n (Log scale)"
xticks = np.array([0, 30, 60, 90, 120, 150, 180])
yticks = np.array([1.e-16, 1.e-12, 1.e-8, 1.e-4])
xlim = np.array([np.min(xticks), np.max(xticks)])
ylim = np.array([np.min(yticks), np.max(yticks)])
# featured angles
for i in range(len(ithetas)):
    adddata = np.zeros((2,2))
    adddata[0,0] += xs[ithetas[i]]
    adddata[1,0] += xs[ithetas[i]]
    adddata[0,1] += ylim[0]
    adddata[1,1] += ylim[-1]
    plotdata.append(adddata)
    ptns.append([4,2])
    cols.append('k')
    mtype.append(' ')
# plot graph in detailed view
plotgraph(plotdata, xlabel, ylabel, xticks, yticks, xlim, ylim, dirname+'/eigen_elastic2',
          figsize=[63,40], adjl=0.25, ptns=ptns, cols=cols, mtype=mtype, logscale=True, eps=True)


### plot graph of potential energy in directions of trivial eigenmodes of elastic analysis
xs = np.rad2deg(theta_all)
for iplot in range(6):
    # arrange data and line type
    ys = np.zeros((2,nstep))
    for i in range(nstep):
        ys[:,i] = energy_all[i][iplot]
    plotdata = []
    for i in range(2):
        adddata = np.zeros((nstep,2))
        adddata[:,0] += xs
        adddata[:,1] += ys[i]*10**11
        plotdata.append(adddata)
        ptns = [[2,3],[2,3]]
        cols = ['r','b']
        mtype = ['o','x']
    # plot settings
    xlabel = r"$\theta$ [deg.]"
    ylabel = r"Energy [$\times 10^{-11}$]"
    xticks = np.array([0, 30, 60, 90, 120, 150, 180])
    yticks = np.array([-1, -0.5, 0, 0.5, 1])
    xlim = np.array([np.min(xticks),np.max(xticks)])
    ylim = np.array([np.min(yticks),np.max(yticks)])
    # featured angles
    for i in range(len(ithetas)):
        adddata = np.zeros((2,2))
        adddata[0,0] += xs[ithetas[i]]
        adddata[1,0] += xs[ithetas[i]]
        adddata[0,1] += ylim[0]
        adddata[1,1] += ylim[-1]
        plotdata.append(adddata)
        ptns.append([4,2])
        cols.append('k')
        mtype.append(' ')
    # plot graph
    plotgraph(plotdata, xlabel, ylabel, xticks, yticks, xlim, ylim,
              dirname+'/energy_mode%i'%(iplot+1), ptns=ptns, cols=cols, mtype=mtype, eps=True)
    # output values
    with open(dirname+'/energy_mode%i.dat'%(iplot+1), 'w') as f:
        f.write('index theta hinge panel\n')
        for i in range(nstep):
            f.write('%i %.8e %.8e %.8e\n'%(i,xs[i],ys[0,i],ys[1,i]))


### plot graph of potential energy in directions of non-trivial eigenmodes of elastic analysis
# arrange data of mode 7
imode = 6
ys = np.zeros((2,nstep))
for i in range(nstep):
    ys[:,i] = energy_all[i][imode]
plotdata = []
for i in range(2):
    adddata = np.zeros((nstep,2))
    adddata[:,0] += xs
    adddata[:,1] += ys[i]*10**6
    plotdata.append(adddata)
# plot settings
ptns = [[2,3],[2,3]]
cols = ['r','b']
mtype = ['o','x']
xlabel = r"$\theta$ [deg.]"
ylabel = r"Energy [$\times 10^{-6}$]"
xticks = np.array([0, 30, 60, 90, 120, 150, 180])
yticks = np.array([0, 0.25, 0.5, 0.75, 1.0])
xlim = np.array([np.min(xticks),np.max(xticks)])
ylim = np.array([-(np.max(yticks)-np.min(yticks))/20, np.max(yticks)])
# featured angles
for i in range(len(ithetas)):
    adddata = np.zeros((2,2))
    adddata[0,0] += xs[ithetas[i]]
    adddata[1,0] += xs[ithetas[i]]
    adddata[0,1] += ylim[0]
    adddata[1,1] += ylim[-1]
    plotdata.append(adddata)
    ptns.append([4,2])
    cols.append('k')
    mtype.append(' ')
# plot graph
plotgraph(plotdata, xlabel, ylabel, xticks, yticks, xlim, ylim,
          dirname+'/energy_mode%i'%(imode+1), ptns=ptns, cols=cols, mtype=mtype, eps=True)
# output values
with open(dirname+'/energy_mode%i.dat'%(imode+1), 'w') as f:
    f.write('index theta hinge panel\n')
    for i in range(nstep):
        f.write('%i %.8e %.8e %.8e\n'%(i,xs[i],ys[0,i],ys[1,i]))

# arrange data of mode 8
imode = 7
ys = np.zeros((2,nstep))
for i in range(nstep):
    ys[:,i] = energy_all[i][imode]
plotdata = []
for i in range(2):
    adddata = np.zeros((nstep,2))
    adddata[:,0] += xs
    adddata[:,1] += ys[i]*10**2
    plotdata.append(adddata)
# plot settings
ptns = [[2,3],[2,3]]
cols = ['r','b']
mtype = ['o','x']
xlabel = r"$\theta$ [deg.]"
ylabel = r"Energy [$\times 10^{-2}$]"
xticks = np.array([0, 30, 60, 90, 120, 150, 180])
yticks = np.array([0, 1, 2, 3, 4, 5, 6])
xlim = np.array([np.min(xticks),np.max(xticks)])
ylim = np.array([-(np.max(yticks)-np.min(yticks))/20, np.max(yticks)])
# featured angles
for i in range(len(ithetas)):
    adddata = np.zeros((2,2))
    adddata[0,0] += xs[ithetas[i]]
    adddata[1,0] += xs[ithetas[i]]
    adddata[0,1] += ylim[0]
    adddata[1,1] += ylim[-1]
    plotdata.append(adddata)
    ptns.append([4,2])
    cols.append('k')
    mtype.append(' ')
# plot graph
plotgraph(plotdata, xlabel, ylabel, xticks, yticks, xlim, ylim,
          dirname+'/energy_mode%i'%(imode+1), ptns=ptns, cols=cols, mtype=mtype, eps=True)
# output values
with open(dirname+'/energy_mode%i.dat'%(imode+1), 'w') as f:
    f.write('index theta hinge panel\n')
    for i in range(nstep):
        f.write('%i %.8e %.8e %.8e\n'%(i,xs[i],ys[0,i],ys[1,i]))

# arrange data of mode 9
imode = 8
ys = np.zeros((2,nstep))
for i in range(nstep):
    ys[:,i] = energy_all[i][imode]
plotdata = []
for i in range(2):
    adddata = np.zeros((nstep,2))
    adddata[:,0] += xs
    adddata[:,1] += ys[i]*10
    plotdata.append(adddata)
# plot settings
ptns = [[2,3],[2,3]]
cols = ['r','b']
mtype = ['o','x']
xlabel = r"$\theta$ [deg.]"
ylabel = r"Energy [$\times 10^{-1}$]"
xticks = np.array([0, 30, 60, 90, 120, 150, 180])
yticks = np.array([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
xlim = np.array([np.min(xticks),np.max(xticks)])
ylim = np.array([-(np.max(yticks)-np.min(yticks))/20, np.max(yticks)])
# featured angles
for i in range(len(ithetas)):
    adddata = np.zeros((2,2))
    adddata[0,0] += xs[ithetas[i]]
    adddata[1,0] += xs[ithetas[i]]
    adddata[0,1] += ylim[0]
    adddata[1,1] += ylim[-1]
    plotdata.append(adddata)
    ptns.append([4,2])
    cols.append('k')
    mtype.append(' ')
# plot graph
plotgraph(plotdata, xlabel, ylabel, xticks, yticks, xlim, ylim,
          dirname+'/energy_mode%i'%(imode+1), ptns=ptns, cols=cols, mtype=mtype, eps=True)
# output values
with open(dirname+'/energy_mode%i.dat'%(imode+1), 'w') as f:
    f.write('index theta hinge panel\n')
    for i in range(nstep):
        f.write('%i %.8e %.8e %.8e\n'%(i,xs[i],ys[0,i],ys[1,i]))

# arrange data of mode 10
imode = 9
ys = np.zeros((2,nstep))
for i in range(nstep):
    ys[:,i] = energy_all[i][imode]
plotdata = []
for i in range(2):
    adddata = np.zeros((nstep,2))
    adddata[:,0] += xs
    adddata[:,1] += ys[i]*10
    plotdata.append(adddata)
# plot settings
ptns = [[2,3],[2,3]]
cols = ['r','b']
mtype = ['o','x']
xlabel = r"$\theta$ [deg.]"
ylabel = r"Energy [$\times 10^{-1}$]"
xticks = np.array([0, 30, 60, 90, 120, 150, 180])
yticks = np.array([0, 1, 2, 3, 4, 5])
xlim = np.array([np.min(xticks),np.max(xticks)])
ylim = np.array([-(np.max(yticks)-np.min(yticks))/20, np.max(yticks)])
# featured angles
for i in range(len(ithetas)):
    adddata = np.zeros((2,2))
    adddata[0,0] += xs[ithetas[i]]
    adddata[1,0] += xs[ithetas[i]]
    adddata[0,1] += ylim[0]
    adddata[1,1] += ylim[-1]
    plotdata.append(adddata)
    ptns.append([4,2])
    cols.append('k')
    mtype.append(' ')
# plot graph
plotgraph(plotdata, xlabel, ylabel, xticks, yticks, xlim, ylim,
          dirname+'/energy_mode%i'%(imode+1), ptns=ptns, cols=cols, mtype=mtype, eps=True)
# output values
with open(dirname+'/energy_mode%i.dat'%(imode+1), 'w') as f:
    f.write('index theta hinge panel\n')
    for i in range(nstep):
        f.write('%i %.8e %.8e %.8e\n'%(i,xs[i],ys[0,i],ys[1,i]))

# arrange data of mode 11
imode = 10
ys = np.zeros((2,nstep))
for i in range(nstep):
    ys[:,i] = energy_all[i][imode]
plotdata = []
for i in range(2):
    adddata = np.zeros((nstep,2))
    adddata[:,0] += xs
    adddata[:,1] += ys[i]*10
    plotdata.append(adddata)
# plot settings
ptns = [[2,3],[2,3]]
cols = ['r','b']
mtype = ['o','x']
xlabel = r"$\theta$ [deg.]"
ylabel = r"Energy [$\times 10^{-1}$]"
xticks = np.array([0, 30, 60, 90, 120, 150, 180])
yticks = np.array([0, 1, 2, 3, 4, 5])
xlim = np.array([np.min(xticks),np.max(xticks)])
ylim = np.array([-(np.max(yticks)-np.min(yticks))/20, np.max(yticks)])
# featured angles
for i in range(len(ithetas)):
    adddata = np.zeros((2,2))
    adddata[0,0] += xs[ithetas[i]]
    adddata[1,0] += xs[ithetas[i]]
    adddata[0,1] += ylim[0]
    adddata[1,1] += ylim[-1]
    plotdata.append(adddata)
    ptns.append([4,2])
    cols.append('k')
    mtype.append(' ')
# plot graph
plotgraph(plotdata, xlabel, ylabel, xticks, yticks, xlim, ylim,
          dirname+'/energy_mode%i'%(imode+1), ptns=ptns, cols=cols, mtype=mtype, eps=True)
# output values
with open(dirname+'/energy_mode%i.dat'%(imode+1), 'w') as f:
    f.write('index theta hinge panel\n')
    for i in range(nstep):
        f.write('%i %.8e %.8e %.8e\n'%(i,xs[i],ys[0,i],ys[1,i]))

# arrange data of multiple mode plot
imodes = np.array([6,7,8,9,10])
ys = np.zeros((2,nstep,len(imodes)))
for i in range(nstep):
    for j in range(len(imodes)):
        ys[:,i,j] = energy_all[i][imodes[j]]
plotdata = []
# plot settings
ptns = []
cols = []
mtype = []
for i in range(len(imodes)):
    for j in range(2):
        adddata = np.zeros((nstep,2))
        adddata[:,0] += xs
        adddata[:,1] += ys[j,:,i]*10
        plotdata.append(adddata)
    ptns = ptns + [[2,3],[2,3]]
    cols = cols + ['r', 'b']
    mtype = mtype + ['o','x']
xlabel = r"$\theta$ [deg.]"
ylabel = r"Energy [$\times 10^{-1}$]"
xticks = np.array([0, 30, 60, 90, 120, 150, 180])
yticks = np.array([0, 1, 2, 3, 4, 5])
xlim = np.array([np.min(xticks),np.max(xticks)])
ylim = np.array([-(np.max(yticks)-np.min(yticks))/20, np.max(yticks)])
# featured angles
for i in range(len(ithetas)):
    adddata = np.zeros((2,2))
    adddata[0,0] += xs[ithetas[i]]
    adddata[1,0] += xs[ithetas[i]]
    adddata[0,1] += ylim[0]
    adddata[1,1] += ylim[-1]
    plotdata.append(adddata)
    ptns.append([4,2])
    cols.append('k')
    mtype.append(' ')
# plot graph
plotgraph(plotdata, xlabel, ylabel, xticks, yticks, xlim, ylim,
          dirname+'/energy_mode%i_%i'%(imodes[0]+1,imodes[-1]+1), ptns=ptns, cols=cols, mtype=mtype, eps=True)

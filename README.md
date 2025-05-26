# Rigid_and_Flat_Foldable_Grid_Origami_Structure

Data corresponding to the paper "Rigid- and flat-foldable grid origami structure exhibiting bifurcation of mechanism in non-flat state" submitted to the International Journal of Solids and Structures.
In this directory you will find two folders, `panelpin` and `abaqus`.

* `panelpin` contains program files for infinitesimal mechanism and eigenvalue analysis of the panel-pin model.
* `abaqus` contains Abaqus input files of natural frequency and geometrically nonlinear large deformation analysis of the finite element model.

## Folder Description
### panelpin
This folder contains the following files and folders. The programs are written using Python 3.9.7 with the libraries Numpy 1.20.3 and Matplotlib 3.4.3.

* `analysis.py` is the main program to implement the whole workflow of the infinitesimal mechanism and eigenvalue analysis of the panel-pin model. You can define the value of parameters determining the shape of the structure in lines 14 - 25 as follows:

```python
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
```
* `plot_graph.py` is the program to draw graphs of the singular values of the compatibility matrix, the eigenvalues of the tangent stiffness matrix, and so on. You need to manually set the range of axes to be displayed on the graphs.

* `plot_mode.py` is the program to draw figures of the mechanism modes and eigenmodes of the panel-pin model.

* `result` will contain the result files, graphs, and figures of the modes after the programs are executed.


### abaqus
This folder contains the folders containing input files for analysis of $3\times3$ structure by using Abaqus 2020 in each loading case.

* `load1_neg` contains the input files for loading condition 1 with the negative loading factor.
* `load1_pos` contains the input files for loading condition 1 with the positive loading factor.
* `load2_neg` contains the input files for loading condition 2 with the negative loading factor.
* `load2_pos` contains the input files for loading condition 2 with the positive loading factor.
* `load3_neg` contains the input files for loading condition 3 with the negative loading factor.
* `load3_pos` contains the input files for loading condition 3 with the positive loading factor.

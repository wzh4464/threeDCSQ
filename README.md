# CellShapeAnalysis


the code is built in hierarchical structure, one sub-dict is a package, the higher packages can use functions in lower packages.

## Documents

### Lineage Tree Draw
* use ./lineage_stat/draw_test.py -> 
  * cell fate label tree -> draw_cell_fate_lineage_tree_01paper()

## Project directory file tree

Brief introduction on code directory.

```html
root/: work directory environment
  |--DATA/: the transformation digital results in csv files or plotted figures, too large to upload on github
  |--analysis/: 3rd layer, the analysis code including spherical harmonic transformation, contact area, clustering and PCA
     |--SH_analyses.py: functions compare sherical harmonics reconstruction and original shapes.
  |--data_scripts/: 1st layer, useless temporally.
  |--experiment/: 4th layer, useless temporally.
  |--lineage_stat/: 2nd layer, combine embryos, generate cell treelib files and cell lineage tree plot
     |--data_structure.py: build the cell lineage tree by CD files. (exist or lost in every frame)
     |--generate_life_span.py: generate the CD file
     |--lineage_tree.py: plot lineage tree with cells' values
     |--draw_test.py: combine embryos and draw lineage tree entry.
  |--transformation/: 2nd layer, transform or extract features form 3D cells
     |--R_matrix_representation.py: sample on 3D surface
     |--SH_representation.py: transform spherical matrix to spherical harmonics matrix
     |--PCA.py: some scripts draw PCA shapes.
  |--utils/: 1st layer, all basic functions includes sample, transformation, contact detection
     |--cell_func.py: get pairs of cell labels and names, calculate surface area, calculate volume, find contact or not
     |--draw_func.py: some functions useful in drawing missions.
     |--general_func.py: Cartesian coordinate system and spherical coordinate system conversion
     |--sh_cooperation.py: SPHARM array to vector, vector to array
     |--shape_model.py: research on cell-cell contact
     |--shape_preprocess.py: cell CONTACT INFORMATION including points coordinates, cell surface extraction, erosion or dialation operation
     |--spherical_func.py: sample for spherical harmonics transformation
  |--
```
## Files and Function usages

## Scientific Concepts

### Curvature Of cell

#### Gaussian curvature using libigl

https://libigl.github.io/libigl-python-bindings/tut-chapter1/


**Principal curvatures**： the biggest and smallest radius of the points. 

https://zh.wikipedia.org/wiki/%E4%B8%BB%E6%9B%B2%E7%8E%87
https://zh.wikipedia.org/wiki/%E9%AB%98%E6%96%AF%E6%9B%B2%E7%8E%87

## Code Implementation

### Lineage tree draw
* generate the tree files with command first go to lineage_stat folder
```bash
    $ python generate_life_span.py  
```
1. this file would use function **construct_basic_tree** which would build a tree base on CD file with x position by cells' generation which time list is empty.
2. code in file **generate_life_span.py** would add cells' frames to the trees base on CD files.
3.  the CShaper embryo info, so i use embryo 06 to build basic tree in funcion **draw_PCA_combined**. 

* draw the average tree
    1. first, function **draw_PCA_combined** would construct a average lineage tree for all embryos.
    2. second, the function **get_combined_lineage_tree** would go through this lineage tree and get the frame cells' values depend on time/frame resolutions.
    3. third, at the drawing step, we would calculate all average value first and give cells values at different tp.
    4. fourth, we go through the tree again and do interpolation for the lost cells. 

* the legend frontzise is set at function **draw_life_span_tree**. 

## Python Environment 
Please update conda by running

    $ conda update -n base -c defaults conda


*  My environment setting
```
# environment location: C:\Users\zelinli6\miniconda3\envs\CellShapeAnalysis

$ conda activate CellShapeAnalysis
```

## TODO List

- [x] FIGURE01 workflow figure 
    - [x] replace array with matrix in figure: 2D Spherical matrix, SPHARM coefficient matrix

- [x] FIGURE02 the figure help reader understand 2D Spherical matrix.

- [x] FIGURE03 confirm point distance unit, use correct distance, volume and surface in 2DMAP feature array.
  - [x] calculate 2DMap PCA.
  - [ ] draw 2DMap PCA and SPAHRM PCA in histogram.
  - [x] combine five feature schematic diagram.

- [x] FIGURE04 average lineage tree.

- [x] FIGURE05 shape reproducibility. (linear relation)

- [x] FIGURE06 skin cell recognize.

- [x] FIGURE07 cluster result (internal error and external error)

### opend3d: points surface display 

o3d configuration:
{
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : false,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 58.949796990685456, 73.062096966802002, 17.53928827322666 ],
			"boundingbox_min" : [ -58.050203009314544, -64.937903033197998, -32.46071172677334 ],
			"field_of_view" : 60.0,
			"front" : [ -0.022170625825107981, 0.031107593735803508, -0.9992701241218469 ],
			"lookat" : [ 0.44979699068545642, 4.0620969668020024, -7.46071172677334 ],
			"up" : [ 0.039903629142300223, 0.99874686470810126, 0.030205969890268435 ],
			"zoom" : 0.69999999999999996
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}


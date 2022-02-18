# CellShapeAnalysis

## TODO List

### 01paper
- [x] FIGURE01 workflow figure 
    - [x] replace array with matrix in figure: 2D Spherical matrix, SPHARM coefficient matrix

- [x] FIGURE02 the figure help reader understand 2D Spherical matrix.

- [x] FIGURE03 confirm point distance unit, use correct distance, volume and surface in 2DMAP feature array.
  - [x] calculate 2DMap PCA.
  - [] draw 2DMap PCA and SPAHRM PCA in histogram.
  - [ ] combine five feature schematic diagram.

- [ ] FIGURE04 average lineage tree.

- [x] FIGURE05 shape reproducibility. (linear relation)

- [ ] FIGURE06 skin cell recognize.

- [ ] NOT NOW - FIGURE07 cluster result (internal error and external error)

### points surface display

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

## Files and Function usages

### Curvature Of cell

#### Gaussian curvature using libigl

https://libigl.github.io/libigl-python-bindings/tut-chapter1/


**Principal curvatures**： the biggest and smallest radius of the points. 

https://zh.wikipedia.org/wiki/%E4%B8%BB%E6%9B%B2%E7%8E%87
https://zh.wikipedia.org/wiki/%E9%AB%98%E6%96%AF%E6%9B%B2%E7%8E%87


### Lineage tree draw
* generate the tree files with command first go to lineage_stat folder
```bash
    $ python generate_life_span.py  
```
1. this file would use function **construct_basic_tree** which would build a tree base on CD file with x position by cells' generation which time list is empty.
2. code in file **generate_life_span.py** would add cells' frames to the trees base on CD files.
3.  the CShaper embryo info, so i use embryo 06 to build basic tree in funcion **draw_PCA_combined**. 
4. the parameter max_time in function **get_combined_lineage_tree** is very import.
```
04   begin cell ABa  cell number 819 embryo max frame 150
05   begin cell ABa  cell number 1081 embryo max frame 170
06   begin cell ABa  cell number 1223 embryo max frame 210
07   begin cell AB  cell number 807 embryo max frame 165
08   begin cell ABa  cell number 897 embryo max frame 160
09   begin cell ABa  cell number 825 embryo max frame 160
10   begin cell ABa  cell number 775 embryo max frame 160
11   begin cell ABa  cell number 799 embryo max frame 170
12   begin cell ABa  cell number 811 embryo max frame 165
13   begin cell ABa  cell number 769 embryo max frame 150
14   begin cell ABa  cell number 775 embryo max frame 155
15   begin cell AB  cell number 765 embryo max frame 170
16   begin cell ABa  cell number 763 embryo max frame 160
17   begin cell ABa  cell number 765 embryo max frame 160
18   begin cell ABa  cell number 775 embryo max frame 160
19   begin cell ABa  cell number 775 embryo max frame 160
20   begin cell ABa  cell number 791 embryo max frame 170
```

* draw the average tree
    1. first, function **draw_PCA_combined** would construct a average lineage tree for all embryos.
    2. second, the function **get_combined_lineage_tree** would go through this lineage tree and get the frame cells' values depend on time/frame resolutions.
    3. third, at the drawing step, we would calculate all average value first and give cells values at different tp.
    4. fourth, we go through the tree again and do interpolation for the lost cells. 

* the legend frontzise is set at function **draw_life_span_tree**. 
## environment 
Please update conda by running

    $ conda update -n base -c defaults conda


*  My environment setting
```
# environment location: C:\Users\zelinli6\miniconda3\envs\CellShapeAnalysis

$ conda activate CellShapeAnalysis
```
    


# CellShapeAnalysis

## Files and Function usages

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
    


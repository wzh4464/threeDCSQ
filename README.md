# CellShapeAnalysis

the code is built in hierarchical structure, one sub-dict is a package, the higher packages can use functions in lower packages.

## project dictionary file tree
```html
root/: work dictionary environment
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
     |--shape_preprocess.py: cell surface extraction, erosion or dialation operation
     |--spherical_func.py: sample for spherical harmonics transformation
  |--
```
## Files and Function usages

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
    3.  


## environment 
Please update conda by running

    $ conda update -n base -c defaults conda


*  My environment setting

    * environment location: C:\Users\zelinli6\miniconda3\envs\CellShapeAnalysis

    * conda activate CellShapeAnalysis


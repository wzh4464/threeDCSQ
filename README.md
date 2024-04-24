# 3DCSQ: An effective method for quantification, visualization and analysis of 3D cell shape during early embryogenesis 

Zelin Li\*, Jianfeng Cao, Guoye Guan\*, Chao Tang, Zhongying Zhao, and Hong Yan

\* Corresponding author. 

## Introduction

This is the implementation of the above paper. We develop 3 shape features to quantify cell morphology. We also conduct multiple experiments and evaluations on real living biological worm's embryos, *C. elegans*. 

## Function Usages Introduction
### test1.py/calculate_SPHARM_embryo_for_cells: calculate the spherical harmonics (transform) coefficient, *eigenharmonic*, for 1 embryo (3D+T) data.
### test1.py/SPHARM_eigenharmonic: calculate every cell's *Eigenharmonic Weight Vector*

### test1.py/Map2D_grid_csv: calculate the spherical grid coefficient, *eigengrid*, for 1 embryo (3D+T) data.
### test1.py/Map_2D_eigengrid: calculate every cell's *Eigengrid Weight Vector*

### transformation/test1.py/do_sampling_with_interval: do sampling on the cell objects one by one for spherical grid.


## Documents
The code is built in hierarchical structure, one sub-dict is a package, the higher packages can use functions in lower packages.

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
## Abstract
Embryogenesis, inherently three-dimensional, poses significant challenges in quantification when approached through 3D fluorescence imaging. Traditional descriptors such as volume, surface, and mean curvature often fall short, providing only a global view and lacking in local detail and reconstruction capability. Addressing this, we introduce an effective integrated method, 3D Cell Shape Quantification (3DCSQ), for transforming digitized 3D cell shapes into analytical feature vectors. This method uniquely combines spherical grids, spherical harmonics, and principal component analysis for a comprehensive approach to cell shape quantification, analysis, and visualization. We demonstrate 3DCSQ's effectiveness in recognizing cellular morphological phenotypes and clustering cells, utilizing feature vectors that are rigorously tested. Applied to Caenorhabditis elegans embryos, from 4- to 350-cell stages, 3DCSQ reliably identifies and quantifies biologically reproducible cellular patterns, including distinct skin cell deformations. By integrating cellular surface extraction, feature vector development, and cell shape clustering, 3DCSQ offers a robust platform for exploring cell shape's relationship with cell fate, enhancing our understanding of embryogenesis. This method not only systematizes cell shape description and evaluation but also monitors cell differentiation through shape changes, presenting a significant advancement in biological imaging and analysis.

Keywords: spherical harmonics (SPHARM), cell shape quantification, morphological reproducibility, lineage analysis, Caenorhabditis elegans (C. elegans)

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



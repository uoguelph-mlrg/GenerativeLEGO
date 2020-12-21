# Building LEGO Using Deep Generative Models of Graphs
This repo contains code for the paper [Building LEGO Using Deep Generative Models of Graph](https://ml4eng.github.io/camera_readys/55.pdf) by Rylee Thompson, [Elahe Ghalebi](https://scholar.google.com/citations?user=h5ZwVzcAAAAJ&hl=en), [Terrance DeVries](https://scholar.google.ca/citations?user=VFPOOsoAAAAJ&hl=en), and [Graham W. Taylor](https://www.gwtaylor.ca/). 

## Setup
To install the required packages run the command `pip install -r requirements.txt`. To download the dataset used in our paper from [Combinatorial 3D Shape Generation via Sequential Assembly](https://arxiv.org/abs/2004.07414), run the command `python extract_dataset.py`. This will download the dataset from [their Github](https://github.com/POSTECH-CVLab/Combinatorial-3D-Shape-Generation) and convert it to the form we use.

## Results
### Permutation results
To recreate the results from our permutation analysis, run the command `python permutation_script.py`.
### Generative model
To retrain our generative model, run `python DGMG_train.py`. You can see all the tweakable hyperparameters in setup_train.py.


## Beyond DGMG
The file `examples.ipynb` contains examples on how to use the code we developed for generating and validating LEGO graphs. It should be relatively straightforward to use these examples to try the dataset with another generative graph model. This code was made to be fairly general and extensible to other datasets, but will require some tweaking to the source code to get it up and running.

## Citations
If you use this code, please cite
```
@article{thompson2020LEGO,
  title={Building LEGO Using Deep Generative Models of Graphs},
  author={Thompson, Rylee and Elahe, Ghalebi and DeVries, Terrance  and Taylor, Graham W},
  journal={Machine Learning for Engineering Modeling, Simulation, and Design  
Workshop at Neural Information Processing Systems},
  year={2020}
}
```


## References
[1] Jungtaek Kim et al. “Combinatorial 3D Shape Generation via Sequential Assembly”. In: (Apr.2020). arXiv:2004.07414 [cs.CV].

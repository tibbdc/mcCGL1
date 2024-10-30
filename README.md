# mcCGL1
Construction and Analysis of a Multi-Constrained Model of Corynebacterium glutamicum

## Installation

1. create ETGEMs environment using conda:
```shell
$ conda create -n ETGEMs python=3.8   
$ conda activate ETGEMs 
```

2. install related packages using pip:
```shell 
$ pip install cobra  
$ pip install pubchempy  
$ pip install Bio
$ pip install bioservices
$ pip install pyprobar   
$ pip install plotly   
$ pip install seaborn
$ pip install pyomo==6.5.0
$ pip install equilibrator_api
$ pip install xmltodict
$ pip install jupyterlab   
$ python -m ipykernel install --user --name ETGEMs --display-name "ETGEMs"   
```

3. install solver:
```shell 
$ cd ./CPLEX221/python  
$ python setup.py install
```

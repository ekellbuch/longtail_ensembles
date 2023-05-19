# Installation instructions:
Instructions using conda

Check name of existing conda environments (to not overwrite them): 
```
conda list
```

Then create a new environment ``ensembles``as follows: 

```
conda create -n ensembles python=3.8
```

Now move into the root directory of this repo:
```
cd /path/to/this/repo
```

Activate your new environment, install dependencies and python package: 
```
conda activate env_name
conda install pip 
pip install -r docs/requirements.txt
pip install -e ./src
```
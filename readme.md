# Neural Model Simulation 

## Get started
1. Clone this repo in your local machine.
```bash
git clone https://github.com/pooyaalavian/neural-modeling-ode-solver.git
```
2. Open the folder in VS Code.
3. Open VS Code terminal in this folder.
4. Create a new python virtual env for this project. It must by python 3, recommended python 3.10 or higher (check with `python --version`).
```bash
python -m venv venv
               ~~~~
``` 
This will create a folder inside this folder named `venv` (the first `venv` is the name of the package, the second `venv`, with `~~~~` is the name of the environment you create).

5. Activate your python environment.
  - In VS Code, make sure `python` extension is installed. 
  Then `ctrl+shift+P`, type `python interpreter`. Select it and then select the environment you just created.
  - To activate the environment in your terminal, type:
  ```
  # in windows:
  .\venv\Scripts\activate
  # in mac/linux
  source ./venv/bin/activate
  ```

6. Install the required packages.
```
pip install -r requirements.txt
```
If you notice any issues, please see the `requirements.txt` file and install the problematic packages manually.

7. Start by running `start.ipynb` notebook.

## Input parameters
Input parameters are given to model in a JSON file.
See `defaults.json` for an example. 

The `defaults.json` file can be used directly or can be used as a base for modified parametersets.

Files in `src/` are utilities to read JSON files and parse the parameters in the correct way. 

You can create your own `params.json` file by cloning `defaults.json` and changing the parameters, or by loading `defaults.json` and applying your changes inside your python code.

To load a file named `my-params.json`, use this code snippet:
```py
from src.param import ParameterSet
p = ParameterSet('my-params.json')
```

To save your parameters after modification into a file, use this:
```py
# first change a parameter
p.exc1.tau=99
# then save the modified parameterset
p.save('new-param.json')
```

If you want to save only the changes compared to default, use:
```py
p.saveDelta('new-param.json')
```

TODO:
1. get new maps from abhirup and try to find the oscillations again
2. run pac experiments on the new ones

# scale down to one E and one I
# turn off noise
# play with AMPA projections not NMDA
# until you get stable focus
# get the power vs input plot in this regime
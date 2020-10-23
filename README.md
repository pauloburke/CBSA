# Constraint-Based Simulation Algorithm (CBSA)

CBSA is a Python package for simulation of biochemical systems using the Constraint-Based Simulation Algorithm.


[![PyPI](https://img.shields.io/pypi/v/cbsa.svg?color=b44e48)](https://pypi.org/project/cbsa)
![PyPI - License](https://img.shields.io/pypi/l/cbsa.svg?color=lightgray)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/cbsa.svg?color=lightgreen)

Table of contents
-----------------

* [Installation](#installation)
* [Usage](#usage)
* [License](#license)


Installation
------------

CBSA can be installed on your computer using pip.

Just run the following command:
```
python3 -m pip install cbsa --user --upgrade
```

Usage
-----

CBSA 

### Simple reaction example

Consider the following reaction system:

![equation](https://latex.codecogs.com/gif.latex?A%20%5Cleftrightarrow%20B%20%5Crightarrow%20C)

Using the Constrain-Based Modeling, the Stoichiometric matrix becomes:

![equation](https://latex.codecogs.com/gif.latex?S%20%3D%20%5Cbegin%7Bbmatrix%7D%20-1%20%26%201%20%26%200%20%5C%5C%201%20%26%20-1%20%26%20-1%20%5C%5C%200%20%26%200%20%26%201%20%5Cend%7Bbmatrix%7D)

A sample code to simulate this system is:


```python
from cbsa import ReactionSystem
import matplotlib.pyplot as plt
import numpy as np

S = [[-1, 1, 0],
     [ 1,-1,-1],
     [ 0, 0, 1]]

R = [[0,0,0],
     [0,0,0],
     [0,0,0]]

x0 = [50,0,0]
k = [0.5,0.1,0.8]
max_dt = 0.1
alpha=0.5
total_sim_time = 10

cbsa = ReactionSystem(S,R)

cbsa.setup()
cbsa.set_x(x0)
cbsa.set_k(k)

cbsa.setup_simulation(use_opencl=False,alpha=alpha,max_dt=max_dt)
cbsa.compute_simulation(total_sim_time)
cbsa_data = np.array(cbsa.simulation_data)

plt.plot(cbsa_data[:,0],cbsa_data[:,1],label="A")
plt.plot(cbsa_data[:,0],cbsa_data[:,2],label="B")
plt.plot(cbsa_data[:,0],cbsa_data[:,3],label="C")
plt.legend()
plt.show()
```

![example 1 image](docs/images/example_1.png)


License
-------

CBSA is licensed under the MIT License.  Please see the file [LICENCE](LICENSE) for more information.





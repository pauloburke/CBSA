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
import cbsa
import matplotlib.pyplot as plt
import numpy as np

S = [[-1, 1, 0],
     [ 1,-1,-1],
     [ 0, 0, 1]]

init_mols = [50,0,0]
k = [0.5,0.1,0.8]
max_dt = 0.1
total_sim_time = 10

sim = cbsa.ReactionSystem(S)
sim.setup()
sim.set_x(init_mols)
sim.set_k(k)
sim.set_max_dt(max_dt)

sim.setup_simulation()
sim.compute_simulation(total_sim_time)
sim_data = np.array(sim.simulation_data)

plt.plot(sim_data[:,0],sim_data[:,1],label="A")
plt.plot(sim_data[:,0],sim_data[:,2],label="B")
plt.plot(sim_data[:,0],sim_data[:,3],label="C")
plt.legend()
plt.show()
```

![example 1 image](docs/images/example_1.png)


License
-------

CBSA is licensed under the MIT License.  Please see the file [LICENCE](LICENSE) for more information.





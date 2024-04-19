# DECT - Differentiable Euler Characteristic Transform
Differentiable Euler Characteristic Transform


## Installation
Our code has been developed using python 3.10 and using pytorch 2.0.1 installed 
with CUDA 11.7. 
After installing the above, install the requirements in the requirements.txt.

```{python}
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

## Usage

To run a single experiment, run `single_main.py` and modify the path to the 
right experiment. 
The configuration files for each experiment can be found under the 
`experiment` folder and the parameters are in the `.yaml` files.

To run all experiments in a folder, update the path in `main.py` to that 
folder and run `main.py`.

All datasets will be downloaded and preprocessed when first ran via the 
`torch_geometric` package. 
The TU Datasets are small and run fast, so for testing purposes it is recommended
to run these first.

Alternatively, for research purposes the ECT is installable as a python package. 
To install DECT, run the following in the terminal

```{bash}
pip install "git+https://github.com/aidos-lab/DECT/#subdirectory=dect"
```

For example usage, we provide the `example.ipynb` file and the code therein reproduces the 
ECT of the gif of this readme. 
The code is provided on an as is basis. You are cordially invited to both contribute and 
provide feedback. Do not hesitate to contact us.

## Examples 

The core of our method, the differentiable computation of the Euler Characteristic 
transform, can be found in the `./models/layers/layers.py` folder.
Since the code is somewhat terse, highly vectorised and optimized for batch 
processing, we provide an example computation that illustrates the core 
principle of our method. 


## License

Our code is released under a BSD-3-Clause license. This license
essentially permits you to freely use our code as desired, integrate it
into your projects, and much more --- provided you acknowledge the
original authors. Please refer to [LICENSE.md](./LICENSE.md) for more
information. 

## Issues

This project is maintained by members of the [AIDOS Lab](https://github.com/aidos-lab).
Please open an [issue](https://github.com/aidos-lab/TARDIS/issues) in
case you encounter any problems.

After setting up the conda environment using the requirements.yaml in this folder, mmcv most probably won't be installed properly, so run the following command: (check the versions of PyTorch, Cuda and mmcv [here](https://mmcv.readthedocs.io/en/latest/get_started/installation.html#install-mmcv))

```bash
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
```

After the environment is set up, from the root of the repository, run the following command:

```bash
pip install -e .
```

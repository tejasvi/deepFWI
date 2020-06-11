# deepFWI
* The dependencies are present in [environments.yml](environments.yml). To create environment run:<br>
`conda env create -f environment.yml`
* The entry point for training and inference is [arch/main.py](arch/main.py). After specifying the configuration in the script, run:
`python main.py`
* The [arch](arch) directory contains the architecture implementation.
  * The [arch/dataloader](arch/dataloader) directory contains the implementation specific to the training data.
  * The [arch/model](arch/model) contains the model implementation.
  * The [arch/base.py](arch/base.py) has the common implementation used by every model.
* Code documentation is present in [arch/docs.md](arch/docs.md).
* The [data](data) contains the EDA and preprocessing done for each dataset.
  * The [data/fwi_global](data/fwi_global) has Jupyter notebooks for input and output variables used for global FWI prediction.
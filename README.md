# deepFWI
* The dependencies are present in [environment.yml](environment.yml). To create environment run:<br>
`conda env create -f environment.yml`
<br>`conda activate wildfire-dl`

    >The setup is tested on Ubuntu 18.04 only. Other systems may have problems while creating the environment. See [this](https://github.com/conda/conda/issues/7311) issue for further details.
* The entry point for training and inference is [arch/main.py](arch/main.py). After specifying the configuration in the script or through command line, run:
`python main.py`
  * **Usage**: main.py `[-h] [-init-features 11] [-in-channels 8] [-epochs 100]
               [-learning-rate 0.01] [-loss mae] [-batch-size 1] [-split 0.2]
               [-use-16bit True] [-gpus 1] [-optim one_cycle] [-model unet]
               [-out fwi_global] [-forecast-dir /nvme0/fwi-forecast]
               [-forcings-dir /nvme1/fwi-forcings] [-thresh 10.4]
               [-comment None] [-test False] [-checkpoint]`

  * **Optional arguments**:
    <pre>  -h, --help            show this help message and exit
    -init-features 11     Architecture complexity
    -in-channels 8        Number of input channels
    -epochs 100           Number of training epochs
    -learning-rate 0.01   Maximum learning rate
    -loss mae             Loss function: mae or mse
    -batch-size 1         Batch size of the input
    -split 0.2            Test split fraction
    -use-16bit True       Use 16-bit precision for training
    -gpus 1               Number of GPUs to use
    -optim one_cycle      Leraning rate optimizer: one_cycle or cosine
    -model unet           Model to use: unet
    -out fwi_global       Output data for training
    -forecast-dir /nvme0/fwi-forecast
                          Directory containing forcast data
    -forcings-dir /nvme1/fwi-forcings
                          Directory containing forcings data
    -thresh 10.4          Threshold for accuracy: Half of output MAD
    -comment None         Used for logging
    -test False           Use model for evaluation
    -checkpoint           Path to the test model checkpoint</pre>
* The [arch](arch) directory contains the architecture implementation.
  * The [arch/dataloader](arch/dataloader) directory contains the implementation specific to the training data.
  * The [arch/model](arch/model) contains the model implementation.
  * The [arch/base.py](arch/base.py) has the common implementation used by every model.
* Code documentation is present in [arch/docs.md](arch/docs.md).
* The [data](data) contains the EDA and preprocessing done for each dataset.
  * The [data/fwi_global](data/fwi_global) has Jupyter notebooks for input and output variables used for global FWI prediction.
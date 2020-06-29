# deepFWI: Prediction of Fire Weather Index

## Getting Started:
- **Clone this repo**: 
<br> `git clone https://github.com/wikilimo/deepFWI.git`
<br> `cd deepFWI`

* **Install dependencies**: To create the environment, run
<br> `conda env create -f environment.yml`
<br> `conda activate wildfire-dl`

    >The setup is tested on Ubuntu 18.04 only and will not work on any non-Linux systems. See [this](https://github.com/conda/conda/issues/7311) issue for further details.
## Running Inference
* **Testing data**:<br>
  Ensure the access to fwi-forcings and fwi-reanalysis data.
* **Obtain pre-trained model**:<br>
  Place the model checkpoint file somewhere in your system and note the filepath.
* **Run the inference script**:<br>
  `python arch/test.py -in-channels=8 -out-channels=1 -forcings-dir='path/to/forcings' -reanalysis-dir='path/to/reanalysis' -checkpoint-file='path/to/exp0-checkpoint'`
    > The number of input channels is technically equal to four times number of input days and the number of output channels is equal to number of output days. The specification will change in future to specifying the number of days instead of number of channels.

## Implementation overview
* The entry point for training is [arch/main.py](arch/main.py)
  * **Example Usage**: `python arch/main.py [-h]
               [-init-features 16] [-in-channels 16] [-out-channels 1]
               [-epochs 100] [-learning-rate 0.001] [-loss mse]
               [-batch-size 1] [-split 0.2] [-use-16bit True] [-gpus 1]
               [-optim one_cycle] [-min-data False]
               [-clip-fwi False] [-model unet_tapered_multi] [-out exp2]
               [-forcings-dir /path/to/forcings]
               [-reanalysis-dir /path/to/reanalysis]
               [-mask dataloader/mask.npy] [-thresh 9.4]
               [-comment None]`
               
* The entry point for inference is [arch/test.py](arch/test.py)
  * **Example Usage**: `python arch/test.py [-h]
               [-init-features 16] [-in-channels 16] [-out-channels 1]
               [-learning-rate 0.001] [-loss mse]
               [-batch-size 1] [-split 0.2] [-use-16bit True] [-gpus 1]
               [-min-data False] [-case-study False]
               [-clip-fwi False] [-model unet_tapered_multi] [-out exp2]
               [-forcings-dir /path/to/forcings]
               [-reanalysis-dir /path/to/reanalysis]
               [-mask dataloader/mask.npy] [-thresh 9.4]
               [-comment None] [-checkpoint-file]`

* **Configuration Details**:
<br> Optional arguments (default values indicated below):

    `  -h, --help                   show this help message and exit`<pre>
    -init-features 16                   Architecture complexity
    -in-channels 16                     Number of input channels
    -out-channels 1                     Number of output channels
    -epochs 100                         Number of training epochs
    -learning-rate 0.001                Maximum learning rate
    -loss mse                           Loss function: mae, mse
    -batch-size 1                       Batch size of the input
    -split 0.2                          Test split fraction
    -use-16bit True                     Use 16-bit precision for training (train only)
    -gpus 1                             Number of GPUs to use
    -optim one_cycle                    Learning rate optimizer: one_cycle or cosine (train only)
    -min-data False                     Use small amount of data for sanity check
    -case-study False                   Limit the analysis to Australian region (inference only)
    -clip-fwi False                     Limit the analysis to the data points with 0.5 < fwi < 60 (inference only)
    -model unet_tapered_multi           Model to use: unet, exp0_m, unet_lite, unet_tapered, exp1_m, unet_tapered_multi
    -out exp2                           Output data for training: fwi_global, exp0, exp1, exp2
    -forecast-dir /path/to/forecast     Directory containing forecast data
    -forcings-dir /path/to/forcings     Directory containing forcings data
    -reanalysis-dir /path/to/reanalysis Directory containing reanalysis data
    -mask dataloader/mask.npy           File containing the mask stored as the numpy array
    -thresh 9.4                         Threshold for accuracy: Half of output MAD
    -comment Comment of choice!         Used for logging
    -checkpoint-file                    Path to the test model checkpoint</pre>
    
* The [arch/](arch) directory contains the architecture implementation.
  * The [arch/dataloader/](arch/dataloader) directory contains the implementation specific to the training data.
  * The [arch/model/](arch/model) directory contains the model implementation.
  * The [arch/base.py](arch/base.py) directory has the common implementation used by every model.

* Code documentation is present in [arch/docs.md](arch/docs.md).
* The [data/](data) directory contains the Exploratory Data Analysis and Preprocessing required for each dataset demonstrated via Jupyter Notebooks.
  * Forcings data: [data/fwi_global/fwi_forcings.ipynb](data/fwi_global/fwi_forcings.ipynb)
  * Reanalysis data: [data/fwi_global/fwi_reanalysis.ipynb](data/fwi_global/fwi_reanalysis.ipynb)
  * Forecast data: [data//fwi_global/fwi_forecast.ipynb](data/fwi_global/fwi_forecast.ipynb)

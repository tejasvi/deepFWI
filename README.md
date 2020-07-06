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
* **Quick example**:<br>
  The [inference.ipynb](examples/inference.ipynb) demonstrates the end-to-end procedure of loading data, creating model from saved checkpoint, and getting the predictions.
* **Testing data**:<br>
  Ensure the access to fwi-forcings and fwi-reanalysis data.
* **Obtain pre-trained model**:<br>
  Place the model checkpoint file somewhere in your system and note the filepath.
* **Run the inference script**:<br>
  * Optionally set $FORCINGS_DIR and $REANALYSIS_DIR to override $PWD as the default location of data.
  `python arch/test.py -in-channels=8 -out-channels=1 -forcings-dir=${FORCINGS_DIR:-$PWD} -reanalysis-dir=${REANALYSIS_DIR:-$PWD} -checkpoint-file='path/to/checkpoint'`
    > The number of input channels is technically equal to four times the number of input days and the number of output channels is equal to number of output days. The specification will change in future to specifying the number of days instead of number of channels.

## Implementation overview
* The entry point for training is [arch/train.py](arch/train.py)
  * **Example Usage**: `python arch/train.py [-h]
               [-init-features 16] [-in-channels 16] [-out-channels 1]
               [-epochs 100] [-learning-rate 0.001] [-loss mse]
               [-batch-size 1] [-split 0.2] [-use-16bit True] [-gpus 1]
               [-optim one_cycle] [-dry-run False]
               [-clip-fwi False] [-model unet_tapered] [-out fwi_reanalysis]
               [-forcings-dir ${FORCINGS_DIR:-$PWD}]
               [-reanalysis-dir ${REANALYSIS_DIR:-$PWD}]
               [-mask dataloader/mask.npy] [-thresh 9.4]
               [-comment None]`
               
* The entry point for inference is [arch/test.py](arch/test.py)
  * **Example Usage**: `python arch/test.py [-h]
               [-init-features 16] [-in-channels 16] [-out-channels 1]
               [-learning-rate 0.001] [-loss mse]
               [-batch-size 1] [-split 0.2] [-use-16bit True] [-gpus 1]
               [-dry-run False] [-case-study False]
               [-clip-fwi False] [-model unet_tapered] [-out fwi_reanalysis]
               [-forcings-dir ${FORCINGS_DIR:-$PWD}]
               [-reanalysis-dir ${REANALYSIS_DIR:-$PWD}]
               [-mask dataloader/mask.npy] [-thresh 9.4]
               [-comment None] [-checkpoint-file]`

* **Configuration Details**:
<br> Optional arguments (default values indicated below):

    `  -h, --help                   show this help message and exit`<pre>
    -init-features 16                       Architecture complexity
    -in-channels 16                         Number of input channels
    -out-channels 1                         Number of output channels
    -epochs 100                             Number of training epochs
    -learning-rate 0.001                    Maximum learning rate
    -loss mse                               Loss function: mae, mse
    -batch-size 1                           Batch size of the input
    -split 0.2                              Test split fraction
    -use-16bit True                         Use 16-bit precision for training (train only)
    -gpus 1                                 Number of GPUs to use
    -optim one_cycle                        Learning rate optimizer: one_cycle or cosine (train only)
    -dry-run False                          Use small amount of data for sanity check
    -case-study False                       Limit the analysis to Australian region (inference only)
    -clip-fwi False                         Limit the analysis to the data points with 0.5 < fwi < 60 (inference only)
    -test-set /path/to/pickled/list         Load test-set filenames from specified file instead of random split
    -model unet_tapered               Model to use: unet, unet_downsampled, unet_snipped, unet_tapered
    -out fwi_reanalysis                     Output data for training: fwi_forecast or fwi_reanalysis
    -forecast-dir ${FORECAST_DIR:-$PWD}     Directory containing forecast data. Alternatively set $FORECAST_DIR
    -forcings-dir ${FORCINGS_DIR:-$PWD}     Directory containing forcings data. Alternatively set $FORCINGS_DIR
    -reanalysis-dir ${REANALYSIS_DIR:-$PWD} Directory containing reanalysis data. Alternatively set $REANALYSIS_DIR
    -mask dataloader/mask.npy               File containing the mask stored as the numpy array
    -thresh 9.4                             Threshold for accuracy: Half of output MAD
    -comment Comment of choice!             Used for logging
    -save-test-set False                    Save the test-set file names to the specified filepath 
    -checkpoint-file                        Path to the test model checkpoint</pre>
    
* The [arch/](arch) directory contains the architecture implementation.
  * The [arch/dataloader/](arch/dataloader) directory contains the implementation specific to the training data.
  * The [arch/model/](arch/model) directory contains the model implementation.
  * The [arch/base.py](arch/base.py) directory has the common implementation used by every model.

* Code documentation is present in [arch/docs.md](arch/docs.md).
* The [data/](data) directory contains the Exploratory Data Analysis and Preprocessing required for each dataset demonstrated via Jupyter Notebooks.
  * Forcings data: [data/fwi_global/fwi_forcings.ipynb](data/fwi_global/fwi_forcings.ipynb)
  * Reanalysis data: [data/fwi_global/fwi_reanalysis.ipynb](data/fwi_global/fwi_reanalysis.ipynb)
  * Forecast data: [data//fwi_global/fwi_forecast.ipynb](data/fwi_global/fwi_forecast.ipynb)

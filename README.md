# deepFWI: Prediction of Fire Weather Index

## Getting Started:
- **Clone this repo**:
<br> `git clone https://github.com/wikilimo/deepFWI.git`
<br> `cd deepFWI`

* **Using conda**: To create the environment, run
<br> `conda env create -f environment.yml`
<br> `conda activate wildfire-dl`

* **Using docker**: To create the image and container, run
<br> `docker build -t deepfwi .`
<br> `docker docker run -it deepfwi`

    >The setup is tested on Ubuntu 18.04 only and will not work on any non-Linux systems. See [this](https://github.com/conda/conda/issues/7311) issue for further details.
## Running Inference
* **Examples**:<br>
  The [inference_2_1.ipynb](examples/inference_2_1.ipynb) and [inference_4_10.ipynb](examples/inference_4_10.ipynb) notebooks demonstrate the end-to-end procedure of loading data, creating model from saved checkpoint, and getting the predictions for 2 day input, 1 day output; and 4 day input, 10 day output experiments respectively.
* **Testing data**:<br>
  Ensure the access to fwi-forcings and fwi-reanalysis data.
* **Obtain pre-trained model**:<br>
  Place the model checkpoint file somewhere in your system and note the filepath.
  * Checkpoint file for 2 day input, 1 day FWI prediction is available [here](src/model/checkpoints/pre_trained/2_1/epoch_41_100.ckpt)
  * Checkpoint file for 4 day input, 10 day FWI prediction is available [here](src/model/checkpoints/pre_trained/4_10/epoch_99_100.ckpt)
* **Run the inference script**:<br>
  * Optionally set `$FORCINGS_DIR` and `$REANALYSIS_DIR` to override `$PWD` as the default location of data.
  `python src/test.py -in-days=2 -out-days=1 -forcings-dir=${FORCINGS_DIR:-$PWD} -reanalysis-dir=${REANALYSIS_DIR:-$PWD} -checkpoint-file='path/to/checkpoint'`

## Implementation overview
* The entry point for training is [src/train.py](src/train.py)
  * **Example Usage**: `python src/train.py [-h] [-in-days 4] [-out-days 1] [-forcings-dir ${FORCINGS_DIR:-$PWD}] [-reanalysis-dir ${REANALYSIS_DIR:-$PWD}]`

* The entry point for inference is [src/test.py](src/test.py)
  * **Example Usage**: `python src/test.py [-h] [-in-days 4] [-out-days 1] [-forcings-dir ${FORCINGS_DIR:-$PWD}] [-reanalysis-dir ${REANALYSIS_DIR:-$PWD}] [-checkpoint-file]`

* **Configuration Details**:
<br> Optional arguments (default values indicated below):

    `  -h, --help show this help message and exit`
<pre>    -init-features 16                       Architecture complexity
    -in-days 4                              Number of input days
    -out-days 1                             Number of output days
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
    -clip-output False                      Limit the inference to the output values within supplied range (e.g. 0.5,60)
    -boxcox False                           Apply boxcox transformation with specified lambda while training and the inverse boxcox transformation during the inference.
    -binned False                           Show the extended metrics for supplied comma separated binned FWI value range
    -round-to-zero False                    Round off the target values below the specified threshold to zero
    -test-set /path/to/pickled/list         Load test-set filenames from specified file instead of random split
    -model unet_tapered                     Model to use: unet, unet_downsampled, unet_snipped, unet_tapered, unet_interpolated
    -out fwi_reanalysis                     Output data for training: fwi_forecast or fwi_reanalysis
    -forecast-dir ${FORECAST_DIR:-$PWD}     Directory containing forecast data. Alternatively set $FORECAST_DIR
    -forcings-dir ${FORCINGS_DIR:-$PWD}     Directory containing forcings data. Alternatively set $FORCINGS_DIR
    -reanalysis-dir ${REANALYSIS_DIR:-$PWD} Directory containing reanalysis data. Alternatively set $REANALYSIS_DIR
    -mask dataloader/mask.npy               File containing the mask stored as the numpy array
    -comment Comment of choice!             Used for logging
    -save-test-set False                    Save the test-set file names to the specified filepath
    -checkpoint-file                        Path to the test model checkpoint</pre>

Code walk-through can be found at [Code_Structure_Overview.md](Code_Structure_Overview.md).

* The [src/](src) directory contains the architecture implementation.
  * The [src/dataloader/](src/dataloader) directory contains the implementation specific to the training data.
  * The [src/model/](src/model) directory contains the model implementation.
  * The [src/model/base_model.py](src/model/base_model.py) script has the common implementation used by every model.

* The [data/EDA/](data/EDA/) directory contains the Exploratory Data Analysis and Preprocessing required for each dataset demonstrated via Jupyter Notebooks.
  * Forcings data: [data/EDA/fwi_forcings.ipynb](data/EDA/fwi_forcings.ipynb)
  * Reanalysis data: [data/EDA/fwi_reanalysis.ipynb](data/EDA/fwi_reanalysis.ipynb)
  * Forecast data: [data/EDA/fwi_forecast.ipynb](data/EDA/fwi_forecast.ipynb)
  * FRP data: [data/EDA/frp.ipynb](data/EDA/frp.ipynb)
  
* Code walk-through can be found at [Code_Structure_Overview.md](Code_Structure_Overview.md).

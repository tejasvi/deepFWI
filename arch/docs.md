
    
# Module `main` {#main}

Primary training and evaluation script.




    
## Functions


    
### Function `get_hparams` {#main.get_hparams}



    
> `def get_hparams(init_features: ('Architecture complexity', 'option') = 20, in_channels: ('Number of input channels', 'option') = 8, epochs: ('Number of training epochs', 'option') = 100, learning_rate: ('Maximum learning rate', 'option') = 0.001, loss: ('Loss function: mae or mse', 'option') = 'mse', batch_size: ('Batch size of the input', 'option') = 1, split: ('Test split fraction', 'option') = 0.2, use_16bit: ('Use 16-bit precision for training (train only)', 'option') = True, gpus: ('Number of GPUs to use', 'option') = 1, optim: ('Learning rate optimizer: one_cycle or cosine (train only)', 'option') = 'one_cycle', min_data: ('Use small amount of data for sanity check', 'option') = False, model: ('Model to use: unet or exp0_m or unet_lite or unet_tapered', 'option') = 'unet_tapered', out: ('Output data for training: fwi_global or exp0', 'option') = 'exp0', forecast_dir: ('Directory containing forecast data', 'option') = '/nvme0/fwi-forecast', forcings_dir: ('Directory containing forcings data', 'option') = '/nvme1/fwi-forcings', reanalysis_dir: ('Directory containing reanalysis data', 'option') = '/nvme0/fwi-reanalysis', thresh: ('Threshold for accuracy: Half of output MAD', 'option') = 9.4, comment: ('Used for logging', 'option') = 'None', checkpoint: ('Path to the test model checkpoint', 'option') = '')`


The project wide arguments. Run `python main.py -h` for usage details.

###### Returns

<code>Dict</code>
:   Dictionary containing configuration options.



    
### Function `main` {#main.main}



    
> `def main(hparams)`


Main training routine specific for this project
:param hparams:

    
### Function `str2num` {#main.str2num}



    
> `def str2num(s)`


Converts parameter strings to appropriate types.

###### Parameters

**```s```** :&ensp;<code>str</code>
:   Parameter value

###### Returns

<code>undefined</code>
:   Type converted parameter






    
# Module `exp0` {#exp0}

Experiment 0 dataset class to be used with U-Net model.





    
## Classes


    
### Class `ModelDataset` {#exp0.ModelDataset}



> `class ModelDataset(out_var=None, out_mean=None, forecast_dir=None, forcings_dir=None, reanalysis_dir=None, transform=None, hparams=None, **kwargs)`


The dataset class responsible for loading the data and providing the samples for
training.


    
#### Ancestors (in MRO)

* [dataloader.fwi_global.ModelDataset](#dataloader.fwi_global.ModelDataset)
* [torch.utils.data.dataset.Dataset](#torch.utils.data.dataset.Dataset)








    
# Module `fwi_global` {#fwi_global}

Dataset class for fwi-forcings and fwi-forecast.





    
## Classes


    
### Class `ModelDataset` {#fwi_global.ModelDataset}



> `class ModelDataset(out_var=None, out_mean=None, forecast_dir=None, forcings_dir=None, reanalysis_dir=None, transform=None, hparams=None, **kwargs)`


The dataset class responsible for loading the data and providing the samples for
training.


    
#### Ancestors (in MRO)

* [torch.utils.data.dataset.Dataset](#torch.utils.data.dataset.Dataset)






    
#### Methods


    
##### Method `test_step` {#fwi_global.ModelDataset.test_step}



    
> `def test_step(self, model, batch, batch_idx)`


Called during manual invocation on test data.

    
##### Method `training_step` {#fwi_global.ModelDataset.training_step}



    
> `def training_step(self, model, batch, batch_idx)`


Called inside the training loop with the data from the training dataloader
passed in as <code>batch</code>.

    
##### Method `validation_step` {#fwi_global.ModelDataset.validation_step}



    
> `def validation_step(self, model, batch, batch_idx)`


Called inside the validation loop with the data from the validation dataloader
passed in as <code>batch</code>.



    
# Module `base` {#base}

Base model implementing helper methods.





    
## Classes


    
### Class `BaseModel` {#base.BaseModel}



> `class BaseModel(hparams)`


The primary module containing all the training functionality. It is equivalent to
PyTorch nn.Module in all aspects.

#### Usage


Passing hyperparameters:

    >>> div=3
        x=269//div
        y=183//div
        params = dict(
            in_width=x,
            in_length=y,
            in_depth=7,
            output_size=x*y,
            drop_prob=0.5,
            epochs=20,
            optimizer_name="adam",
            batch_size=1
        )
    >>> from argparse import Namespace
    >>> hparams = Namespace(**params)
    >>> model = Model(hparams)

Pass in hyperparameters as a <code>argparse.Namespace</code> or a <code>dict</code> to the
model.


    
#### Ancestors (in MRO)

* [pytorch_lightning.core.lightning.LightningModule](#pytorch_lightning.core.lightning.LightningModule)
* [abc.ABC](#abc.ABC)
* [pytorch_lightning.utilities.device_dtype_mixin.DeviceDtypeModuleMixin](#pytorch_lightning.utilities.device_dtype_mixin.DeviceDtypeModuleMixin)
* [pytorch_lightning.core.grads.GradInformation](#pytorch_lightning.core.grads.GradInformation)
* [pytorch_lightning.core.saving.ModelIO](#pytorch_lightning.core.saving.ModelIO)
* [pytorch_lightning.core.hooks.ModelHooks](#pytorch_lightning.core.hooks.ModelHooks)
* [torch.nn.modules.module.Module](#torch.nn.modules.module.Module)






    
#### Methods


    
##### Method `add_bias` {#base.BaseModel.add_bias}



    
> `def add_bias(self, bias)`




    
##### Method `configure_optimizers` {#base.BaseModel.configure_optimizers}



    
> `def configure_optimizers(self)`


Return optimizers and learning rate schedulers.
At least one optimizer is required.

    
##### Method `forward` {#base.BaseModel.forward}



    
> `def forward(self, x)`


Forward pass

    
##### Method `prepare_data` {#base.BaseModel.prepare_data}



    
> `def prepare_data(self, ModelDataset=None, force=False)`


Load and split the data for training and test.

    
##### Method `test_dataloader` {#base.BaseModel.test_dataloader}



    
> `def test_dataloader(self)`


Implement one or multiple PyTorch DataLoaders for testing.

The dataloader you return will not be called every epoch unless you set
:paramref:`~pytorch_lightning.trainer.Trainer.reload_dataloaders_every_epoch` to <code>True</code>.

It's recommended that all data downloads and preparation happen in :meth:<code>prepare\_data</code>.

- :meth:`~pytorch_lightning.trainer.Trainer.fit`
- ...
- :meth:<code>prepare\_data</code>
- :meth:<code>train\_dataloader</code>
- :meth:<code>val\_dataloader</code>
- :meth:<code>test\_dataloader</code>


###### Note

Lightning adds the correct sampler for distributed and arbitrary hardware.
There is no need to set it yourself.


###### Return

Single or multiple PyTorch DataLoaders.


###### Example

.. code-block:: python

    def test_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (1.0,))])
        dataset = MNIST(root='/path/to/mnist/', train=False, transform=transform,
                        download=True)
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        return loader


###### Note

If you don't need a test dataset and a :meth:<code>test\_step</code>, you don't need to implement
this method.

    
##### Method `test_epoch_end` {#base.BaseModel.test_epoch_end}



    
> `def test_epoch_end(self, outputs)`


Called at the end of a test epoch with the output of all test steps.

.. code-block:: python

    # the pseudocode for these calls
    test_outs = []
    for test_batch in test_data:
        out = test_step(test_batch)
        test_outs.append(out)
    test_epoch_end(test_outs)


###### Args

**```outputs```**
:   List of outputs you defined in :meth:<code>test\_step\_end</code>, or if there
    are multiple dataloaders, a list containing a list of outputs for each dataloader



###### Return

Dict or OrderedDict: Dict has the following optional keys:

- progress_bar -> Dict for progress bar display. Must have only tensors.
- log -> Dict of metrics to add to logger. Must have only tensors (no images, etc).


###### Note

If you didn't define a :meth:<code>test\_step</code>, this won't be called.

- The outputs here are strictly for logging or progress bar.
- If you don't need to display anything, don't return anything.
- If you want to manually set current step, specify it with the 'step' key in the 'log' Dict


###### Examples

With a single dataloader:

.. code-block:: python

    def test_epoch_end(self, outputs):
        test_acc_mean = 0
        for output in outputs:
            test_acc_mean += output['test_acc']

        test_acc_mean /= len(outputs)
        tqdm_dict = {'test_acc': test_acc_mean.item()}

        # show test_loss and test_acc in progress bar but only log test_loss
        results = {
            'progress_bar': tqdm_dict,
            'log': {'test_acc': test_acc_mean.item()}
        }
        return results

With multiple dataloaders, <code>outputs</code> will be a list of lists. The outer list contains
one entry per dataloader, while the inner list contains the individual outputs of
each test step for that dataloader.

.. code-block:: python

    def test_epoch_end(self, outputs):
        test_acc_mean = 0
        i = 0
        for dataloader_outputs in outputs:
            for output in dataloader_outputs:
                test_acc_mean += output['test_acc']
                i += 1

        test_acc_mean /= i
        tqdm_dict = {'test_acc': test_acc_mean.item()}

        # show test_loss and test_acc in progress bar but only log test_loss
        results = {
            'progress_bar': tqdm_dict,
            'log': {'test_acc': test_acc_mean.item(), 'step': self.current_epoch}
        }
        return results

    
##### Method `test_step` {#base.BaseModel.test_step}



    
> `def test_step(self, batch, batch_idx)`


Called during manual invocation on test data.

    
##### Method `train_dataloader` {#base.BaseModel.train_dataloader}



    
> `def train_dataloader(self)`


Implement a PyTorch DataLoader for training.


###### Return

Single PyTorch :class:`~torch.utils.data.DataLoader`.

The dataloader you return will not be called every epoch unless you set
:paramref:`~pytorch_lightning.trainer.Trainer.reload_dataloaders_every_epoch` to <code>True</code>.

It's recommended that all data downloads and preparation happen in :meth:<code>prepare\_data</code>.

- :meth:`~pytorch_lightning.trainer.Trainer.fit`
- ...
- :meth:<code>prepare\_data</code>
- :meth:<code>train\_dataloader</code>


###### Note

Lightning adds the correct sampler for distributed and arbitrary hardware.
There is no need to set it yourself.


###### Example

.. code-block:: python

    def train_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (1.0,))])
        dataset = MNIST(root='/path/to/mnist/', train=True, transform=transform,
                        download=True)
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        return loader

    
##### Method `training_epoch_end` {#base.BaseModel.training_epoch_end}



    
> `def training_epoch_end(self, outputs)`


Called at the end of validation to aggregate outputs.
:param outputs: list of individual outputs of each validation step.

    
##### Method `training_step` {#base.BaseModel.training_step}



    
> `def training_step(self, batch, batch_idx)`


Called inside the training loop with the data from the training dataloader
passed in as <code>batch</code>.

    
##### Method `val_dataloader` {#base.BaseModel.val_dataloader}



    
> `def val_dataloader(self)`


Implement one or multiple PyTorch DataLoaders for validation.

The dataloader you return will not be called every epoch unless you set
:paramref:`~pytorch_lightning.trainer.Trainer.reload_dataloaders_every_epoch` to <code>True</code>.

It's recommended that all data downloads and preparation happen in :meth:<code>prepare\_data</code>.

- :meth:`~pytorch_lightning.trainer.Trainer.fit`
- ...
- :meth:<code>prepare\_data</code>
- :meth:<code>train\_dataloader</code>
- :meth:<code>val\_dataloader</code>
- :meth:<code>test\_dataloader</code>


###### Note

Lightning adds the correct sampler for distributed and arbitrary hardware
There is no need to set it yourself.


###### Return

Single or multiple PyTorch DataLoaders.


###### Examples

.. code-block:: python

    def val_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (1.0,))])
        dataset = MNIST(root='/path/to/mnist/', train=False,
                        transform=transform, download=True)
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        return loader

    # can also return multiple dataloaders
    def val_dataloader(self):
        return [loader_a, loader_b, ..., loader_n]


###### Note

If you don't need a validation dataset and a :meth:<code>validation\_step</code>, you don't need to
implement this method.


###### Note

In the case where you return multiple validation dataloaders, the :meth:<code>validation\_step</code>
will have an argument <code>dataset\_idx</code> which matches the order here.

    
##### Method `validation_epoch_end` {#base.BaseModel.validation_epoch_end}



    
> `def validation_epoch_end(self, outputs)`


Called at the end of validation to aggregate outputs.
:param outputs: list of individual outputs of each validation step.

    
##### Method `validation_step` {#base.BaseModel.validation_step}



    
> `def validation_step(self, batch, batch_idx)`


Called inside the validation loop with the data from the validation dataloader
passed in as <code>batch</code>.


-----

# Build and Train Mamba🐍 Model from Scratch :

This script implements the process of building, training, and fine-tuning the Mamba model from scratch. Using customizable command-line arguments, it allows for either pre-training a new model with specified parameters or fine-tuning a pretrained Mamba model. 

The project Details can be found in the [This blog post](https://medium.com/@ayoubkirouane3/our-mamba-model-journey-reflections-and-exciting-next-steps-for-darija-ai-fe856594f298) .

## Pre-Training / Fine-Tuning : 

Ensure your raw text data is placed in the `data` folder, the data should be pre-processed and cleaned for optimal training results.
#### Usage : 
- To `pre-train a new model from scratch` for `50 epochs` (default=`mamba-130m` architecture : here we start the training from scratch , you can change the architecture by changing the model arges in `train.py` file ) :
```
python train.py --pre_train True --n_epochs 50
```
- To `fine-tune a pretrained model` for `200 epochs` (here we start from existing Checkpoint):

    * 'state-spaces/mamba-2.8b-slimpj'
    * 'state-spaces/mamba-2.8b'
    * 'state-spaces/mamba-1.4b'
    * 'state-spaces/mamba-790m'
    * 'state-spaces/mamba-370m'
    * 'state-spaces/mamba-130m'

```
python train.py --finetune True --pretrained_model_name state-spaces/mamba-370m --n_epochs 200
```

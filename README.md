# PlantPathogen analysis 
Project for CSE 120

## Dummy Data

For downloading dummy data (codeparrat_json) for testing go to llamaData folder in Megatron-Llama/llamaData run:
```
python example_data.py
```

## Setup

Run an instance of the PyTorch container and mount Megatron:

```
docker pull nvcr.io/nvidia/pytorch:23.10-py3
```
    
```
docker run --gpus all -it --rm \
-v /home/broski209/documents/Megatron-Llama:/workspace/megatron \
nvcr.io/nvidia/pytorch:23.10-py3

 ```

    
## Preprocess Data
After mounting be sure to have nltk and sentencepiece as that will be needed for running experiment.

 ```
pip install sentencepiece
pip install nltk
 ```

To preprocess data do:


## Running Experiment

 ```
python tools/preprocess_data.py \
--input /workspace/megatron/llamaData/codeparrot_data.json \
--output-prefix my-llama \
--tokenizer-type Llama2Tokenizer \
--tokenizer-model /workspace/megatron/llamaData/tokenizerFiles/tokenizer.model \
--workers 4 \
--json-keys content
 ```
NOTE: 
Change codeparrot_data.json to your actual dataset if you are using a different one
--workers is changeable
--json-keys is changeable with default of text

1. Train:
    ```
    # Currently defaulted to a effb2
    python train.py
    ```

    ```
    # Can also specify changes in command line (See args.py)
    python train.py --num_epochs 5
    ```
If receiving error on train.py related to libcuda, inside train.py change:
device = "cuda" if torch.cuda.is_available() else "cpu" -> device = "cpu"

2. View Graph on tensorboard:
    ```
    tensorboard --logdir <logs_directory>

    #Example
    tensorboard --logdir ./models/
    
    ```
    Note: Tensorboard looks better outside conda environment(If using)

3. Test:
    ```
    # To test need to specify model_name, extra(subfolder name), and load_checkpoint - Example:
    python test.py --model_name EffNetB2 --extra Test --load_checkpoint 4
    ```

## How we will use Git
** This is subject to change. **


In an effort to make using Git seemless, everyone needs to follow the following guidelines: Each member has their own branch. All of your main work should be done in your branch. Once you are ready to get your changes merged, open a pull request to the "staging" branch. Once you have done this, the rest of the team will review the code and ensure that there will be no merge conflicts before the changes are pushed to the master branch. We also want to make sure all linting, type-checking, and unit tests pass before the changes are merged. Once the team is happy with what is in the staging branch, a pull request should be opened to merged the changes into the master branch. Someone will then accept the pull request, merging the changes.

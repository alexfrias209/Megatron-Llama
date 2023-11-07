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
*To preprocess data do:
 ```
python /workspace/megatron/tools/preprocess_data.py \
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


## Running Experiment

NOTE: 

Change global batch size so that the models are trained with a global batch-size of 4M tokens. Will depends on # of GPUS
 ```
/workspace/megatron/examples/llama7B.sh
/workspace/megatron/examples/llama13B.sh
/workspace/megatron/examples/llama70B.sh
 ```


NOTE: If cuda out of memory and want to test on singleGPU run
 ```
/workspace/megatron/examples/llamaTEST.sh
 ```

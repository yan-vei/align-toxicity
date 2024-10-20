## LLMs as Annotators: The Case of Irony in Implicit Hate Speech Detection

This is the repository with the code for the experiments for the project. Additionally, all the datasets and results are available
in the compiled view in the folders ./datasets and ./results already.

## Installation
Make sure you have Python >= 3.10 and pip >= 20.1 installed, then run:
```
pip install -r requirements.txt
```
Or, alternatively, if you use conda:
```
conda env create --name prompts --file=environment.yaml
```

If you wish to log results into W&B, make sure to create a .env file
which contains your W&B API key. Otherwise, W&B is not used for anything other can calculating compute times during sweeps.

## Experiments
It is best to run the experiments by making alternations in the config files under the /config folder.
These config files are managed by hydra.

To run the experiment with the settings other than default, you need to specify the model,
the dataset, and the setting: zero-shot or fewshot, for example:
```
python main.py model=vicuna dataset=hatexplain basic.is_fewshot=False
```
For example, the code above will run a zero-shot experiment for the Vicuna-13b model on the HateXplain dataset.

## Results
You can access the results by using the code from the ./explore_labels.ipyinb notebook. It calculates
all the performance metrics and semantic clusters. 
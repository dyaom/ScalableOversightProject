# ScalableOversightProject

## Setup
Install the required packages listed in `requirements.txt`. This is best done through a virtual environment; for example, on a Unix-like system:
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If your hardware supports PyTorch on GPU, update your `torch` installation with the command from https://pytorch.org/get-started/locally/, updating the installed version. For example, on Windows with CUDA 12.6, 
> pip3 install --update torch torchvision --index-url https://download.pytorch.org/whl/cu126

## Running the Project

First, ensure that the correct dataset is located in `data/dataset/`. The dataset should be a three-column `.csv` file, with exactly the columns `description`, `resume_accept`, `resume_reject`. Each row represents one resume pair and one experiment.

Next, run the `run_houdini.py` script to let the Houdini models generate resume selections. For example, this can be done using:
```
source .venv/bin/activate
cd src/
python run_houdini.py --dataset cnamuangtoun_subset_50.csv --models openai/gpt-4o meta-llama/Llama-3.1-8B-Instruct
```

Use the dataset filename located in `data/dataset`.
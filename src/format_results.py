import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd

# this script is in the src/ folder
HOME_DIRECTORY = Path(__file__).resolve().parents[1]
GUARD_DIRECTORY = HOME_DIRECTORY  / "data/guard"
RESULT_DIRECTORY = HOME_DIRECTORY / "data/win_rates"

def get_filepath(model):
    return ''.join(c if c not in r'\/:<>"?*' else '_' for c in model)

def format_results(guard_models: list[str], houdini_models: list[str],
                   dataset: str, output_path: Optional[str]):
    """
    The oversight_scaling_laws Elo scripts expect a .json of win rates
    with items "guard", "houdini", "guard_win_prob", "total_games".
    Read the separate guard output csv files and output a single such json. 
    """

    if output_path is None:
        output_file = RESULT_DIRECTORY / dataset
        output_file = output_file.with_suffix('.json')
    else:
        output_file = RESULT_DIRECTORY / output_path
    
    # iterate through every guard/houdini pair, looking for the csv
    results = []
    for guard_model in guard_models:
        guard_path = get_filepath(guard_model)
        for houdini_model in houdini_models:
            houdini_path = get_filepath(houdini_model)
            data_path = GUARD_DIRECTORY / guard_path / houdini_path / dataset

            # check if this csv exists
            if not data_path.is_file():
                print(f"Missing: Guard {guard_path}\tHoudini {houdini_path}")
                continue
                
            # if it does exist, calculate its stats
            data = pd.read_csv(data_path)
            # filter only to games with valid (0,1) results
            data = data[data['correct'].isin([0,1])]
            total_games = len(data)
            guard_win_prob = data['correct'].mean()

            results.append({
                'guard': guard_model,
                'houdini': houdini_model,
                'guard_win_prob': guard_win_prob,
                'total_games': total_games
            })
    
    # and write to json
    RESULT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as file:
        json.dump(results, file, indent=4)

def parse_args():
    parser = argparse.ArgumentParser(
        description = "Script to format csv into json for Elo processing"
    )
    
    parser.add_argument(
        "--guard_models", "--guard_model",
        type=str,
        nargs='+',
        help="The models to check as Guard models"
    )

    parser.add_argument(
        "--houdini_models", "--houdini_model",
        type=str,
        nargs='+',
        help="The models to check as Houdini models"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="The dataset file in data/dataset/ to use"
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="The filename to write the resulting .json to. Defaults to the dataset filename."
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    format_results(args.guard_models, args.houdini_models,
                   args.dataset, args.output_path)
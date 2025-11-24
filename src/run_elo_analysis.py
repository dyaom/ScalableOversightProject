import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd

from elo_utils import get_elo_bounds

# this script is in the src/ folder
HOME_DIRECTORY = Path(__file__).resolve().parents[1]
WIN_RATE_DIRECTORY = HOME_DIRECTORY / "data/win_rates"
RESULT_DIRECTORY = HOME_DIRECTORY / "data/domain_elos"

def run_elo_analysis(dataset: str, output_path: Optional[str] = None):
    """
    Prepare and run the data using the elo_utils.py script provided
    in the oversight_scaling_laws project
    """
    dataset_path = WIN_RATE_DIRECTORY / dataset
    
    with open(dataset_path) as file:
        data = json.load(file)
    data_df = pd.DataFrame(data)

    # get our two matrices (win rate, games played)
    win_rate_df = data_df.pivot_table(
        values='guard_win_prob', index='guard', columns='houdini'
    )
    num_played_df = data_df.pivot_table(
        values='total_games', index='guard', columns='houdini', fill_value=0
    )

    # pull out the ordered model lists
    guard_models = win_rate_df.index.tolist()
    houdini_models = win_rate_df.columns.tolist()

    # convert to numpy matrix for elo_utils
    win_rate_matrix = win_rate_df.to_numpy()
    num_played_matrix = num_played_df.to_numpy()

    # and perform our Elo calculations using the paper's code
    (
        row_elos, col_elos, # unused outputs?
        row_mean, row_low, row_high,
        col_mean, col_low, col_high
    ) = get_elo_bounds(win_rate_matrix, num_played_matrix)

    # structure as another json for output
    output = {
        "dataset": dataset,
        "guard_models": guard_models,
        "guard_elos": {
            "mean": row_mean.tolist(),
            "low": row_low.tolist(),
            "high": row_high.tolist(),
        },
        "houdini_models": houdini_models,
        "houdini_elos": {
            "mean": col_mean.tolist(),
            "low": col_low.tolist(),
            "high": col_high.tolist(),
        }
    }

    # and write the output to json
    if output_path is None:
        output_path = dataset
    output_file = RESULT_DIRECTORY / output_path
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as file:
        json.dump(output, file, indent=4)
    

def parse_args():
    parser = argparse.ArgumentParser(
        description = "Script to format csv into json for Elo processing"
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
    run_elo_analysis(args.dataset, args.output_path)

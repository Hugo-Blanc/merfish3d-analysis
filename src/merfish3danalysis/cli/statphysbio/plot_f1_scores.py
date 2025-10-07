from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import ast

def decode_results_to_dataframe(decode_results: dict) -> pd.DataFrame:
    """
    Convert decode_results dictionary into a Pandas DataFrame.
    Parameters
    ----------
    decode_results : dict
        The dictionary containing decode results.
    Returns
    -------
    pd.DataFrame
        A DataFrame representation of the decode results.
    """
    # Flatten the dictionary into a list of records
    records = []
    for key, value in decode_results.items():
        # If the value is a dictionary, include its keys as columns
        record = {**ast.literal_eval(key), **value}
        records.append(record)
    
    # Convert the list of records into a DataFrame
    return pd.DataFrame(records)


def plot_f1_heatmaps(res_path : Path ):
    sns.set_theme()
    
    with open(res_path, 'r') as f:
        decode_results = json.load(f)

    df = decode_results_to_dataframe(decode_results)
    
    save_folder = res_path.parent / "F1 heatmap"
    save_folder.mkdir(parents=True, exist_ok=True)
    metric = "F1 Score"
    fdr_values = np.unique(df["fdr"].to_numpy())
    for fdr_id, fdr_value in enumerate(fdr_values):
        df_heatmap = df[df["fdr"]==fdr_value].pivot(index="min_pixels", columns="ufish_threshold",
                                values=metric).iloc[::-1]
        # Draw a heatmap with the numeric values in each cell
        f, ax = plt.subplots(figsize=(9, 6))
        sns.heatmap(df_heatmap, annot=True, fmt=".2f", ax=ax, )
        f.suptitle(f"{metric} for FDR {fdr_value}")
        f.savefig(save_folder / f"{fdr_id}_{metric}_FDR_{fdr_value}_heatmap.png", dpi=300)


if __name__ == "__main__":
    res_path = Path(r"/mnt/d/EQUIPEX/Data/2025012025_statphysbio_simulation/fixed/sim_acquisition/decode_params_results.json")
    plot_f1_heatmaps(root_path=res_path)
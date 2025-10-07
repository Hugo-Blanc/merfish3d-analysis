from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
import ast

def plot_f1_sweep_from_file(
    f1_sweep_path: Path,
    sweep_info: str = ''
):
    """Plotresult from sweep through decoding parameters and calculated F1 scores.
    
    Parameters
    ----------
    root_path : Path
        The root path of the experiment.
    """
    sns.set_theme()
    save_folder = f1_sweep_path.parent
    # load and format json into a pandas Dataframe
    with open(f1_sweep_path) as f:
        f1_sweep = json.load(f)
    tidy_f1_sweep = {i:ast.literal_eval(key)|value for i, (key,value) in enumerate(list(f1_sweep.items()))}
    df_f1_sweep = pd.DataFrame.from_dict(tidy_f1_sweep, orient="index")

    # Plot and save the results as annotated heatmap
    metrics = ["F1 Score", "Precision", "Recall"]
    for metric in metrics :
        metric_heatmap = (
            df_f1_sweep
            .pivot(index="mag_thresh", columns="spotmap_threshold", values=metric)
        )
        # Draw a heatmap with the numeric values in each cell
        f, ax = plt.subplots(figsize=(9, 6))
        max_val = metric_heatmap.max().max()
        sns.heatmap(metric_heatmap,mask=metric_heatmap == max_val, annot=True, fmt="n", linewidths=.5, ax=ax, vmin=0, vmax=1, cmap="RdYlGn")
        sns.heatmap(metric_heatmap, mask=metric_heatmap != max_val, annot=True, fmt="n", annot_kws={"weight":'bold'}, linewidths=.5, ax=ax, vmin=0, vmax=1, cmap="RdYlGn", cbar=False)
        fig_name = f"Heatmap of {metric} for f1 sweep of {sweep_info}"
        f.suptitle(fig_name)
        f.savefig(save_folder / f"{fig_name}.png")

if __name__ == "__main__":
    f1_sweep_path = Path(r"/home/hblanc01/Data/fake_cells_16bit_example/sim_acquisition_ufish/decode_params_results_ufish2.json")
    run_info = f1_sweep_path.stem.split("decode_params_results_")[1]
    plot_f1_sweep_from_file(f1_sweep_path=f1_sweep_path, sweep_info=run_info)
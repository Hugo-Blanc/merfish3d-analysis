"""
Sweep through decoding parameters and calculate F1-score using known ground truth.

Shepherd 2025/08 - update for new BiFISH simulations.
Shepherd 2024/12 - create script to run on simulation.
"""

from merfish3danalysis.qi2labDataStore import qi2labDataStore
from merfish3danalysis.PixelDecoder import PixelDecoder, time_stamp
from pathlib import Path
import pandas as pd
from scipy.spatial import cKDTree
import numpy as np
import json
import seaborn as sns
import matplotlib.pyplot as plt
import ast


def calculate_F1_with_radius(
    qi2lab_coords: np.ndarray,
    qi2lab_gene_ids: np.ndarray,
    gt_coords: np.ndarray,
    gt_gene_ids: np.ndarray,
    radius: float
) -> dict:
    """Calculate F1 score based on spatial proximity and gene identity.

    Parameters
    ----------
    qi2lab_coords: NDArray
        z,y,x coordinates for found spots in microns. World coordinates.
    qi2lab_gene_ids: NDArray
        matched gene ids for found spots
    gt_coords: NDArray,
        z,y,x, coordinates for ground truth spots in microns. World coordinates.
    gt_gene_ids: NDArray
        match gene ids for ground truth spots
    radius: float
        search radius in 3D

    Returns
    -------
    resuts: dict
        results for F1 calculation.
    """

    gt_tree = cKDTree(gt_coords)

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    matched_gt_indices = set()
    for i, query_coord in enumerate(qi2lab_coords):
        qi2lab_gene_id = qi2lab_gene_ids[i]

        nearby_indices = gt_tree.query_ball_point(query_coord, r=radius)

        if not nearby_indices:
            false_positives += 1
            continue

        match_found = False
        for idx in nearby_indices:
            if idx in matched_gt_indices:
                continue

            if gt_gene_ids[idx] == qi2lab_gene_id:
                match_found = True
                true_positives += 1
                matched_gt_indices.add(idx)
                break

        if not match_found:
            false_positives += 1

    false_negatives = len(gt_gene_ids) - len(matched_gt_indices)

    precision = true_positives / \
        (true_positives + false_positives) if (true_positives +
                                               false_positives) > 0 else 0
    recall = true_positives / \
        (true_positives + false_negatives) if (true_positives +
                                               false_negatives) > 0 else 0
    f1 = 2 * precision * recall / \
        (precision + recall) if (precision + recall) > 0 else 0

    return {
        "F1 Score": round(f1, 3),
        "Precision": round(precision, 3),
        "Recall": round(recall, 3),
        "True Positives": true_positives,
        "False Positives": false_positives,
        "False Negatives": false_negatives,
    }


def calculate_F1(
    root_path: Path,
    gt_path: Path,
    search_radius: float
):
    """Calculate F1 using ground truth.

    Parameters
    ----------
    root_path: Path
        path to experiment
    gt_path: Path
        path to ground truth file
    search_radius: float
        search radius for a sphere in microns. Should be 2-3x the z step,
        depending on the amount of low-pass blur applied.

    Returns
    -------
    results: dict
        dictionary of results for F1 score calculation 
    """

    # initialize datastore
    # datastore_path = root_path / Path(r"qi2labdatastore")
    datastore = qi2labDataStore(root_path)
    gene_ids, _ = datastore.load_codebook_parsed()
    decoded_spots = datastore.load_global_filtered_decoded_spots()
    gt_spots = pd.read_csv(gt_path)
    gene_ids = np.array(gene_ids)

    # Extract coordinates and gene_ids from analyzed
    qi2lab_coords = decoded_spots[[
        'global_z', 'global_y', 'global_x']].to_numpy()
    qi2lab_gene_ids = decoded_spots['gene_id'].to_numpy()

    test_tile_data = datastore.load_local_corrected_image(
        tile=0, round=0, return_future=False)

    gt_coords = gt_spots[['Z', 'Y', 'X']].to_numpy().astype(np.float32)
    gt_coords[:, 0] *= datastore.voxel_size_zyx_um[0]
    gt_coords[:, 1] *= datastore.voxel_size_zyx_um[1]
    gt_coords[:, 2] *= datastore.voxel_size_zyx_um[2]
    # gt_coords[:, [0, 2]] = gt_coords[:, [2, 0]]
    gt_coords_offset = gt_coords
    gt_gene_ids = gene_ids[(gt_spots['Gene Label'].to_numpy(dtype=int))]

    results = calculate_F1_with_radius(
        qi2lab_coords,
        qi2lab_gene_ids,
        gt_coords_offset,
        gt_gene_ids,
        search_radius
    )

    return results


def decode_pixels(
    root_path: Path,
    min_mag_threshold: float,
    spotmap_threshold: float,
):
    """Run pixel decoding with the given parameters.

    Parameters
    ----------
    root_path : Path
        The root path of the experiment.
    mag_threshold : float
        The magnitude threshold
    spotmap_threshold : float
        The spotmap threshold.
    """
    # TODO fix
    mag_threshold = [min_mag_threshold, 10]

    # datastore_path = root_path / Path(r"qi2labdatastore")
    datastore = qi2labDataStore(root_path)
    merfish_bits = datastore.num_bits

    decoder = PixelDecoder(
        datastore=datastore,
        use_mask=False,
        merfish_bits=merfish_bits,
        verbose=0
    )

    decoder._global_normalization_vectors()
    decoder.optimize_normalization_by_decoding(
        n_random_tiles=1,
        n_iterations=1,
        minimum_pixels=9,
        spotmap_threshold=spotmap_threshold,
        magnitude_threshold=mag_threshold
    )

    decoder.decode_all_tiles(
        assign_to_cells=False,
        prep_for_baysor=False,
        minimum_pixels=9,
        magnitude_threshold=mag_threshold,
        spotmap_threshold=spotmap_threshold
    )


def sweep_decode_params(
    root_path: Path,
    gt_path: Path,
    save_folder: Path = None,
    spotmap_threshold_range: tuple[float] = (0.05, 0.1),
    spotmap_threshold_step: float = 0.01,
    mag_threshold_range: tuple[float] = (0.5, 1),
    mag_threshold_step: float = 0.1,
):
    """Sweep through decoding parameters and calculate F1 scores.

    Parameters
    ----------
    root_path : Path
        The root path of the experiment.
    gt_path : Path
        The path to the ground truth spots.
    spotmap_threshold_range : tuple, default [0.05,0.3]
        The range of spotmap thresholds to sweep through.
    spotmap_threshold_step : float, default .05
        The step size for the spotmap threshold sweep
    mag_threshold_range : tuple, default [1.0,2.0]
        The range of minimum magnitude threshold to sweep through.
    mag_threshold_step : float, default 0.05
        The step size for the magnitude threshold.
    """
    assert gt_path.exists(), f"GT path not found: {gt_path}"

    if save_folder == None:
        save_folder = root_path

    mag_values = np.arange(
        mag_threshold_range[0],
        mag_threshold_range[1],
        mag_threshold_step,
        dtype=np.float32
    ).tolist()

    spotmap_values = np.arange(
        spotmap_threshold_range[0],
        spotmap_threshold_range[1],
        spotmap_threshold_step,
        dtype=np.float32
    ).tolist()

    results = {}
    save_path = save_folder / f"decode_params_results {root_path.name}.json"

    for spotmap in spotmap_values:
        for mag in mag_values:
            params = {
                "fdr": .05,
                "min_pixels": 9,
                "mag_thresh": round(mag, 2),
                "spotmap_threshold": round(spotmap, 2)
            }

            try:
                print(time_stamp(
                ), f"spotmap threshold: {round(spotmap, 2)}; magnitude threshold: {round(mag, 2)}")
                decode_pixels(
                    root_path=root_path,
                    min_mag_threshold=round(mag, 2),
                    spotmap_threshold=round(spotmap, 2),
                )

                result = calculate_F1(
                    root_path=root_path,
                    gt_path=gt_path,
                    search_radius=0.75
                )
            except Exception as e:
                result = {"error": str(e)}

            results[str(params)] = result

            print(result)

            with save_path.open(mode='w', encoding='utf-8') as file:
                json.dump(results, file, indent=2)


def plot_heatmap_f1_sweep(
    root_path: Path,
    save_folder: Path = None,
    sweep_info: str = ''
):
    """Plot result from sweep through decoding parameters and calculated F1 scores.

    Parameters
    ----------
    root_path : Path
        The root path of the experiment.
    """
    if save_folder == None:
        save_folder = root_path

    sns.set_theme()

    # load and format json into a pandas Dataframe
    # TODO find a better io handling
    f1_sweep_path = save_folder / \
        f"decode_params_results {root_path.name}.json"
    with open(f1_sweep_path) as f:
        f1_sweep = json.load(f)
    tidy_f1_sweep = {i: ast.literal_eval(
        key) | value for i, (key, value) in enumerate(list(f1_sweep.items()))}
    df_f1_sweep = pd.DataFrame.from_dict(tidy_f1_sweep, orient="index")

    # Plot and save the results as annotated heatmap
    metrics = ["F1 Score", "Precision", "Recall"]
    for metric in metrics:
        metric_heatmap = (
            df_f1_sweep
            .pivot(index="mag_thresh", columns="spotmap_threshold", values=metric)
        )
        # Draw a heatmap with the numeric values in each cell
        f, ax = plt.subplots(figsize=(9, 6))
        max_val = metric_heatmap.max().max()
        sns.heatmap(metric_heatmap, mask=metric_heatmap == max_val, annot=True,
                    fmt="n", linewidths=.5, ax=ax, vmin=0, vmax=1, cmap="RdYlGn")
        sns.heatmap(metric_heatmap, mask=metric_heatmap != max_val, annot=True, fmt="n", annot_kws={
                    "weight": 'bold'}, linewidths=.5, ax=ax, vmin=0, vmax=1, cmap="RdYlGn", cbar=False)
        fig_name = f"Heatmap of {metric} for f1 sweep {sweep_info}"
        f.suptitle(fig_name)
        f.savefig(save_folder / f"{fig_name}.png")


if __name__ == "__main__":
    root_path = Path(
        r"/home/hblanc01/Data/simu_igfl/grid_simu_SNR_SBR_Density/qi2labdatastore/qi2labdatastore Img 5 simu MERFISH 8 bits sbr 5 snr 5 density 5 sample 0")
    gt_path = Path(
        r"/home/hblanc01/Data/simu_igfl/grid_simu_SNR_SBR_Density/gt_points/gt 5 simu MERFISH 8 bits sbr 5 snr 5 density 5 sample 0.csv")
    run_info = ""
    sweep_decode_params(root_path=root_path, gt_path=gt_path,
                        spotmap_threshold_range=(0.1, 0.6),
                        spotmap_threshold_step=0.05,
                        mag_threshold_range=(0.5, 2),
                        mag_threshold_step=0.1,)
    plot_heatmap_f1_sweep(root_path=root_path, sweep_info=run_info)

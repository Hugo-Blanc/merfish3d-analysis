"""
Convert igfl simulated data into a fake acquisition

The simulation comes as a 4D tiff file with only 1 round and all channels already registered in the same global space.
Analyse each tiff file as a separate fake acquisition. 

Required user parameters for system dependent variables are at end of script.

Blanc 2025/10 - Adapt code to inhouse igfl simulated data
Shepherd 2025/08 - update for new BiFISH simulations.
Shepherd 2024/12 - create script based on metadata from Max in statphysbio lab.
"""

import builtins
import gc
from itertools import compress
from tqdm import tqdm
from psfmodels import make_psf
from merfish3danalysis.qi2labDataStore import qi2labDataStore
import multiprocessing as mp
from pathlib import Path
from tifffile import imread, imwrite
from merfish3danalysis.utils.dataio import read_metadatafile, write_metadata
from typing import Optional
import numpy as np
import pandas as pd
import shutil
import json

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=FutureWarning)
mp.set_start_method('spawn', force=True)


def convert_simulation(
    root_path: Path,
):
    """Convert statphysbio simulation into a fake acquisition qi2lab datastore.

    Parameters
    ----------
    root_path: Path
        path to simulation
    output_path: Optional[Path]
        path to save fake acquisition. Default = None
    """
    # Intit
    # ------------------------
    # load metadata
    metadata_path = root_path / Path("metadata_simulation.json")
    with open(metadata_path, 'r') as file:
        metadata = json.load(file)
    num_ch = metadata["num_channels"]
    num_bits = metadata["num_bits"]
    yx_pixel_um = metadata["pixel_size"]["xy"]
    z_pixel_um = metadata["pixel_size"]["z"]

    # load simulated data
    tile_folder = root_path / "Tiles"
    img_paths = list(tile_folder.glob("Img*.tiff"))
    img_paths.sort()

    # reshape simulation to match experimental design
    img_idx = 0
    simulation_data = np.swapaxes(imread(img_paths[img_idx]), 0, 1)
    print(img_paths[img_idx])
    root_name = img_paths[img_idx].stem
    print(f"simulation shape: {simulation_data.shape}")
    fake_stage_position_zyx_um = [
        0.0,
        -1*yx_pixel_um*(simulation_data.shape[-2]//2),
        -1*yx_pixel_um*(simulation_data.shape[-1]//2)
    ]
    fake_tile_id = 0  # TODO change with Img ID
    round_id = 1

    # create sim files folder
    simufiles_folder = root_path / Path("Sim files")
    simufiles_folder.mkdir(exist_ok=True)

    # execute fake experiment. Don't write all metadata to images, just what we need.
    stage_metadata_path = simufiles_folder / \
        f"{root_name}_r{str(round_id).zfill(4)}_tile{str(fake_tile_id).zfill(4)}_stage_positions.csv"
    current_stage_data = {'stage_x': float(fake_stage_position_zyx_um[2]),
                          'stage_y': float(fake_stage_position_zyx_um[1]),
                          'stage_z': float(fake_stage_position_zyx_um[0]),
                          }
    write_metadata(current_stage_data, stage_metadata_path)

    # copy codebook and load file to simulated acquisition folder
    sim_codebook_path = root_path / Path(f"{metadata["codebook_name"]}.csv")
    codebook = pd.read_csv(sim_codebook_path)

    # copy load bit_order file to simulated acquisition folder
    sim_acq_bitorder_path = simufiles_folder / Path("bit_order.csv")
    channels_wl = [str(wl) for wl in metadata["wavelength_channels"].values()]
    channels_names = ["fiducial",] + [f"wl{n}" for n in range(num_bits)]
    df_experiment_order = pd.DataFrame(
        [[1,] + [x for x in range(1, num_bits+1)]], columns=channels_names)
    df_experiment_order.to_csv(sim_acq_bitorder_path, index=False)
    experiment_order = df_experiment_order.values

    # load missing experiment metadata
    num_z = metadata["tile_size"]["z"]
    offset = float(0)
    channel_order_bool = False
    if channel_order_bool:
        channel_order = "reversed"
    else:
        channel_order = "forward"
    ri_s = float(metadata["microscope_parameters"]["RI_specimen"])

    em_wavelengths_um = [float(wl)
                         for wl in channels_wl]  # selected by channel IDs
    # TODO check if it's necessary to fetech real ex wl values
    ex_wavelengths_um = em_wavelengths_um
    noise_map = None
    affine_zyx_px = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]], dtype=np.float32)

    # Create datastore
    # ------------------------
    # create qi2labdatastore folder
    qi2labdatastore_folder = root_path / "qi2labdatastore"

    # TODO add loop over sim images

    # initialize datastore
    datastore_path = qi2labdatastore_folder / f"qi2labdatastore {root_name}"
    datastore = qi2labDataStore(datastore_path)

    # parameters from qi2lab microscope metadata
    datastore.channels_in_data = channels_names
    datastore.num_rounds = 1
    datastore.codebook = codebook
    datastore.experiment_order = experiment_order
    datastore.num_tiles = 1
    if z_pixel_um < 0.5:
        datastore.microscope_type = "3D"
    else:
        datastore.microscope_type = "2D"
    datastore.camera_model = "simulated"
    datastore.tile_overlap = 0.2  # default
    datastore.e_per_ADU = float(1)
    datastore.offset = float(0)
    datastore.na = float(metadata["microscope_parameters"]["NA"])
    datastore.ri = float(metadata["microscope_parameters"]["RI_immersion"])
    datastore.binning = 1
    datastore.voxel_size_zyx_um = [z_pixel_um, yx_pixel_um, yx_pixel_um]

    # generate PSFs
    # --------------
    channel_psfs = []
    for channel_id in range(len(channels_names[1:])):
        if datastore.microscope_type == "3D":
            psf = make_psf(
                z=num_z,
                nx=51,
                dxy=datastore.voxel_size_zyx_um[1],
                dz=datastore.voxel_size_zyx_um[0],
                NA=datastore.na,
                wvl=em_wavelengths_um[channel_id % len(channels_wl)],
                ns=ri_s,
                ni=datastore.ri,
                ni0=datastore.ri,
                model="vectorial",
            ).astype(np.float32)
            psf = psf / np.sum(psf, axis=(0, 1, 2))
            channel_psfs.append(psf)
        else:
            psf = make_psf(
                z=1,
                nx=51,
                dxy=datastore.voxel_size_zyx_um[1],
                dz=datastore.voxel_size_zyx_um[0],
                NA=datastore.na,
                wvl=em_wavelengths_um[channel_id],
                ns=ri_s,
                ni=datastore.ri,
                ni0=datastore.ri,
                model="vectorial",
            ).astype(np.float32)
            psf = psf / np.sum(psf, axis=(0, 1, 2))
            channel_psfs.append(psf)
    channel_psfs = np.asarray(channel_psfs, dtype=np.float32)
    datastore.channel_psfs = channel_psfs

    # Update datastore state to note that calibrations are done
    datastore_state = datastore.datastore_state
    datastore_state.update({"Calibrations": True})
    datastore.datastore_state = datastore_state

    # Loop over data and create datastore.
    for round_idx in tqdm(range(datastore.num_rounds), desc="rounds"):
        # Get all stage positions for this round
        position_list = []
        for tile_idx in range(datastore.num_tiles):
            stage_position_path = simufiles_folder / Path(
                root_name
                + "_r"
                + str(round_idx + 1).zfill(4)
                + "_tile"
                + str(tile_idx).zfill(4)
                + "_stage_positions.csv"
            )
            stage_positions = read_metadatafile(stage_position_path)
            stage_x = np.round(float(stage_positions["stage_x"]), 2)
            stage_y = np.round(float(stage_positions["stage_y"]), 2)
            stage_z = np.round(float(stage_positions["stage_z"]), 2)
            temp = [stage_z, stage_y, stage_x]
            position_list.append(np.asarray(temp))
        position_list = np.asarray(position_list)

        for tile_idx in tqdm(range(datastore.num_tiles), desc="tile", leave=False):
            # initialize datastore tile
            # this creates the directory structure and links fiducial rounds <-> readout bits
            if round_idx == 0:
                datastore.initialize_tile(tile_idx)

            # load raw image
            raw_image = simulation_data

            # load raw data and make sure it is the right shape. If not, write
            # zeros for this round/stage position.

            if tile_idx == 0 and round_idx == 0:
                correct_shape = simulation_data.shape
            # if raw_image is None or raw_image.shape != correct_shape:
            #     if raw_image.shape[0] < correct_shape[0]:
            #         print("\nround=" + str(round_idx + 1) + "; tile=" + str(tile_idx + 1))
            #         print("Found shape: " + str(raw_image.shape))
            #         print("Correct shape: " + str(correct_shape))
            #         print("Replacing data with zeros.\n")
            #         raw_image = np.zeros(correct_shape, dtype=np.uint16)
            #     else:
            #         # print("\nround=" + str(round_idx + 1) + "; tile=" + str(tile_idx + 1))
            #         # print("Found shape: " + str(raw_image.shape))
            #         size_to_trim = raw_image.shape[1] - correct_shape[1]
            #         raw_image = raw_image[:,size_to_trim:,:].copy()
            #         # print("Correct shape: " + str(correct_shape))
            #         # print("Corrected to shape: " + str(raw_image.shape) + "\n")

            # Correct if channels were acquired in reverse order (red->purple)
            if channel_order == "reversed":
                raw_image = np.flip(raw_image, axis=0)

            # Correct for known camera gain and offset
            if raw_image.dtype != np.uint16:
                print(f"Image dtype is {raw_image.dtype}. Convert to uint16.")
                raw_image = raw_image.astype(np.uint16)
            gain_corrected = True
            hot_pixel_corrected = False

            # load stage position
            corrected_y = position_list[tile_idx, 1]
            corrected_x = position_list[tile_idx, 2]
            corrected_x = np.round(corrected_x, 2)
            corrected_y = np.round(corrected_y, 2)
            stage_z = np.round(position_list[tile_idx, 0], 2)
            stage_pos_zyx_um = np.asarray(
                [stage_z, corrected_y, corrected_x], dtype=np.float32
            )

            # write fidicual data (ch_idx = 0) and metadata, use first wl as fake
            datastore.save_local_corrected_image(
                raw_image[0],
                tile=tile_idx,
                psf_idx=0,
                gain_correction=gain_corrected,
                hotpixel_correction=hot_pixel_corrected,
                shading_correction=False,
                round=round_idx,
            )
            datastore.save_local_stage_position_zyx_um(
                stage_pos_zyx_um,
                affine_zyx_px,
                tile=tile_idx,
                round=round_idx
            )
            datastore.save_local_wavelengths_um(
                (ex_wavelengths_um[0], em_wavelengths_um[0]),
                tile=tile_idx,
                round=round_idx,
            )

            # write readout channels and metadata
            for bit_idx in tqdm(range(num_bits), desc="bit channels", leave=False):
                datastore.save_local_corrected_image(
                    raw_image[bit_idx+1],
                    tile=tile_idx,
                    psf_idx=bit_idx,
                    gain_correction=gain_corrected,
                    hotpixel_correction=False,
                    shading_correction=False,
                    bit=bit_idx,
                )
                datastore.save_local_wavelengths_um(
                    (ex_wavelengths_um[bit_idx % len(channels_wl)],
                     em_wavelengths_um[bit_idx % len(channels_wl)]),
                    tile=tile_idx,
                    bit=bit_idx,
                )

    datastore.noise_map = np.zeros(
        (3, correct_shape[1], correct_shape[2]), dtype=np.float32)
    datastore._shading_maps = np.ones(
        (3, correct_shape[1], correct_shape[2]), dtype=np.float32)
    datastore_state = datastore.datastore_state
    datastore_state.update({"Corrected": True})
    datastore.datastore_state = datastore_state


if __name__ == "__main__":
    root_path = Path(
        r"/home/hblanc01/Data/simu_igfl/grid_simu_SNR_SBR_Density")
    convert_simulation(root_path=root_path)

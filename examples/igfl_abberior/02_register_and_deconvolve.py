"""
Perform registration on Droso LD qi2labdatastore. By default creates a max 
projection downsampled polyDT OME-TIFF for cellpose parameter optimization.

"""

from merfish3danalysis.qi2labDataStore import qi2labDataStore
from merfish3danalysis.DataRegistration import DataRegistration
from pathlib import Path
import numpy as np
import gc
from tqdm import tqdm
from tifffile import TiffWriter
from typing import Optional

def local_register_data(root_path: Path):
    """Register each tile across rounds in local coordinates.

    Parameters
    ----------
    root_path: Path
        path to experiment
    """

    # initialize datastore
    datastore_path = root_path / Path(r"qi2labdatastore")
    datastore = qi2labDataStore(datastore_path)
    
    # initialize registration class
    registration_factory = DataRegistration(
        datastore=datastore, 
        perform_optical_flow=True, 
        overwrite_registered=False,
        save_all_polyDT_registered=False, 
        decon_iters=10,
        decon_background=0,
    )

    # run local registration across rounds
    registration_factory.register_all_tiles()

    # update datastore state
    datastore_state = datastore.datastore_state
    datastore_state.update({"LocalRegistered": True})
    datastore.datastore_state = datastore_state


def global_register_data(
    root_path : Path, 
    create_max_proj_tiff: Optional[bool] = True
):
    """Register all tiles in first round in global coordinates.

    Parameters
    ----------
    root_path: Path
        path to experiment
    
    create_max_proj_tiff: Optional[bool]
        create max projection tiff in the segmentation/cellpose directory. 
        Default = True
    """

    from multiview_stitcher import spatial_image_utils as si_utils
    from multiview_stitcher import msi_utils, registration, fusion
    import dask.diagnostics
    import dask.array as da

    # initialize datastore
    datastore_path = root_path / Path(r"qi2labdatastore")
    datastore = qi2labDataStore(datastore_path)

    # load tile positions
    for tile_idx, tile_id in enumerate(datastore.tile_ids):
        round_id = datastore.round_ids[0]
        tile_position_zyx_um = datastore.load_local_stage_position_zyx_um(
            tile_id, round_id
        )

    # convert local tiles from first round to multiscale spatial images
    msims = []
    for tile_idx, tile_id in enumerate(tqdm(datastore.tile_ids, desc="tile")):
        round_id = datastore.round_ids[0]

        voxel_zyx_um = datastore.voxel_size_zyx_um

        scale = {"z": voxel_zyx_um[0], "y": voxel_zyx_um[1], "x": voxel_zyx_um[1]}

        tile_stage_position_zyx_um, tile_affine_zyx_px = datastore.load_local_stage_position_zyx_um(
            tile_id, round_id
        )

        tile_grid_positions = {
            "z": np.round(tile_stage_position_zyx_um[0], 2),
            "y": np.round(tile_stage_position_zyx_um[1], 2),
            "x": np.round(tile_stage_position_zyx_um[2], 2),
        }

        im_data = []
        im_data = datastore.load_local_registered_image(
            tile=tile_id, round=round_id, return_future=False
        )

        sim = si_utils.get_sim_from_array(
            da.expand_dims(im_data, axis=0),
            dims=("c", "z", "y", "x"),
            scale=scale,
            translation=tile_grid_positions,
            transform_key="stage_metadata",
        )

        msim = msi_utils.get_msim_from_sim(sim, scale_factors=[])
        msims.append(msim)
        del im_data
        gc.collect()

    # perform registration in 2 steps, from most downsampling to least.
    with dask.config.set(**{"array.slicing.split_large_chunks": False}):
        with dask.diagnostics.ProgressBar():
            _ = registration.register(
                msims,
                reg_channel_index=0,
                transform_key="stage_metadata",
                new_transform_key="translation_registered_3x",
                registration_binning={"z": 3, "y": 9, "x": 9},
                post_registration_do_quality_filter=True,
            )

            _ = registration.register(
                msims,
                reg_channel_index=0,
                transform_key="translation_registered_3x",
                new_transform_key="translation_registered",
                registration_binning={"z": 1, "y": 3, "x": 3},
                post_registration_do_quality_filter=True,
            )

    # extract and save transformations into datastore
    for tile_idx, msim in enumerate(msims):
        affine = msi_utils.get_transform_from_msim(
            msim, transform_key="translation_registered"
        ).data.squeeze()
        affine = np.round(affine, 2)
        origin = si_utils.get_origin_from_sim(
            msi_utils.get_sim_from_msim(msim), asarray=True
        )
        spacing = si_utils.get_spacing_from_sim(
            msi_utils.get_sim_from_msim(msim), asarray=True
        )

        datastore.save_global_coord_xforms_um(
            affine_zyx_um=affine,
            origin_zyx_um=origin,
            spacing_zyx_um=spacing,
            tile=tile_idx,
        )

    # perform and save downsampled fusion
    with dask.config.set(**{"array.slicing.split_large_chunks": False}):
        fused_sim = fusion.fuse(
            [msi_utils.get_sim_from_msim(msim, scale="scale0") for msim in msims],
            transform_key="translation_registered",
            output_spacing={
                "z": voxel_zyx_um[0],
                "y": voxel_zyx_um[1], 
                "x": voxel_zyx_um[1],
            },
            output_chunksize=128,
            overlap_in_pixels=64,
        )
        # * np.round(voxel_zyx_um[0] / voxel_zyx_um[1], 1),

        fused_msim = msi_utils.get_msim_from_sim(fused_sim, scale_factors=[])
        affine = msi_utils.get_transform_from_msim(
            fused_msim, transform_key="translation_registered"
        ).data.squeeze()
        origin = si_utils.get_origin_from_sim(
            msi_utils.get_sim_from_msim(fused_msim), asarray=True
        )
        spacing = si_utils.get_spacing_from_sim(
            msi_utils.get_sim_from_msim(fused_msim), asarray=True
        )

        del fused_msim

        datastore.save_global_fidicual_image(
            fused_image=fused_sim.data.compute(scheduler="threads", num_workers=12),
            affine_zyx_um=affine,
            origin_zyx_um=origin,
            spacing_zyx_um=spacing,
        )

        del fused_sim
        gc.collect()

    # update datastore state
    datastore_state = datastore.datastore_state
    datastore_state.update({"GlobalRegistered": True})
    datastore_state.update({"Fused": True})
    datastore.datastore_state = datastore_state
    
    # write max projection OME-TIFF for cellpose GUI
    if create_max_proj_tiff:
        # load downsampled, fused polyDT image and coordinates 
        polyDT_fused, _, _, spacing_zyx_um = datastore.load_global_fidicual_image(return_future=False)
        # create max projection
        polyDT_max_projection = np.max(np.squeeze(polyDT_fused),axis=0)
        del polyDT_fused
        
        filename = 'DAPI_max_projection.ome.tiff'
        filename_path = datastore._datastore_path / Path("fused") / Path(filename)
        with TiffWriter(filename_path, bigtiff=True, mode='w') as tif:
            metadata={
                'axes': 'YX',
                'SignificantBits': 16,
                'PhysicalSizeX': spacing_zyx_um[2],
                'PhysicalSizeXUnit': 'µm',
                'PhysicalSizeY': spacing_zyx_um[1],
                'PhysicalSizeYUnit': 'µm',
            }
            options = dict(
                compression='zlib',
                compressionargs={'level': 8},
                predictor=True,
                photometric='minisblack',
                resolutionunit='MICROMETER',
            )
            tif.write(
                polyDT_max_projection,
                resolution=(
                    1e6 / spacing_zyx_um[1],
                    1e6 / spacing_zyx_um[2]
                ),
                **options,
                metadata=metadata
            )
    
if __name__ == "__main__":
    root_path = Path(r"/mnt/d/EQUIPEX/Data/20250404_Abberior_Merfish_7C")
    local_register_data(root_path)
    global_register_data(root_path)
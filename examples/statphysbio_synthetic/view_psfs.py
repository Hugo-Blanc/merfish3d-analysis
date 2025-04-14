import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=FutureWarning)
import napari
from pathlib import Path
from cmap import Colormap
from multiview_stitcher import vis_utils

from merfish3danalysis.qi2labDataStore import qi2labDataStore

# import multiprocessing as mp
# mp.set_start_method('spawn', force=True)


def view_psfs(root_path: Path):
    """Load and view all individual channels using neuroglancer.
    
    Parameters
    ----------
    root_path: Path
        path to experiment
    """
    
    # generate 17 colormaps
    colormaps = [
        Colormap("cmap:white").to_napari(),
        Colormap("cmap:magenta").to_napari(),
        Colormap("cmap:cyan").to_napari(),
        Colormap("cmap:red").to_napari(),
        Colormap("cmap:yellow").to_napari(),
        Colormap("cmasher:cosmic").to_napari(),
        Colormap("cmasher:dusk").to_napari(),
        Colormap("cmasher:eclipse").to_napari(),
        Colormap("cmasher:emerald").to_napari(),
        Colormap("chrisluts:BOP_Orange").to_napari(),
        Colormap("cmasher:sapphire").to_napari(),
        Colormap("chrisluts:BOP_Blue").to_napari(),
        Colormap("cmap:magenta").to_napari(),
        Colormap("cmap:cyan").to_napari(),
        Colormap("cmap:red").to_napari(),
        Colormap("cmap:yellow").to_napari(),
        Colormap("cmasher:cosmic").to_napari(),
    ]
    # initialize datastore
    datastore_path = root_path / Path(r"qi2labdatastore")
    datastore = qi2labDataStore(datastore_path)
    datastore_state = datastore.datastore_state
    assert datastore_state["Calibrations"], "Calibration is not done, no psfs to display."
    
    psfs= datastore.channel_psfs
    print(datastore.voxel_size_zyx_um)
    
    # populate napari viewer with all channels
    viewer = napari.Viewer()
    for ch_idx in range(psfs.shape[0]):
        # # use different contrast limits for polyDT vs FISH channels
        # if ch_idx == 0:
        #     contrast_limits = [0,1000]
        # else:
        #     contrast_limits = [10,500]
        viewer.add_image(psfs[ch_idx],
            blending="additive",
            colormap = colormaps[ch_idx],
            contrast_limits=[0,1]
        )
    napari.run()
    
if __name__ == "__main__":
    root_path = Path(r"/mnt/d/EQUIPEX/Data/2025012025_statphysbio_simulation/fixed/sim_acquisition")
    view_psfs(root_path)
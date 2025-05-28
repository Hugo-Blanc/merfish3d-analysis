import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=FutureWarning)
import napari
from pathlib import Path
from cmap import Colormap
from multiview_stitcher import vis_utils

from merfish3danalysis.qi2labDataStore import qi2labDataStore
from merfish3danalysis.PixelDecoder import PixelDecoder
# import multiprocessing as mp
# mp.set_start_method('spawn', force=True)


def view_decoded_tile(root_path: Path, tile_idx: int = 0):
    """Load and view all individual channels using neuroglancer.
    
    Parameters
    ----------
    root_path: Path
        path to experiment
    """
    # initialize datastore
    datastore_path = root_path / Path(r"qi2labdatastore")
    datastore = qi2labDataStore(datastore_path)
    merfish_bits = datastore.num_bits
    
    minimum_pixels_per_RNA = 2,
    ufish_threshold = 0.25,

    # initialize decodor class
    decoder = PixelDecoder(
        datastore=datastore, 
        use_mask=False, 
        merfish_bits=merfish_bits, 
        verbose=2
    )
    
    decoder.decode_one_tile(tile_idx=tile_idx, display_results=True,
        minimum_pixels= minimum_pixels_per_RNA,
        ufish_threshold= ufish_threshold)
    
    
if __name__ == "__main__":
    root_path = Path(r"/mnt/d/EQUIPEX/Data/2025012025_statphysbio_simulation/fixed/sim_acquisition")
    view_decoded_tile(root_path, tile_idx=0)
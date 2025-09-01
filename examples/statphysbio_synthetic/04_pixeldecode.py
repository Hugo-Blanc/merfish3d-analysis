"""
Decode using qi2lab GPU decoder.

Shepherd 2025/08 - update for new BiFISH simulations.
Shepherd 2024/12 - create script to run on simulation.
"""

from merfish3danalysis.qi2labDataStore import qi2labDataStore
from merfish3danalysis.PixelDecoder import PixelDecoder
from pathlib import Path

def decode_pixels(
    root_path: Path,
    minimum_pixels_per_RNA: int = 9,
    ufish_threshold: float = 0.25,
    magnitude_threshold: float = 1.5,
    fdr_target: float = .05
):
    """Perform pixel decoding.

    Parameters
    ----------
    root_path: Path
        path to experiment
    minimum_pixels_per_RNA : int
        minimum pixels with same barcode ID required to call a spot. Default = 9.
    ufish_threshold : float
        threshold to accept ufish prediction. Default = 0.25
    magnitude_threshold: float
        minimum magnitude across all normalized bits required to accept a spot. Default = 1.5
    fdr_target : float
        false discovery rate (FDR) target. Default = .05
    """

    # initialize datastore
    datastore_path = root_path / Path(r"qi2labdatastore")
    datastore = qi2labDataStore(datastore_path)
    merfish_bits = datastore.num_bits

    # initialize decodor class
    decoder = PixelDecoder(
        datastore=datastore, 
        use_mask=False, 
        merfish_bits=merfish_bits, 
        verbose=1
    )

    decoder.optimize_normalization_by_decoding(
        n_random_tiles=1,
        n_iterations=1,
        magnitude_threshold=magnitude_threshold,
        minimum_pixels=minimum_pixels_per_RNA,
        ufish_threshold=ufish_threshold
    )

    # decoder.decode_one_tile(
    #     tile_idx = 0,
    #     gpu_id=0,
    #     display_results=False,
    #     magnitude_threshold=magnitude_threshold,
    #     minimum_pixels=minimum_pixels_per_RNA,
    #     ufish_threshold=ufish_threshold, 
    #     use_normalization=False,
    # )
    
    
    """
    if you need to access normalizations, they are class properties that can
    be accessed as follows:
    
    global_background_normalization = datastore.global_background_vector
    global_foreground_normalization = datastore.global_normalization_vector
    
    iterative_background_normalization = datastore.iterative_background_vector
    iterative_foreground_normalization = datastore.iterative_normalization_vector
    
    These are 1xN arrays, where N is the number of bits in the codebook.
    
    To set a vector, you can do as follows. Assume a 16 bit codebook,
    
    datastore.iterative_background_vector = np.zeros(16,dtype=np.float32)
    """
    
    decoder.decode_all_tiles(
        assign_to_cells=False,
        prep_for_baysor=False,
        magnitude_threshold=magnitude_threshold,
        minimum_pixels=minimum_pixels_per_RNA,
        ufish_threshold=ufish_threshold,
        fdr_target=fdr_target
    )
    

if __name__ == "__main__":
    root_path = Path(r"/home/hblanc01/Data/fake_cells_16bit_example/sim_acquisition")
    decode_pixels(root_path=root_path)
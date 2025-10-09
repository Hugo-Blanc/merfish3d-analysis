See comments in each file for specifics.

Use simulation code (https://github.com/Hugo-Blanc/simu-merfish) to generate simulated merfish microscopy data

At the bottom of each python file, change the path to where you unzipped the data.

Order to run:

01_convert_simulation_to_datastore.py (seconds)
02_register_and_deconvolve.py (minutes)
03_pixeldecode.py (minutes)
04_calculate_F1.py (seconds)

Note: decoding parameters have not been optimized, so the F1 score can be improved.

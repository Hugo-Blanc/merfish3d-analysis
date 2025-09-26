See comments in each file for specifics.

Use simulation code (https://github.com/Hugo-Blanc/simu-merfish) to generate simulated merfish microscopy data

At the bottom of each python file, change the path to where you unzipped the data.

Order to run:

01_convert_simulation_to_experiment.py (seconds)
02_convert_to_datastore.py (seconds)
03_register_and_deconvolve.py (minutes)
04_pixeldecode.py (minutes)
05_calculate_F1.py (seconds)

Note: decoding parameters have not been optimized, so the F1 score can be improved.

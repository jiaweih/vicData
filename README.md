# vicData
Function
---------
1. Process VIC output files and calculate annual maximum values of precipitation and run_base (runoff + baseflow)
2. Calculate annual mean precipitation
3. Find dates for each annual maximum events

Command
---------
python vicData.py -i input_dir -o output_dir_data -prec output_dir_dates_prec -run_base output_dir_dates_run_base

Parameters
---------
input_dir: input directories containing files to be processed (all files will be processed)

output_dir_data: output data directory to store processed data

output_dir_dates_prec: output directory to store dates of annual maximum precipitation

output_dir_dates_run_base: output directory to store dates of annual maximum run_base

Returns
---------
Output data files contain four columns: year, annuam max prec, annual max run_base, annual mean prec

Output date files contain four columns: year, month, day, maximum annual data

Examples
---------
python vicData.py -i example_fluxes -o example_outputs_data -prec example_dates_prec -run_base example_dates_run_base

Note: if vicData.py is not under the same directory as input_dir and output_dir, absolute 
	  path of input_dir and output_dir will be required

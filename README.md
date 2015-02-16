# vicData
Function
---------
1. Process VIC output files and calculate annual maximum values of precipitation and run_base (runoff + baseflow)
2. Calculate annual mean precipitation
3. Find dates for each annual maximum events

Command
---------
`python vicData.py -i input_dir`

Parameters
---------
**input_dir**: input directories containing files to be processed (all files will be processed)

Returns
---------
Output data files contain four columns: year, annual max prec, annual max run_base, annual mean prec

Output date files contain four columns: year, month, day, maximum annual data

Example of usage
---------
`python vicData.py -i example_fluxes`

Outputs will be in following directories: example_outputs_data, example_dates_prec, example_dates_run_base

Note: if **vicData.py** is not under the same directory as input_dir and output_dir, absolute 
	  path of input_dir and output_dir will be required

# vicFitting
Function
----------
Fit distributions to annual maximum run_base (runoff + baseflow) based on L-moments; four distributions are considered: PE3, GUM, GEV, GNO

Calculate estimates of annual maximum run_base based on certain return_periods

Command 
----------
`python vicFitting.py -i in_dir`

Inputs
----------
**in_dir**: input directory containing the fluxes files for each cell

Outputs
----------
**figures**: "figures" directory will be created under current directory, figures of fitting distributions for each cell will be generated in pdf format

**estimates**: "estimates" directory will be created under current directory, estimates of run_base based on certain return periods for each distribution will be calculated for each cell

Example of usage
-----------------
`python vicFitting.py -i example_fluxes`

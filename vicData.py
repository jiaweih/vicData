#!/usr/bin/env python
"""
	Function
	---------
	Process VIC output files and generate annual maximum values of precipitation and run_base (runoff + baseflow)

	Command
	---------
	python vicData.py -i input_dir -o output_dir

	Parameters
	---------
	input_dir: input directories containing files to be processed (all files will be processed)
	output_idr: output directories to store processed output files

	Examples
	---------
	python vicData.py -i example_fluxes -o example_outputs
	Note: if vicData.py is not under the same directory as input_dir and output_dir, absolute 
		  path of input_dir and output_dir will be required
"""


import numpy as np
import pandas as pd
import os
from datetime import datetime
import argparse


def main():
    in_dir,out_dir = command_lines()
    for filename in os.listdir(in_dir):
    	in_path = os.path.join(in_dir,filename)
        df = read_files(in_path)
        df_max = get_max(df)
        write_files(df_max,out_dir,filename)

def read_files(in_path,startYear=1916,endYear=2006,startIndex=274,endIndex=33511):
	"""
	Read data into DataFrame
	"""
    start = datetime(startYear,1,1)
    end = datetime(endYear,12,31)
    dates = pd.date_range(start=start,end=end)
    header = ['prec','runoff','baseflow','run_base']
    data = np.genfromtxt(in_path)
    start_pos = startIndex - 1
    end_pos = endIndex
    prec = data[start_pos:end_pos,3]
    runoff = data[start_pos:end_pos,12]
    baseflow = data[start_pos:end_pos,13]
    run_base = runoff + baseflow
    data_extract = np.transpose([prec,runoff,baseflow,run_base])
    df = pd.DataFrame(data_extract, index=dates, columns = header)
    return df

def get_max(df):
	"""
	Read in daily DataFrames and produce annual maximum DataFrames
	"""
    df_max = df.resample('A',how='max')
    return df_max

def write_files(df_max,out_dir,filename,startYear=1916,endYear=2006):
	"""
	Read in annual maximum DataFrames and save values in ascill files
	"""
    time = np.arange(startYear,endYear+1)
    prec = df_max['prec'].values
    run_base = df_max['run_base'].values
    write_arrays = np.transpose([time,prec,run_base])
    if not os.path.exists(out_dir):
    	os.makedirs(out_dir)
    out_path = os.path.join(out_dir,filename)
    np.savetxt(out_path,write_arrays,fmt=('%i','%8.3f','%8.3f'))

def command_lines():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir",
                        type=str,
                        help="Input directory",
                        required=True)
    parser.add_argument("-o", "--output_dir",
                        type=str,
                        help="Output directory",
                        required=True)
    args = parser.parse_args()
    return args.input_dir,args.output_dir

if __name__ == "__main__":
	main()
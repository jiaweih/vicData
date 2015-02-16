#/bin/bash
### convert the files to NetCDF 4
### original files are NetCDF 3 (classic) 

for file in ./data/*
do
	fn=`basename $file`
	nccopy -k netCDF-4 -d 4 $file ./data/$fn
done



#!/bin/bash
### regrid downloaded data to the grid and scope we want
### merge the 3-hourly wind data to daily mean

target_PLACE=/Volumes/HDTemp/jiawei/pacific/PLACE/BCSD+ANUSPLIN300+ACCESS1-0_historical_rcp45_pr.nc
target_UNSR=/Volumes/HDTemp/jiawei/pacific/UNSR/BCSD+ANUSPLIN300+ACCESS1-0_historical_rcp45_pr.nc
indir=/Volumes/HDTemp/jiawei/pacific/wind/download
outdir_PLACE=/Volumes/HDTemp/jiawei/pacific/wind/regridded/PLACE
outdir_UNSR=/Volumes/HDTemp/jiawei/pacific/wind/regridded/UNSR
for file in $indir/*
do
	echo $file
	file_name=`basename $file`
	cdo remapbil,$target_PLACE $file tmp.PLACE.nc
	cdo daymean tmp.PLACE.nc $outdir_PLACE/$file_name
	cdo remapbil,$target_UNSR $file tmp.UNSR.nc
	cdo daymean tmp.UNSR.nc $outdir_UNSR/$file_name
	rm tmp.PLACE.nc
	rm tmp.UNSR.nc
done

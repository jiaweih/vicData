#!/bin/bash

indir=/Volumes/HDTemp/jiawei/pacific/wind/regridded/UNSR
outdir=/Volumes/HDTemp/jiawei/pacific/wind/calculated/UNSR

for year in {1980..2009}
do
	echo $year
	uwind=$indir/uwnd.10m.$year.nc
	vwind=$indir/vwnd.10m.$year.nc
	wind=wind.10m.$year.nc
	cdo merge $uwind $vwind tmp.nc
	cdo expr,'wind=sqrt(uwnd*uwnd+vwnd*vwnd)' tmp.nc $outdir/$wind
	rm tmp.nc
done


ncrcat $outdir/wind.10m.????.nc /Volumes/HDTemp/jiawei/pacific/wind/concat/wind.UNSR.1980.2009.nc


#/bin/bash

#scenarios='rcp45 rcp85'
scenarios='rcp85'
vars='pr tasmax tasmin'

while read line
do
  model=$line
  echo $model
  for scenario in $scenarios
    do
       echo $scenario	
       for var in $vars
	 do
	    echo $var
 	    wget --output-document=./data/${model}_historical_${scenario}_${var}.nc --header "Cookie: beaker.session.id=6ef2c85b663a424ca9b605bc128c492f" http://tools.pacificclimate.org/dataportal/downscaled_gcms/data/pr+tasmax+tasmin_day_${model}_historical+${scenario}_r1i1p1_19500101-21001231.nc.nc?${var}[0:55152][98:169][182:305]&
	 done
    done
done < $1

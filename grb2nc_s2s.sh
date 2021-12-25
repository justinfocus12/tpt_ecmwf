# From https://confluence.ecmwf.int/display/LDAS/Converting+from+grib+to+netCDF+with+cdo
module load cdo/intel/1.9.10
datadir="/scratch/jf4241/ecmwf_data/s2s_data/2021-12-23"
echo $datadir
for grib_filename in $datadir/*.grb
do
    nc_filename="${grib_filename/grb/nc}"
    cdo -w -f nc copy $grib_filename $nc_filename 
    echo "converted to new filename ${nc_filename}"
done




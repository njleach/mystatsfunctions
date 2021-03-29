# This file downloads the gridded data used in the OLSE module examples
wget "ftp://ftp2.psl.noaa.gov/Datasets/20thC_ReanV3/Monthlies/miscSI-MO/prmsl.mon.mean.nc" -O "20CRv3_mslp.nc" &
wget "ftp://ftp2.psl.noaa.gov/Datasets/20thC_ReanV3/Monthlies/accumsSI-MO/apcp.mon.mean.nc" -O "20CRv3_prcp.nc" &
wget "ftp://ftp2.psl.noaa.gov/Datasets/20thC_ReanV3/timeInvariantSI/land.nc" -O "20CRv3_lsm.nc" &
wget "ftp://ftp2.psl.noaa.gov/Datasets/20thC_ReanV3/spreads/Monthlies/accumsSI-MO/apcp.mon.mean.nc" -O "20CRv2_prcp_spread.nc"

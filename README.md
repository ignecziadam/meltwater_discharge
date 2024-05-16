# Meltwater discharge data processing description
This document describes the content of the script files published in the following paper: A high-resolution, operational pan-Arctic meltwater discharge database from 1950 to 2021.  
doi: xxxxxxxxxx

The document also provides guidance for code execution.

Adam Igneczi, Jonathan Bamber (20/04/2024)

# Introduction
We provide Bash Shell and Python scripts which are used in our data processing algorithm. We also provide 2 Python scripts which were used for data comparisons and graph production. Here, we give guidance on the directory structure, that users need to create on the machine. Input data need to be placed into these folders prior to running any scripts (see below).

The Python scripts are called by the Bash Shell scripts (except for the 2 data comparison and visualisation scripts). Various Anaconda environments are called by these Bash Shells (hard-coded in the Bash scripts). Software environments are installed on a Linux machine, see the instructions below to set them up. Scripts are discussed in the order they need to be executed. Users only need to manually run the Bash Shell scripts, the appropriate Python scripts are called be these (except for the 2 data comparison and visualisation scripts). 

Bash Shell scripts need positional arguments, appropriate guidance is provided here and in comments within the script files. Bash Shell scripts also have hard-coded environmental variables, e.g. main data and results directories. These might need to be changed depending on the directory structure the user has set up. Please see guidance here and in the comments within the script files.

# Software environments  
Conda environment for the MAR transformations  
> conda config --add conda-forge  
> conda config --set channel_priority strict  

Conda environment for data preprocessing  
> conda create -n gdal_env gdal
> conda create -n geotransform_env rioxarray xarray dask netCDF4 bottleneck geopandas openpyxl  

Conda environment for routing operations (might not work on certain Linux versions)   
> conda create -n routing_env whitebox geopandas rioxarray xarray netCDF4  

First time use, initialise Whitebox (remain in the terminal window)  
> activate routing_env  
> python  
> import whitebox  
> wbt = whitebox.WhiteboxTools()  

Test that download worked  
> print(wbt.version())  

Whitebox is ready to use.  

# Directory structure and data prep
Create these directories and place data in the directories as described here prior to running any scripts.

- DATA_DIR  
  	> This contains the input data  
	- DATA_DIR/COP_DEM  
		> Place the raw Copernicus_DEM download here (we used the Copernicus GLO-90 DGED DEM product). Retain original naming of the folders and products.
	- DATA_DIR/MAR
		> Place the annual MAR netCDF files here (or files from another RCM).  
  		> NOTE: As there are many versions of MAR and other RCMs (different model versions, variables, coverage, files, CRS, etc.) users might need to modify the scripts that deal with MAR pre-processing (e.g.: preprocess_mar.py).
	
- RESULTS_DIR  
  	> This will contain the output data, and some region specific data inputs  
	- RESULTS_DIR/RGI_x_xxxxxx  
		> Folder for investigated RGI regions, create this for each processed RGI region.  
	- RESULTS_DIR/RGI_x_xxxxxx/IceMask  
		> Place ice mask shapefiles here (downloaded from the Randolph Glacier Inventory). In the case of Greenland, the GIMP ice mask raster is also placed here (the name of this raster is hardcoded to create_icemask.sh).
	- RESULTS_DIR/RGI_x_xxxxxx/selectors  
		- RGI_x_xxxxxx_TileList.csv  
			> Place a csv containing the list of Copernicus_DEM tiles for the region. These lists can be constructed by intersecting the Copernicus DEM tile grid shapefile and the RGI region domain, and exporting the selected records from the attribute table (in QGIS, or ArcGIS).
		- RGI_x_xxxxxx_cutline_laeaa.shp  
			> Place a shapefile here that contains the the RGI first order region outline (downloaded from the Randolph Glacier Inventory).
		
Other subdirectories are created automatically by the scripts, as discussed below.

- HOME_DIR  
  	> Batch Shell scripts need to be placed in a home directory (this can be anywhere).
	- HOME_DIR/scripts_python  
		> Place all the provided Python scirpts here.  
  		> NOTE: this directory needs to be called "scripts_python"

# Batch Shell and Python scripts
In this section we briefly explain what each script is doing. We structure this section according to the sequence the scripts need to run. Batch Shell scripts and the corresponding Python scripts are explained together

## 1. Select and move DEM and LandMask tiles into the appropriate regional folder
The process need to be called twice, once for DEM and once for the LandMask. The script need to be executed (twice) for each investigated RGI region separately. Appropriate subfolders are automatically created.

Script : wrangle_files.sh (calls wrangle_files.py)

Hard-coded environmental variables (might need changing)  
- INPUT_DIR  
	> The directory where the Copernicus DEM data is located, e.g.: DATA_DIR/COP_DEM
- OUTPUT_DIR  
 	> The main results directory, e.g.: RESULTS_DIR
	
Positional arguments (need to be declared when running the script)
- $1 REGION [RGI_no_name]  
	> Name of the RGI region folder in the RESULTS_DIR, e.g.: RESULTS_DIR/RGI_x_xxxxxx.
- $2 PRODTYPE [DEM/LandMask]  
	> The type of data to move, LandMask or DEM.

## 2. Create DEM and LandMask mosaics
Reprojectes and mosaics the DEM and LandMask tiles, placed in the appropriate folders by the previous process. The script need to be executed for each investigated RGI region separately. Appropriate subfolders are automatically created. Intermediate products are automatically deleted.

Script : create_dem_and_landmask.sh

Hard-coded environmental variables (might need changing)
- DATADIR  
	> The main results directory, e.g.: RESULTS_DIR

Positional arguments (need to be declared when running the script)  
- $1 REGION [RGI_no_name]  
	> Name of the RGI region folder in the RESULTS_DIR, e.g.: RESULTS_DIR/RGI_x_xxxxxx.
	
## 3. Create IceMask mosaics
Converts ice mask shapefiles to ice mask rasters, reprojects and snaps them to the appropriate COP-DEM mosaic. In the case of Greenland, it reprojects, resamples, and snaps the GIMP mask to the appropriate COP-DEM mosaic. The script need to be executed for each investigated RGI region separately. Appropriate subfolders are automatically created. Intermediate products are automatically deleted.

Script : create_icemask.sh (calls get_bounding_box.py)

Hard-coded environmental variables (might need changing)  
- DATADIR  
	> The main results directory, e.g.: RESULTS_DIR

Positional arguments (need to be declared when running the script)  
- $1 REGION [RGI_no_name]  
	> Name of the RGI region folder in the RESULTS_DIR, e.g.: RESULTS_DIR/RGI_x_xxxxxx.

## 4. Pre-process MAR (or another RCM)
Selects and filters appropriate variables from raw MAR products. Reprojects these variables, and clips them appropriately. Calculates vertical gradients. It also exports the domain of valid MAR data for use in subsequent processes. Saves these outputs to annual netCDF files. The script need to be executed for each investigated RGI region separately. Appropriate subfolders are automatically created. Intermediate products are automatically deleted.

NOTE: Many parameters and arguments are hard-coded into "preprocess_mar.py" (e.g. naming conventions, CRS info). If using these scripts on another MAR or other RCM product, pleaser review "preprocess_mar.py" carefully. The parallel processing setup also relies on the fact that this version of MAR is supplied in annual netCDF files.
Appropriate pre-processing of MAR/other RCM files (e.g. slicing iunto yearly products) might be needed to utilise the option of parallel processing.

Script : preprocess_mar.sh (calls preprocess_mar.py)

Hard-coded environmental variables (might need changing)  
- INPUT_DIR  
	> The directory where the MAR data is located, e.g.: DATA_DIR/MAR
- OUTPUT_DIR  
	> The main results directory, e.g.: RESULTS_DIR

Positional arguments (need to be declared when running the script)  
- $1 REGION [RGI_no_name]
	> Name of the RGI region folder in the RESULTS_DIR, e.g.: RESULTS_DIR/RGI_x_xxxxxx.
- $2 CORE_Number [No]  
	> The number of CPUs cores used in the parallel processing pool.
	
## 5. Delinate drainage basins
Uses hydrological routing tools to delinate surface drainage basins. Carries out the appropriate filtering steps (using hard-coded parameters). Saves the derived basins into GeoTiff and the outflow points into a ShapeFile. The script need to be executed for each investigated RGI region separately. Appropriate subfolders are automatically created. Intermediate products are automatically deleted.

NOTE: As Whitebox was not compatible with our Linux instance, this script has been executed on our Windwos desktops. Thus there is no Bash Shell written for this process.

Conda environment: routing_env
Script : create_basins.py (run from terminal as "python create_basins.py RGI_x_xxx")

Hard-coded environmental variables (might need changing)  
- WorkingDIR  
	> The main results directory, e.g.: RESULTS_DIR
- MinBasinArea  
	> Size of the smallest basin possible (km^2), basins below this size are merged with their larger neighbours.  
- Resolution  
	> Spatial resolution of the basin product (m).  
- EdgeNoRGI  
	> Remove basin if the number of pixels touching RGI domain edges is larger than this number (integer no.).  
- RelBasinReduction  
	> Remove basins that extend outside the valid MAR domain. This % is the largest fractional area outside the MAR domain that is allowed (%).

Positional arguments (need to be declared when running the script)  
- $1 CurrentDomainSTR [RGI_no_name]  
	> Name of the RGI region folder in the RESULTS_DIR, e.g.: RESULTS_DIR/RGI_x_xxxxxx.

## 6. Downscale MAR products and integrate runoff over the drainage basins
Upsamples ice runoff, ice albedo, and land runoff; corrects for high-resolution ice masks. Downscales these products, by using elevation correction. Sums runoff for drainage basins, separates ice runoff where the ice albedo is below 0.7 (i.e. appropx. below the snowline). Saves the basin specific runoff products in annual netCDF files, and provides daily bulk runoff time-series for the region. The script need to be executed for each investigated RGI region separately. Appropriate subfolders are automatically created. Intermediate products are automatically deleted.

Script : downscale_and_integrate.sh (calls downscale_and_integrate.py)

Hard-coded environmental variables (might need changing)  
- WORKING_DIR  
	> The main results directory, e.g.: RESULTS_DIR

Positional arguments (need to be declared when running the script)  
- $1 REGION [RGI_no_name]  
	> Name of the RGI region folder in the RESULTS_DIR, e.g.: RESULTS_DIR/RGI_x_xxxxxx.  
- $2 CORE_Number [No]  
	> The number of CPUs cores used in the parallel processing pool.

## 7. Concatenate annual files
Grabs annual netCDF files, containing basin specific daily runoff, and concatenates these to a single file covering the period of 1950-2021. The script need to be executed for each investigated RGI region separately.

Script : concatenate_annual_results.sh (calls concatenate_annual_results.py)

Hard-coded environmental variables (might need changing)  
- WORKING_DIR  
	> The main results directory, e.g.: RESULTS_DIR

Positional arguments (need to be declared when running the script)  
- $1 REGION [RGI_no_name]  
	> Name of the RGI region folder in the RESULTS_DIR, e.g.: RESULTS_DIR/RGI_x_xxxxxx.

## 8. Create pan-Arctic summary dataset
Grabs regional basin specific runoff files created in the previous step, calculates monthly sums from the daily data. Concatenates these datasets into a single pan-Arctic file.

Script : summary_results.sh (calls summary_results.py)

Hard-coded environmental variables (might need changing)  
- WORKING_DIR  
	> The main results directory, e.g.: RESULTS_DIR

Hard-coded environmental variables in the Python script  
- regions  
	> List of strings, the strings match up with the names of the processed RGI regions (RGI_x_xxxxxx).
		
## 9-10. Dataset comparison, graph plotting
Plots basin specific runoff, with running means. Compares out dataset with Bamber et al. (2018) and Mankoff et al. (2020).

Scripts  
- plot_local_runoff.py  
	> Plots runoff for a basin, for the whole period covered by the dataset and a single year. Variables identifying the RGI domain and the specific basin, and the start and end years are hard-coded.

- plot_comparisons.py  
	> Compares our dataset with  Bamber et al. (2018) and Mankoff et al. (2020).
	Plots graphs showing the comparison.
	Saves Excel spreadsheets with the data.

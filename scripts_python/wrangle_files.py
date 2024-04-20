# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 22:58:07 2022

@author: sv21669
"""

import csv
import os
import shutil
import sys

InputRegionSTR = sys.argv[1]
ProdTypeSTR = sys.argv[2]

InDIR = os.getenv("INPUT_DIR")  # "/scratch/atlantis2/AP_data/COP_DEM"
OutDIR = os.getenv("OUTPUT_DIR")  # "/scratch/atlantis2/AP_results"

#
OutRegionDIR = os.path.join(OutDIR, InputRegionSTR)
OutProductDIR = os.path.join(OutDIR, InputRegionSTR, ProdTypeSTR)

if os.path.lexists(OutProductDIR) == False:
    os.mkdir(OutProductDIR)
    
if len(os.listdir(OutProductDIR)) > 0:
    print("Folder already processed", flush=True)
    sys.exit()

else:   
    with open(os.path.join(OutRegionDIR, "selectors", InputRegionSTR + "_TileList.csv"), newline="") as File:
        Reader = csv.reader(File)
        TileList = []

        for Row in Reader:
            TileList.append(Row[0])

    Counter = 0

    for Idx, Tile in enumerate(TileList):
        Tile = Tile.split('_')
        Tile.pop(2)
        Tile.pop(-1)
        Tile = '_'.join(Tile)
        
        if ProdTypeSTR == 'DEM':
            SourceFile = os.path.join(InDIR, Tile, 'DEM', Tile + '_DEM.tif')
        else:
            SourceFile = os.path.join(InDIR, Tile, 'AUXFILES', Tile + '_WBM.tif')

        if os.path.lexists(SourceFile) == False:
            print('Skipping ' + str(Idx + 1) + 'of' + str(len(TileList)), flush=True)
            continue
        
        else:
            print('Copying ' + str(Idx + 1) + 'of' + str(len(TileList)), flush=True)
            shutil.copy(SourceFile, OutProductDIR)
            Counter = Counter + 1
            
print('Finished copying ' + str(Counter) + ' files', flush=True)
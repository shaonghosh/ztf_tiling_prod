import sys

import numpy as np
import pandas as pd

import realisticRankedTilesGenerator
from realisticRankedTilesGenerator import RankedTileGenerator


def getProbLost(skymap, failed_obs_file):
    obs_data = pd.read_csv(failed_obs_file, delimiter=',',
                           names=['ID', 'filter', 'fieldID', 'chipID', 'quadID'],
                           skiprows=1)

    obs_data.fieldID = obs_data.fieldID.astype(int)
    obs_data.chipID = obs_data.chipID.astype(int)
    obs_data.quadID = obs_data.quadID.astype(int)
    obsUnique = np.unique(np.vstack((obs_data.fieldID.values,
                                     obs_data.chipID.values,
                                     obs_data.quadID.values)).T, axis=0)
    obsUnique_df = pd.DataFrame(obsUnique, columns=['fieldID', 'chipID', 'quadID'])
    totalProbLost = 0.0
    totalProbLost_corrected = 0.0
    totalQuadArea = 0.0
    tileObj = RankedTileGenerator(tile_pix_map='fixed_preComputed_256_set_1_python3.dat')
    Ranked_Tiles = tileObj.getRankedTiles(fitsfilename=skymap)

    for fieldID, chipID, quadID in zip(obsUnique_df.fieldID.values,
                                       obsUnique_df.chipID.values,
                                       obsUnique_df.quadID.values):
        thisTileProb = Ranked_Tiles[Ranked_Tiles['tile_index'] == fieldID].tile_prob.values[0]
        skymapData = realisticRankedTilesGenerator.getSkymapData(skymap)
        quadProbObj = realisticRankedTilesGenerator.QuadProb()
        thisTileProb_fromQuads = 0.0
        correction = 1.0
        probLost = quadProbObj.getQuadProb(skymapData, fieldID, chipID, quadID)
        probLost_corrected = correction*probLost
        #totalQuadArea += (quadProbObj.num_pix_in_quad)*pixArea
        totalProbLost += probLost
        totalProbLost_corrected += probLost_corrected

    #return [totalProbLost, totalQuadArea]
    return [totalProbLost, totalProbLost_corrected, Ranked_Tiles]


def getNumPix_inTile(tile_rank, Ranked_Tiles, skymapfile):
    Q = realisticRankedTilesGenerator.QuadProb()
    tile_ID = int(Ranked_Tiles.tile_index.values[tile_rank])
    skymapdata = realisticRankedTilesGenerator.getSkymapData(skymapfile)
    pix_in_tile_fromquad = 0
    prob_in_tile = 0
    prob_in_tile_fromquad = 0
    for chipN in range(1, 17):
        for quadN in range(1, 5):
            quadProb = Q.getQuadProb(skymapdata, tile_ID, chipN, quadN)
            prob_in_tile_fromquad += quadProb
            pix_in_quad = Q.num_pix_in_quad
            pix_in_tile_fromquad += pix_in_quad
    pixels_in_tile = tileData[tile_ID][0] ### Computed directly from the tile-data
    prob_in_tile = rt.tile_prob.values[tile_rank]
    return (len(pixels_in_tile), pix_in_tile_fromquad, prob_in_tile, prob_in_tile_fromquad)

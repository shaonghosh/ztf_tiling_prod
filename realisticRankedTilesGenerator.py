# Copyright (C) 2018 Shaon Ghosh
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

import os
import sys
from operator import itemgetter

import numpy as np
import pandas as pd
import healpy as hp
import pickle

from astropy.utils.console import ProgressBar
from astropy import constants as c, units as u
from astropy.wcs import WCS

import ZTF_tiling_realCCD


def getSkymapData(skymapFilename, res=256):
    healpix_skymap = hp.read_map(skymapFilename, verbose=False)
    skymapUD = hp.ud_grade(healpix_skymap, res, power=-2)
    healpix_skymap = skymapUD
    npix = len(healpix_skymap)
    nside = hp.npix2nside(npix)
    theta, phi = hp.pix2ang(nside, np.arange(0, npix))
    ra_pixels = np.rad2deg(phi)
    dec_pixels = np.rad2deg(0.5*np.pi - theta)
    data_pixels = healpix_skymap[np.arange(0, npix)]
    skymapdata = [np.arange(npix), ra_pixels, dec_pixels, data_pixels]
    return skymapdata


class QuadProb:
    '''
    Class :: Instantiate a QuadProb object that will allow us to calculate the
             probability content in a single quadrant.
    '''
    def __init__(self, tileFile='ztf_fields_1.txt'):
        ztf_fieldsdata = np.recfromtxt('ztf_fields_1.txt', names=True)
        ID = ztf_fieldsdata['ID']
        self.RA = ztf_fieldsdata['RA']
        self.Dec = ztf_fieldsdata['Dec']
        self.ID = ID.astype(int)
        self.quadMap = {1: [0, 1, 2, 3], 2: [4, 5, 6, 7],
                        3: [8, 9, 10, 11], 4: [12, 13, 14, 15],
                        5: [16, 17, 18, 19], 6: [20, 21, 22, 23],
                        7: [24, 25, 26, 27], 8: [28, 29, 30, 31],
                        9: [32, 33, 34, 35], 10: [36, 37, 38, 39],
                        11: [40, 41, 42, 43], 12: [44, 45, 46, 47],
                        13: [48, 49, 50, 51], 14: [52, 53, 54, 55],
                        15: [56, 57, 58, 59], 16: [60, 61, 62, 63]}
        self.quadrant_size = np.array([3072, 3080])
        self.quadrant_scale = 1*u.arcsec

    def getWCS(self, quad_cents_RA, quad_cents_Dec):
        _wcs = WCS(naxis=2)
        _wcs.wcs.crpix = (self.quadrant_size + 1.0)/2
        _wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
        wrapped_quad_cent_Dec = max(min(quad_cents_Dec.value, 90), -90)
        _wcs.wcs.crval = [quad_cents_RA.value, wrapped_quad_cent_Dec]
        _wcs.wcs.cd = [[self.quadrant_scale.to(u.deg).value, 0],
                       [0, -self.quadrant_scale.to(u.deg).value]]
        return _wcs

    def getQuadProb(self, skymapdata, fieldID, chipN, quadN):
        '''Method  :: Takes as input the skymap data (return of function
                      getSkymapData), the fieldID the chip number and the
                      quadrant number and returns the
                      probability contained in that quadrant.

        skymapdata :: This is the return of the function getSkymapData
        fieldID    :: The integer value of the first column of the tile
                      file (ztf_fields_1)
        chipN       :: The chip number (1 - 16)
        quadN	   :: The quadrant number (1 - 4)
        '''

        RA_tile_cent = self.RA[self.ID == fieldID][0]
        Dec_tile_cent = self.Dec[self.ID == fieldID][0]
        quadIndex = self.quadMap[chipN][quadN-1]
        Z = ZTF_tiling_realCCD.ZTFtile(RA_tile_cent, Dec_tile_cent)
        quad_cents_RA, quad_cents_Dec = Z.quadrant_centers()
        # print('quadIndex = {}'.format(quadIndex))
        thisQuad_wcs = self.getWCS(quad_cents_RA[quadIndex],
                                   quad_cents_Dec[quadIndex])
        footprint = thisQuad_wcs.calc_footprint(axes=self.quadrant_size)
        RAs = skymapdata[1]
        Decs = skymapdata[2]
        pVals = skymapdata[3]
        points = np.vstack((RAs, Decs)).T
        inside = ZTF_tiling_realCCD.inside(footprint[:, 0], footprint[:, 1],
                                           points[:, 0], points[:, 1])
        self.num_pix_in_quad = np.sum(inside)
        quadProb = np.sum(pVals[inside])
        return quadProb


class RankedTileGenerator:
    def __init__(self, tile_pix_map='fixed_preComputed_256_set_1_python3.dat'):
        File = open(tile_pix_map, 'rb')
        self.tileData = pickle.load(File)
        self.IDs = np.array(list(self.tileData.keys()))
#         self.res = res
#         if fitsfilename:
#             self.skymap = getSkymapData(fitsfilename, res=res)
#         elif skymapdata:
#             self.skymap = [skymapdata[0], skymapdata[1],
#                            skymapdata[2], skymapdata[3]]
#
#         else:
#             print('Either skymap filename need to be provided')
#             print('or, skymap data needs to be supplied')
#             sys.exit(1)

    def getRankedTiles(self, verbose=True, fitsfilename=None,
                       skymapdata=False, res=256):
        if fitsfilename:
            skymap = getSkymapData(fitsfilename, res=res)
        elif skymapdata:
            skymap = [skymapdata[0], skymapdata[1],
                      skymapdata[2], skymapdata[3]]

        self.pixel_id_all = skymap[0]
        self.point_ra_all = skymap[1]
        self.point_dec_all = skymap[2]
        self.point_pVal_all = skymap[3]

        pvalTile = []
        TileProbSum = 0.0
        if verbose:
            print('Computing Ranked-Tiles...')

        x = list(self.tileData.keys())
        z = itemgetter(*x)(self.tileData)
        if verbose:
            with ProgressBar(len(z)) as bar:
                for ii in z:
                    pvalTile.append(np.sum(skymap[-1][ii[0]]))
                    bar.update()
        else:
            for ii in z:
                pvalTile.append(np.sum(skymap[-1][ii[0]]))

        pvalTile = np.array(pvalTile)
        sorted_indices = np.argsort(-1*pvalTile)
        output = np.vstack((self.IDs[sorted_indices],
                            pvalTile[sorted_indices])).T
        df = pd.DataFrame(output, columns=['tile_index', 'tile_prob'])
        return df


class ChooseTilesFromTwoSets(RankedTileGenerator):
    prefix = 'fixed_preComputed_256_set_'
    suffix = '_python3.dat'

    def __init__(self,
                 tile_pix_map1='{}{}{}'.format(prefix, 1, suffix),
                 tile_pix_map2='{}{}{}'.format(prefix, 2, suffix)):
        filename1 = tile_pix_map1
        filename2 = tile_pix_map2
        File1 = open(filename1, 'rb')
        self.tileData1 = pickle.load(File1)
        File2 = open(filename2, 'rb')
        self.tileData2 = pickle.load(File2)
        self.thisTileObj1 = RankedTileGenerator(filename1)
        self.thisTileObj2 = RankedTileGenerator(filename2)

    def chooseTopTile(self, skymap, verbose=False):
        '''
        This function takes as input the list of ranked tiles and a sky map.
        The function then returns the top ranked tile index and a modified sky
        map.The modified sky-map has the probability value of all the pixels
        within this top ranked tile set to zero.
        '''

        df1 = self.thisTileObj1.getRankedTiles(skymapdata=skymap, res=256,
                                               verbose=False)
        df2 = self.thisTileObj2.getRankedTiles(skymapdata=skymap, res=256,
                                               verbose=False)
        df1 = df1.sort_values('tile_prob', ascending=False)
        df2 = df2.sort_values('tile_prob', ascending=False)
        if verbose:
            print(df1.head())
            print(df2.head())
            print('Probability value of top tile in first set = {}'.
                  format(df1.tile_prob.values[0]))
            print('Probability value of top tile in second set = {}'.
                  format(df2.tile_prob.values[0]))
        if df1.tile_prob.values[0] < df2.tile_prob.values[0]:
            pixelIndices = self.tileData2[int(df2.tile_index[0])][0]
            chosenTile = int(df2.tile_index[0])
            if verbose:
                print('Chosen from set 2')
        else:
            pixelIndices = self.tileData1[int(df1.tile_index[0])][0]
            chosenTile = int(df1.tile_index[0])
            if verbose:
                print('Chosen from set 1')
        insideTopTile = np.isin(skymap[0], pixelIndices)
        thisTileProb = np.sum(skymap[3][insideTopTile])
        skymap[3][insideTopTile] = 0.0

        return [skymap, chosenTile, thisTileProb]

    def superTiles(self, skymap, target=False, Num=False):
        '''This function takes as input the data from the skymap and a target
        localization fraction, or a number of tiles, and returns the dataframe
        of ranked tiles obtained from the combined set of tiles.

        skymap = skymap data [pixel_index, ra, dec, prob]

        target = Target probability, must be smaller than the total probability
        enclosed in all the tiles.

        Num = Number of tiles
        One and only one of the two options target and Num will have to be
        used'''
        if not (bool(target) ^ bool(Num)):
            print('One and only one option, target or Num, must be supplied')
            sys.exit(1)
        if target:
            probCovered = 0
            superTiles = []
            tileProbs = []
            tile_set_num = []
            while probCovered < target:
                [skymapdata,
                 chosenTile,
                 thisTileProb] = self.chooseTopTile(skymap)
                superTiles.append(chosenTile)
                tileProbs.append(thisTileProb)
                if int(chosenTile) <= 879:
                    tile_set_num.append(1)
                else:
                    tile_set_num.append(2)
                probCovered += thisTileProb

        elif Num:
            superTiles = []
            tileProbs = []
            tile_set_num = []
            for ii in range(Num):
                [skymapdata,
                 chosenTile,
                 thisTileProb] = self.chooseTopTile(skymap)
                superTiles.append(chosenTile)
                tileProbs.append(thisTileProb)
                if int(chosenTile) <= 879:
                    tile_set_num.append(1)
                else:
                    tile_set_num.append(2)

        output = np.vstack((tile_set_num, superTiles, tileProbs)).T
        df = pd.DataFrame(output, columns=['Set', 'Tile Index', 'Probability'])

        return df


class Observationscore:
    """
    CLASS :: Analysis of observation conducted by ZTF.
    """
    def __init__(self, tileObj, skymap, tileData,
                 obs_file='failed_quadrant_sample.txt'):
        """
        INPUT
        tileObj  :: A tile object created by RankedTileGeneratorself.
        skymap   :: The name of the skymap fileself.
        tileData :: The contents of the tile-pixel map. This is an object
                    defined in tileObj.
        obs_file :: The observation data from ZTF.
        """

        self.tileObj = tileObj
        self.Ranked_Tiles = self.tileObj.getRankedTiles(fitsfilename=skymap)
        self.skymapData = getSkymapData(skymap)
        self.quadProbObj = QuadProb()
        self.tileData = tileData
        self.obs_data = pd.read_csv(obs_file, delimiter=',',
                                    names=['ID', 'filter', 'fieldID',
                                           'chipID', 'quadID'],
                                    skiprows=1)

    def getProbLost(self, noCorrectionNeeded=False):
        """
        METHOD  ::  Computes lost probability and lost coverage area due to
                    failed quadrants.
        INPUT
        noCorrectionNeeded :: Does not perform correction in the quads

        NOTE: Currently the correction is required in each tile. Note that the
              post correction results are correct in a statistical sense.
              As soon as an accurate computation is done this will be included
              and the noCorrectionNeeded argument could be set to True.
        """

        pixArea = hp.nside2pixarea(hp.npix2nside(self.skymapData[0][-1]+1),
                                   degrees=True)

        self.obs_data.fieldID = self.obs_data.fieldID.astype(int)
        self.obs_data.chipID = self.obs_data.chipID.astype(int)
        self.obs_data.quadID = self.obs_data.quadID.astype(int)
        obsUnique = np.unique(np.vstack((self.obs_data.fieldID,
                                         self.obs_data.chipID,
                                         self.obs_data.quadID)).T, axis=0)
        obsUnique_df = pd.DataFrame(obsUnique, columns=['fieldID', 'chipID',
                                                        'quadID'])
        totPLost = 0.0
        totPLost_corrected = 0.0
        totalQArea = 0.0

        # Compute the correction factor for the unique tiles #
        unique_tiles = np.unique(obsUnique_df.fieldID.values)
        if noCorrectionNeeded:
            values = np.ones(len(unique_tiles)*2, dtype=int)
            value_vects = values.reshape(len(unique_tiles), 2)
            corr_dict = dict(zip(unique_tiles, value_vects))
        else:
            corr_dict = {}
            for tile in unique_tiles:
                pix_tile, pix_from_quads,\
                prob_in_tile, prob_from_quads = self.num_pixand_prob_intile(tile)
                factor_area = pix_tile/pix_from_quads
                factor_prob = prob_in_tile/prob_from_quads
                corr_dict[tile] = [factor_area, factor_prob]

        for fieldID, chipID, quadID in zip(obsUnique_df.fieldID.values,
                                           obsUnique_df.chipID.values,
                                           obsUnique_df.quadID.values):
            probLost = self.quadProbObj.getQuadProb(self.skymapData, fieldID,
                                                    chipID, quadID)
            probLost_corrected = probLost * corr_dict[fieldID][1]
            totalQArea += (self.quadProbObj.num_pix_in_quad)*pixArea
            totPLost += probLost
            totPLost_corrected += probLost_corrected
            totalQArea_corrected = totalQArea * corr_dict[fieldID][0]

        return [totPLost, totPLost_corrected, totalQArea, totalQArea_corrected]

    def num_pixand_prob_intile(self, tile_ID):
        pix_in_tile_fromquad = 0
        p_in_tile = 0
        p_in_tile_fromquad = 0
        for chipN in range(1, 17):
            for quadN in range(1, 5):
                quadProb = self.quadProbObj.getQuadProb(self.skymapData,
                                                        tile_ID, chipN, quadN)
                p_in_tile_fromquad += quadProb
                pix_in_quad = self.quadProbObj.num_pix_in_quad
                pix_in_tile_fromquad += pix_in_quad
        pixels_in_tile = self.tileData[tile_ID][0]  # From the tile-data
        p_tile = self.Ranked_Tiles.loc[self.Ranked_Tiles.tile_index == tile_ID]
        pVal_in_tile = p_tile.tile_prob.values[0]
        return_val = (len(pixels_in_tile), pix_in_tile_fromquad,
                      pVal_in_tile, p_in_tile_fromquad)
        return return_val

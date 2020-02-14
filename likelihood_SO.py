import IPython
from astropy import units as u
import numpy as np
# import matplotlib.pyplot as plt
# import plot_project as plotpro
import lib_project as lib

# from pandas import DataFrame as df
# import pandas as pd
# import copy
# from matplotlib.cm import get_cmap
# import matplotlib.text as mtext
# import matplotlib
from fgbuster.observation_helpers import get_sky
import pysm
import V3calc as V3


class sky_map:

    cmb_freq = 200
    dust_freq = 150
    synchrotron_freq = 20

    def __init__(self, nside=128, instrument='SAT', sky_model='c1s0d0',
                 bir_angle=0*u.deg, cal_angle=0*u.deg):
        self._nside = nside
        self._instrument = instrument
        self._sky_model = sky_model
        self._bir_angle = bir_angle
        self._cal_angle = cal_angle

    def _get_nside(self):
        return self._nside

    def _get_instrument(self):
        return self._instrument

    def _get_sky_model(self):
        return self._sky_model

    def _get_bir_angle(self):
        return self._bir_angle

    def _get_cal_angle(self):
        return self._cal_angle

    nside = property(_get_nside)
    instrument = property(_get_instrument)
    sky_model = property(_get_sky_model)
    bir_angle = property(_get_bir_angle)
    cal_angle = property(_get_cal_angle)

    def get_pysm_sky(self):
        sky = pysm.Sky(get_sky(self.nside, self.sky_model))
        self.sky = sky

    def get_frequency(self):
        if self.instrument == 'SAT':
            print(self.instrument)
            self.frequencies = V3.so_V3_SA_bands()
        if self.instrument == 'LAT':
            print(self.instrument)
            self.frequencies = V3.so_V3_LA_beams()

    # def get_freq_maps(self, output=0):
    #     cmb_freq_maps = self.sky.cmb(self.frequencies)
    #     fg_freq_maps = self.sky.dust(self.frequencies) +\
    #         self.sky.synchrotron(self.frequencies)
    #     self.cmb_freq_maps = cmb_freq_maps
    #     self.fg_freq_maps = fg_freq_maps
    #     if output:
    #         return cmb_freq_maps, fg_freq_maps

    def get_freq_maps(self, output=0):
        cmb_freq_maps = self.sky.cmb(sky_map.cmb_freq) * \
            pysm.convert_units('K_RJ', 'K_CMB', sky_map.cmb_freq)
        dust_freq_maps = self.sky.dust(sky_map.dust_freq) * \
            pysm.convert_units('K_RJ', 'K_CMB', sky_map.dust_freq)
        sync_freq_maps = self.sky.synchrotron(sky_map.synchrotron_freq) *\
            pysm.convert_units('K_RJ', 'K_CMB', sky_map.synchrotron_freq)
        self.cmb_freq_maps = cmb_freq_maps
        self.dust_freq_maps = dust_freq_maps
        self.sync_freq_maps = sync_freq_maps
        if output:
            return cmb_freq_maps, dust_freq_maps, sync_freq_maps

    def cmb_rotation(self, output=0):
        cmb_freq_rot = lib.map_rotation(self.cmb_freq_maps, self.bir_angle)
        self.cmb_freq_rot = cmb_freq_rot
        if output:
            return cmb_freq_rot

    def faraday_rotation(self, output=0):
        cmb_map_array = np.array([self.cmb_freq_maps/len(self.frequencies) for
                                  i in range(len(self.frequencies))])
        cmb_faraday = np.sum(lib.map_rotation(cmb_map_array,
                                              self.bir_angle), 0)
        self.cmb_faraday = cmb_faraday
        if output:
            return cmb_faraday

    def get_signal(self, output=0):
        try:
            signal = self.cmb_freq_rot + self.fg_freq_maps
        except AttributeError:
            signal = self.cmb_freq_maps + self.fg_freq_maps
        self.signal = signal
        if output:
            return signal


class Personne:
    compteur = 0

    def __init__(self, nom, prenom='Jean'):
        self.nom = nom
        self.prenom = prenom
        self.age = 24
        self._residence = 'Berlin'
        Personne.compteur += 1

    def _get_residence(self):
        print('On accede a mon lieu de residence !')
        return self._residence

    def _set_residence(self, nouvelle_residence):
        print('{} va demenager a {}'.format(self.prenom, nouvelle_residence))
        self._residence = nouvelle_residence

    residence = property()


class TableauNoir:
    compteur = 0

    def __init__(self):
        self.surface = ''
        TableauNoir.compteur += 1

    def ecrire(self, message):
        if self.surface != '':
            self.surface += '\n'
        self.surface += message

    def effacer(self):
        self.surface = ''

    def afficher(self):
        print(self.surface)

    def combien(cls):
        print('On a cree {} objet TableauNoir'.format(cls.compteur))
    combien = classmethod(combien)

    def info():
        print('ceci est la classe TableauNoir, on peut ecrire dessus et afficher mais aussi effacer')
    info = staticmethod(info)


nside = 128
sky = pysm.Sky(get_sky(nside, 'c1'))
testc = sky_map()
IPython.embed()

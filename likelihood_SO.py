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
from fgbuster import MixingMatrix
from fgbuster.component_model import CMB, Dust, Synchrotron
from scipy.linalg import block_diag
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
    #
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
        try:
            cmb_faraday = np.sum(lib.map_rotation(cmb_map_array,
                                                  self.bir_angle), 0)
            self.cmb_faraday = cmb_faraday
            if output:
                return cmb_faraday
        except TypeError:
            print('In faraday rotation, only one angle given. \n \
                   Faraday rotation impossible, try self.cmb_rotation()')

    def get_signal(self, output=0):
        if self.instrument == 'LAT':
            start_spectra = 0
        else:
            start_spectra = 1

        if hasattr(self, 'cmb_freq_rot'):
            signal_ = np.append(self.cmb_freq_rot[start_spectra:],
                                self.dust_freq_maps[start_spectra:], 0)
            signal = np.append(signal_, self.sync_freq_maps[start_spectra:], 0)
            del signal_
            # signal = self.cmb_freq_rot + self.dust_freq_maps + \
            # self.sync_freq_maps
            print('Signal with cmb rotation')

        elif hasattr(self, 'cmb_faraday'):
            signal_ = np.append(self.cmb_faraday[start_spectra:],
                                self.dust_freq_maps[start_spectra:], 0)
            signal = np.append(signal_, self.sync_freq_maps[start_spectra:], 0)
            del signal_
            print('Signal with Faraday rotation')

        else:
            signal_ = np.append(self.cmb_freq_maps[start_spectra:],
                                self.dust_freq_maps[start_spectra:], 0)
            signal = np.append(signal_, self.sync_freq_maps[start_spectra:], 0)
            print('Signal with no rotation')

        self.signal = signal
        if output:
            return signal

    def get_mixing_matrix(self):

        components = [CMB(), Dust(sky_map.dust_freq),
                      Synchrotron(sky_map.synchrotron_freq)]
        A = MixingMatrix(*components)
        A_ev = A.evaluator(self.frequencies)
        res = [1.59, 20, -3]
        A_ = A_ev(res)
        self.A_ = A_

        if self.instrument == 'SAT':
            mixing_matrix = np.repeat(A_, 2, 0)
        if self.instrument == 'LAT':
            print('Error: LAT in mixing matrix')
            # TODO: get LAT mixing matrix, because of T the second repetetion
            # should be changed ...
            # mixing_matrix = np.repeat(A_, 3, 0)
            # print(' WARNING: LAT has same T mixing matrix as Q & U')

        mixing_matrix = np.repeat(mixing_matrix, 2, 1)

        for i in range(np.shape(mixing_matrix)[0]):
            for j in range(np.shape(mixing_matrix)[1]):
                mixing_matrix[i, j] = mixing_matrix[i, j] *\
                    (((i % 2)-(1-j % 2)) % 2)

        self.mixing_matrix = mixing_matrix

    def get_miscalibration_angle_matrix(self):
        miscal_matrix = 1

        miscal_angles = [0, np.pi/4, np.pi/8] * u.rad
        frequencies_by_instrument = [2, 2, 2]
        try:
            if len(miscal_angles) != len(frequencies_by_instrument) or \
                    sum(frequencies_by_instrument) != len(self.frequencies):
                print('WARNING: miscalibration angles doesnt match the number',
                      'of instrument\n ',
                      'or the number of frequencies by instrument doesnt match'
                      ' the number of frequencies')

            else:
                miscal_matrix = 1
                instrument_nb = 0
                for angle in miscal_angles:

                    rotation_block = np.array(
                        [[np.cos(2*angle),  np.sin(2*angle)],
                         [-np.sin(2*angle), np.cos(2*angle)]
                         ])

                    print(rotation_block)
                    if type(miscal_matrix) == int:
                        miscal_matrix = rotation_block
                        for i in range(frequencies_by_instrument[instrument_nb]-1):
                            miscal_matrix = block_diag(miscal_matrix,
                                                       rotation_block)
                    else:
                        for i in range(frequencies_by_instrument[instrument_nb]):
                            miscal_matrix = block_diag(miscal_matrix,
                                                       rotation_block)
                    instrument_nb += 1

                self.miscal_matrix = miscal_matrix

        except AttributeError:
            print('No instrument frequencies !')

    def get_data(self):
        if hasattr(self, 'miscal_matrix'):
            A_s = np.dot(self.mixing_matrix, self.signal)
            M_A_s = np.dot(self.miscal_matrix, A_s)
            self.data = M_A_s
        else:
            A_s = np.dot(self.mixing_matrix, self.signal)
            self.data = A_s

    def from_pysm2data(self):
        self.get_pysm_sky()
        self.get_frequency()
        self.get_freq_maps()
        self.cmb_rotation()
        self.get_signal()
        self.get_mixing_matrix()
        self.get_miscalibration_angle_matrix()
        self.get_data()
        return self.data


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

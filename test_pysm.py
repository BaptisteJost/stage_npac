import sys, platform, os
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import camb
from camb import model, initialpower
import lib_project as lib
import plot_project as plotpro
import matplotlib.patches as mpatches
import healpy as hp
from astropy import units as u
import lensquest as lq
import copy

import pysm

nside = 64
l_max = nside * 4
raw_cl = True

pars, results, powers = lib.get_basics(l_max, raw_cl)

sky_config ={'dust': pysm.nominal.models('d0', nside)}
sky_config['synchrotron']= pysm.nominal.models('s0', nside)
# sky_config['cmb']= pysm.nominal.models('c1',nside)

sky = pysm.Sky(sky_config)

nu = 150 #np.array([10., 100., 500.])
total_signal = sky.signal()(nu)
dust_signal = sky.dust(nu)
synchrotron_signal = sky.synchrotron(nu)


mask = np.array(hp.ud_grade( hp.read_map("HFI_Mask_GalPlane-apo2_2048_R2.00.fits",field=2).astype(np.bool), nside))

proper_dust_masked = hp.ma(dust_signal)
proper_dust_masked.mask = np.logical_not(mask)

proper_sync_masked = hp.ma(synchrotron_signal)
proper_sync_masked.mask = np.logical_not(mask)

cl_dust_masked =  np.array([hp.anafast(proper_dust_masked, lmax = l_max).T[l,:] *l*(l+1)/(2*np.pi) for l in range(l_max )])
cl_dust =  np.array([hp.anafast(dust_signal, lmax = l_max).T[l,:] *l*(l+1)/(2*np.pi) for l in range(l_max )])

cl_synchrotron_masked = np.array([ hp.anafast(proper_sync_masked, lmax = l_max).T[l,:] * l*(l+1)/(2*np.pi) for l in range(l_max )])
cl_synchrotron = np.array([ hp.anafast(synchrotron_signal, lmax = l_max).T[l,:] * l*(l+1)/(2*np.pi) for l in range(l_max )])


spectra_dict={'camb_total': np.array([lib.cl_rotation(powers['total'], 0*u.deg)[l,:]*l*(l+1)/(2*np.pi) for l in range(l_max)])}
spectra_dict['cl_dust_masked']=cl_dust_masked
spectra_dict['cl_synchrotron_masked']=cl_synchrotron_masked

plotpro.spectra(spectra_dict, linear_cross=True)
plt.show()

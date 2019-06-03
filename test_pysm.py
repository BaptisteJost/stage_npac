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
print('dust_signal shape',np.shape(dust_signal))
mask = hp.ud_grade( hp.read_map("HFI_Mask_GalPlane-apo2_2048_R2.00.fits",field=2), nside).astype(np.bool)
np.set_printoptions(threshold=np.inf)
mask[0]=False
print('type 0',type(mask[0]))
print('type 1',type(mask[1]))
print('mask=',mask)

print('mask shape',np.shape(mask))
masked_dust = np.ma.array( dust_signal[0,:], mask = mask )
print('masked_dust shape',np.shape(masked_dust))
masked_sync = np.ma.array( dust_signal, mask = np.logical_not([mask,mask,mask]) )

print(masked_dust)
hp.mollview(masked_dust)
plt.show()
# hp.mollview(dust_signal[ 1, :], title = "Dust Q @ 150 GHz")
# hp.mollview(dust_signal[ 2, :], title = "Dust U @ 150 GHz")
# hp.mollview(synchrotron_signal[ 1, :], title = "synchrotron Q @ 150 GHz")
# hp.mollview(synchrotron_signal[ 2, :], title = "synchrotron U @ 150 GHz")
#
# plt.show()

# cl_total = hp.anafast(total_signal, lmax = l_max).T

cl_dust = np.array([hp.anafast(dust_signal, lmax = l_max).T[l,:] *l*(l+1)/(2*np.pi) for l in range(l_max )])
cl_synchrotron =np.array([ hp.anafast(synchrotron_signal, lmax = l_max).T[l,:] * l*(l+1)/(2*np.pi) for l in range(l_max )])

# cl_dust = hp.anafast(dust_signal, lmax = l_max).T
# cl_synchrotron = hp.anafast(synchrotron_signal, lmax = l_max).T

print('cl_dust shape',np.shape(cl_dust))
# print('cl_total shape',np.shape(cl_total))

plotpro.spectra({'dust':cl_dust, 'synchrotron':cl_synchrotron, 'total': np.array([lib.cl_rotation(powers['total'], 0*u.deg)[l,:]*l*(l+1)/(2*np.pi) for l in range(l_max)]) , 'dust+sync': cl_dust+cl_synchrotron})
plt.show()

# define the sky templates
# sky_config = {'dust' : d5_config, 'synchrotron' : s3_config}
# sky= pysm.Sky(sky_config)
# dust and synchrotron templates @ 150GHz
# dust = sky.dust(150)
# sync = sky.synchrotron(150)

print('dust=',dust_signal)
print('sync=',synchrotron_signal)
print('dust shape=',np.shape(dust_signal))
print('sync shape=',np.shape(synchrotron_signal))

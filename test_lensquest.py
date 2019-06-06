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


l_max = 1500
raw_cl = False
pars, results, powers = lib.get_basics(l_max, raw_cl)#, lens_potential = True)
for name in powers: print(name)

lens_potential_spectrum = powers['lens_potential']

lensed_spectrum_test =lib.cl_rotation( lib.cl_normalisation(powers['total']), 1*u.deg)
print('lensed_spectrum_test shape',np.shape(lensed_spectrum_test))
unlensed_spectrum_normalised = lib.cl_rotation(lib.cl_normalisation(powers['unlensed_total']), 1*u.deg)
lens_potential_spectrum_normalised = lib.cl_normalisation(lib.cl_normalisation(lens_potential_spectrum)) / (2*np.pi)

print('len_potential shape',np.shape( lens_potential_spectrum_normalised[:,0]))
print('unlensed_spectrum_normalised shape',np.shape( unlensed_spectrum_normalised))

lensquest_spectrum = lq.lenscls(unlensed_spectrum_normalised.T, lens_potential_spectrum_normalised[:,0]).T + unlensed_spectrum_normalised
lensquest_spectrum_then_rotation = lib.cl_rotation( lq.lenscls(lib.cl_rotation(lib.cl_normalisation(powers['unlensed_total']),0*u.deg).T, lens_potential_spectrum_normalised[:,0] ).T + lib.cl_rotation(lib.cl_normalisation(powers['unlensed_total']),0*u.deg) ,1*u.deg )
print('lensquest_spectrum shape',np.shape(lensquest_spectrum))
print('lensquest_spectrum=',lensquest_spectrum)

plot_dict = {'control':np.array([lensed_spectrum_test[l,:]*l*(l+1)/(2*np.pi) for l in range(l_max)]), \
             'lensquest_test':np.array([lensquest_spectrum[l,:]*l*(l+1)/(2*np.pi) for l in range(l_max)]) , \
             'lensquest_then_rotation':np.array([lensquest_spectrum_then_rotation[l,:]*l*(l+1)/(2*np.pi) for l in range(l_max)])}

plotpro.spectra(plot_dict)
plt.show()

print('lensquest_test TE =',plot_dict['lensquest_test'][:,3])
print('diff control lensquest_test=', np.sum(np.abs(plot_dict['control'][:l_max-200,3] - plot_dict['lensquest_test'][:l_max-200,3])))
print('diff lensquest_then_rotation lensquest_test=', np.sum(np.abs(plot_dict['lensquest_then_rotation'][:l_max-200,3] - plot_dict['lensquest_test'][:l_max-200,3])))

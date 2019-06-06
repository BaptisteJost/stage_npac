import numpy as np
import pysm
from astropy import units as u
import healpy as hp



def get_foreground_spectrum(nside, nu_u, mask_file="HFI_Mask_GalPlane-apo2_2048_R2.00.fits", mask_field=2):
    nu = nu_u.to(u.GHz).value
    l_max = nside * 3
    sky_config ={'dust': pysm.nominal.models('d0', nside), 'synchrotron': pysm.nominal.models('s0', nside)}
    sky = pysm.Sky(sky_config)
    dust_signal = sky.dust(nu)
    synchrotron_signal = sky.synchrotron(nu)

    mask = np.array(hp.ud_grade( hp.read_map(mask_file ,field=mask_field).astype(np.bool), nside))

    proper_dust_masked = hp.ma(dust_signal)
    proper_dust_masked.mask = np.logical_not(mask)

    proper_sync_masked = hp.ma(synchrotron_signal)
    proper_sync_masked.mask = np.logical_not(mask)

    cl_dust_masked = np.array([hp.anafast(proper_dust_masked, lmax = l_max).T[l,:] *l*(l+1)/(2*np.pi) for l in range(l_max )])
    cl_synchrotron_masked = np.array([ hp.anafast(proper_sync_masked, lmax = l_max).T[l,:] * l*(l+1)/(2*np.pi) for l in range(l_max )])

    return {'dust':cl_dust_masked, 'synchrotron':cl_synchrotron_masked}

nside = 1024
nu = 150*u.GHz
foreground_dict = get_foreground_spectrum(nside, nu)
print( 'dust shape', np.shape(foreground_dict['dust']) )
print( 'synchrotron shape', np.shape(foreground_dict['synchrotron']) )

np.save('dust_nside{}_nu{}_maskHFI_Mask_GalPlane-apo2_2048_R2_field2'.format(nside,nu.value),foreground_dict['dust'])
np.save('synchrotron_nside{}_nu{}_maskHFI_Mask_GalPlane-apo2_2048_R2_field2'.format(nside,nu.value), foreground_dict['synchrotron'])

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

# l_max = 1734
l_max = 1500
raw_cl = False
pars, results, powers = lib.get_basics(l_max, raw_cl)
for name in powers: print(name)

spectrum = 'unlensed_total' #'unlensed_total'
unchanged_cl = powers[spectrum]

angle_array = np.linspace(0.1 , 3. , 5)
angle_array = np.insert(angle_array, 0, 0.)
angle_array = angle_array * u.deg
print('angle_array',angle_array)

spectra_dict = lib.get_spectra_dict(unchanged_cl, angle_array, include_unchanged = False)
# print(spectra_dict)
beam_array = np.linspace(0,30,4)
for k in beam_array:
    w_inv = (2 * u.uK * u.arcmin)**2
    theta_fwhm = k * u.arcmin
    nl_spectra = lib.get_noise_spectra(w_inv, theta_fwhm, l_max, raw_output = False)
    spectra_dict[ (w_inv, theta_fwhm) ]=nl_spectra



for k in beam_array:
    for a in angle_array:
        lib.spectra_addition(spectra_dict, a, (w_inv, k * u.arcmin))

lensing = True
if lensing == True:
    spectra_dict['lensed_scalar'] =lib.cl_rotation( powers['lensed_scalar'] ,0.*u.deg)
    for k in beam_array:
        for a in angle_array:
            lib.spectra_addition(spectra_dict, (a, (w_inv, k * u.arcmin)) , 'lensed_scalar' )
# print('spectra dict keys=',spectra_dict.keys())
angle_fisher_array = angle_array
# print('angle_fisher_array=',angle_fisher_array)
beam_fisher_array = beam_array
fisher_dict = lib.get_fisher_dict( spectra_dict ,angle_fisher_array , w_inv, beam_fisher_array, lensing)


# fisher_array, fisher_element_array = lib.truncated_fisher_angle( spectra_dict[0*u.deg] , 3. * u.deg, spectra_dict[(3.*u.deg,(w_inv, 30.*u.arcmin))], f_sky = 1, return_elements = True, raw_cl = False, raw_cl_rot = False)
# print('fisher_array',fisher_array)
# print('diff EB-TEB=',fisher_array[-1]-fisher_array[-3])
# print('diff fisher=',np.sum(fisher_dict[(3*u.deg,(w_inv, 30.*u.arcmin))]) - fisher_array[-1])

fisher_trunc_array_dict = lib.get_truncated_fisher_dict( spectra_dict, angle_fisher_array, w_inv, beam_fisher_array )

# ================================= test anisotropic angle======================
# angle_anisotrop = np.random.normal(0,10, powers[spectrum].shape[0])
# cl_rot_anisotrop = lib.cl_rotation(powers[spectrum], angle_anisotrop* np.pi / 180.)

# ================================= plot =======================================

plot_fisher = 0
if plot_fisher == True :
    # fisher, fisher_element = lib.fisher_angle(spectra_dict[0.* u.deg], 0.*u.deg , return_elements = True)
    print('fisher TRUE(?)=',1/np.sqrt(sum(fisher_dict[(3*u.deg,(w_inv,30*u.arcmin))])))

    fig_uncertainty = plotpro.error_on_angle_wrt_scale(fisher_dict[(3*u.deg,(w_inv,30*u.arcmin))])
    plt.show()

    fig_cumu = plotpro.cumulative_error(fisher_dict[(3*u.deg,(w_inv,30*u.arcmin))])
    plt.show()

plot_noise_vs_lensing = 0
if plot_noise_vs_lensing == True :
    fig_white_noise_vs_lensing = plotpro.white_noise_vs_lensing(spectra_dict[(w_inv, 0*u.arcmin)],powers)
    plt.show(fig_white_noise_vs_lensing)

plot_noise_vs_lensing_new = 0
if plot_noise_vs_lensing_new == True :
    fig_white_noise_vs_lensing_new = plotpro.white_noise_vs_lensing(spectra_dict[(w_inv, 0*u.arcmin)],powers)
    plt.show(fig_white_noise_vs_lensing_new)

plot_fisher_noise_beam_array = 0
if plot_fisher_noise_beam_array == True:
    for key,value in fisher_dict.items():
        fig_uncertainty = plotpro.error_on_angle_wrt_scale(fisher_dict[key],label = '{k}'.format(k = key))
    plt.show()
    for key,value in fisher_dict.items():
        fig_cumu = plotpro.cumulative_error(fisher_dict[key],label ='{k}'.format(k = key))
    plt.show()

plot_dcl_over_cl = 0
if plot_dcl_over_cl == True:
    key_rot = angle_array[4]
    plotpro.relative_derivative(spectra_dict, key_rot)
    plt.show()

plot_spectra = 0
if plot_spectra == True:
    plotpro.spectra(spectra_dict)
    plt.show()

plot_truncated_fisher_cumlulative = 0
if plot_truncated_fisher_cumlulative == True:
    plotpro.truncated_fisher_cumulative(fisher_trunc_array_dict, (3*u.deg,(w_inv, 30.*u.arcmin)))
    plt.show()

plot_cumulative_3D = 1
if plot_cumulative_3D == True:
    surface, heatmap = plotpro.cumulative_error_3D(fisher_dict[((3*u.deg, (w_inv, 30. * u.arcmin)) , 'lensed_scalar')],((3*u.deg, (w_inv, 30. * u.arcmin)) , 'lensed_scalar'))
    plt.show(surface)
    plt.show(heatmap)


# ((3*u.deg, (w_inv, 30. * u.arcmin)) , 'lensed_scalar')






""""
---------------------------cimetery-------------------------------------
"""

# #plot the total lensed CMB power spectra versus unlensed, and fractional difference
# plot = True
# if plot == True :
#     fig, ax = plotpro.cl_comparison(unchanged_cl, spectra_dict[angle_array[0]], angle_array[0])
#     fisher_manual = lib.fisher_manual(unchanged_cl)
#     print('fisher MANUAL(angle = 0) = ',fisher_manual)
#     fisher = lib.fisher_angle(powers[spectrum], 0.)
#     print('fisher(angle =',0,') =',fisher)
#     print('relative difference = ',(fisher - fisher_manual)/fisher_manual )
#     print('')
#     for i in range(len(angle_array)-1):
#
#         cl_rot_2 = lib.cl_rotation(powers[spectrum], angle_array[i+1] * np.pi / 180.)
#         fig_2, ax_2 = plotpro.new_angle(cl_rot_2, angle_array[i+1], fig ,ax)
#         fisher = lib.fisher_angle(powers[spectrum], angle_array[i+1] * np.pi / 180)
#         print('fisher(angle =',angle_array[i+1],') =',fisher)
#
#         fisher_manual = lib.fisher_manual(cl_rot_2)
#         print('fisher MANUAL(angle =',angle_array[i+1],') =', fisher_manual)
#         print('relative difference = ',(fisher - fisher_manual)/fisher_manual )
#         print('')
#         fig = fig_2
#         ax = ax_2
#     plt.show(fig)
#
#
#
# do_cl_to_map = False
# if do_cl_to_map == True :
#
#     map = lib.cl_to_map(cl_rot, nside = 2048)
#     hp.mollview(map[0])
#     plt.show()
#     hp.mollview(map[1])
#     plt.show()
#     hp.mollview(map[2])
#     plt.show()
#
#     map = lib.cl_to_map(cl_rot_2, nside = 2048)
#     hp.mollview(map[0])
#     plt.show()
#     hp.mollview(map[1])
#     plt.show()
#     hp.mollview(map[2])
#     plt.show()

# plot_fisher_noise_beam_array = 1
# if plot_fisher_noise_beam_array ==True:
#     key_rot = 0*u.deg
#     key_noise = (w_inv, 0 * u.arcmin)
#
#     fisher, fisher_element = lib.fisher_angle(spectra_dict[key_rot], 0. * u.deg, cl_rot = spectra_dict[key_rot] , return_elements = True)
#     fisher_dict_old = {key_rot : fisher_element}
#     for k in beam_array:
#         key_noise = (w_inv, k * u.arcmin)
#
#         fisher_noise, fisher_element_noise = lib.fisher_angle( spectra_dict[key_rot] ,\
#          angle_array[0], cl_rot = spectra_dict[(key_rot, key_noise )] , return_elements = True, raw_cl_rot=False)
#
#         fisher_dict_old[(key_rot,key_noise)]= fisher_element_noise
#
#     for key,value in fisher_dict_old.items():
#         fig_uncertainty = plotpro.error_on_angle_wrt_scale(fisher_dict_old[key],label = '{k}'.format(k = key))
#     plt.show()
#     for key,value in fisher_dict_old.items():
#         fig_cumu = plotpro.cumulative_error(fisher_dict_old[key],label ='{k}'.format(k = key))
#     plt.show()


# plot_fisher_noise = True
# if plot_fisher_noise == True :
#     angle = 0*u.deg
#     key_rot = angle
#     key_noise = (w_inv, 0*u.arcmin)
#     spectrum_rot_noise = spectra_dict[ (key_rot, key_noise) ]
#
#     fisher_noise, fisher_element_noise = lib.fisher_angle( spectra_dict[key_rot] , angle, cl_rot = spectrum_rot_noise , return_elements = True)
#     fig_uncertainty_noise = plotpro.error_on_angle_wrt_scale(fisher_element_noise, label='noise')
#     plt.show()
#
#     fig_cumu_noise = plotpro.cumulative_error(fisher_element_noise,label = 'noise')
#     plt.show()

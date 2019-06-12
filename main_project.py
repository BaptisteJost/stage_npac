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
import pysm

# l_max = 1734
# nside = 64
l_max = 1500#1500
raw_cl = False
pars, results, powers = lib.get_basics(l_max, raw_cl)
for name in powers: print(name)

spectrum = 'total' #'unlensed_total'
unchanged_cl = powers[spectrum]


angle_array = np.linspace(0. ,360. , 200)
# angle_array = np.insert(angle_array, 0, 0.)
angle_array = angle_array * u.deg
print('angle_array',angle_array)

spectra_dict = lib.get_spectra_dict(unchanged_cl, angle_array, include_unchanged = False)

unlensed_cl = powers['unlensed_total']
# unlensed_cl = lib.cl_normalisation(powers['unlensed_total'])
for angle in angle_array:
    spectra_dict[('unlensed',angle)]=lib.cl_rotation(unlensed_cl, angle)

# print(spectra_dict)
beam_array = np.linspace(0,30,4)
print('beam_array',beam_array)
for k in beam_array:
    w_inv = (2 * u.uK * u.arcmin)**2
    theta_fwhm = k * u.arcmin
    nl_spectra = lib.get_noise_spectra(w_inv, theta_fwhm, l_max, raw_output = False)
    spectra_dict[ (w_inv, theta_fwhm) ]=nl_spectra

for k in beam_array:
    for a in angle_array:
        lib.spectra_addition(spectra_dict, a, (w_inv, k * u.arcmin))
        lib.spectra_addition(spectra_dict, ('unlensed',a), (w_inv, k * u.arcmin))



foregrounds = 1
if foregrounds==True:
    # foreground_dict = lib.get_foreground_spectrum(nside, 150*u.GHz)
    # lib.spectra_addition(foreground_dict, 'dust','synchrotron')
    if l_max >= 3 * 512 :
        print('l_max >= 3*512, pysm uses nside=512 maps, foregrounds spectra of l>1500 are meaningless')
    dust_spectra = np.load('dust_nside1024_nu150.0_maskHFI_Mask_GalPlane-apo2_2048_R2_field2.npy')
    synchrotron_spectra = np.load('synchrotron_nside1024_nu150.0_maskHFI_Mask_GalPlane-apo2_2048_R2_field2.npy')

    spectra_dict['foregrounds'] = dust_spectra[:l_max+1] + synchrotron_spectra[:l_max+1]

    for k in beam_array:
        for a in angle_array:
            lib.spectra_addition(spectra_dict, (a, (w_inv, k * u.arcmin)), 'foregrounds')
            lib.spectra_addition(spectra_dict, (('unlensed',a), (w_inv, k * u.arcmin)), 'foregrounds')

compute_fisher = 0
if compute_fisher == True:
    angle_fisher_array = angle_array
    beam_fisher_array = beam_array
    fisher_dict = lib.get_fisher_dict( spectra_dict ,angle_fisher_array , w_inv, beam_fisher_array, foregrounds= foregrounds)

compute_fisher_trunc = False
if compute_fisher_trunc == True and compute_fisher == True:
    fisher_trunc_array_dict = lib.get_truncated_fisher_dict( spectra_dict, angle_fisher_array, w_inv, beam_fisher_array , foregrounds = foregrounds)

for key, value in spectra_dict.items() :
    print (key)

# ================================= test anisotropic angle======================
# angle_anisotrop = np.random.normal(0,10, powers[spectrum].shape[0])
# cl_rot_anisotrop = lib.cl_rotation(powers[spectrum], angle_anisotrop* np.pi / 180.)

# ================================= plot =======================================

plot_fisher = 0
if plot_fisher == True and compute_fisher == True:
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
if plot_fisher_noise_beam_array == True and compute_fisher == True:
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
if plot_truncated_fisher_cumlulative == True and compute_fisher == True and compute_fisher_trunc == True:
    if foregrounds == True :
        plotpro.truncated_fisher_cumulative(fisher_trunc_array_dict, ((3*u.deg, (w_inv, 30. * u.arcmin)),'foregrounds'))
    else:
        plotpro.truncated_fisher_cumulative(fisher_trunc_array_dict, (3*u.deg, (w_inv, 30. * u.arcmin)))

    plt.show()

plot_cumulative_3D = 0
if plot_cumulative_3D == True and compute_fisher == True:
    if foregrounds == True:
        surface, heatmap = plotpro.cumulative_error_3D(fisher_dict[((3*u.deg, (w_inv, 30. * u.arcmin)),'foregrounds')],((3*u.deg, (w_inv, 30. * u.arcmin)),'foregrounds'))
    else:
        surface, heatmap = plotpro.cumulative_error_3D(fisher_dict[(3*u.deg, (w_inv, 30. * u.arcmin))],(3*u.deg, (w_inv, 30. * u.arcmin)))

    plt.show(surface)
    plt.show(heatmap)

comp_foregrounds = 0
if comp_foregrounds == True and foregrounds == True and compute_fisher == True:
    l_min_range=[2,10,50,100,500]
    color = next(plt.gca()._get_lines.prop_cycler)['color']
    fig_cumu = plotpro.cumulative_error(fisher_dict[(3*u.deg, (w_inv, 30. * u.arcmin))],label ='no foregrounds, lmin={}'.format(l_min_range[0]), l_min =l_min_range[0],color = color )

    fig_cumu = plotpro.cumulative_error(fisher_dict[((3*u.deg, (w_inv, 30. * u.arcmin)),'foregrounds')],label ='foregrounds, lmin={}'.format(l_min_range[0]), l_min =l_min_range[0], dotted = True, color = color)
    for l in l_min_range[1:]:
        color = next(plt.gca()._get_lines.prop_cycler)['color']

        fig_cumu = plotpro.cumulative_error(fisher_dict[(3*u.deg, (w_inv, 30. * u.arcmin))],label ='no foregrounds, lmin={}'.format(l), l_min =l, color = color)
        fig_cumu = plotpro.cumulative_error(fisher_dict[((3*u.deg, (w_inv, 30. * u.arcmin)),'foregrounds')],label ='foregrounds, lmin={}'.format(l), l_min =l, color = color, dotted = True)
    plt.title('cumulative error for rotation {}, noise beam {} and with and without foregrounds and different lmin'.format(3*u.deg, 30. * u.arcmin))
    plt.show()




test_noise = lib.get_noise_spectrum(100 * (u.uK*u.uK * u.rad*u.rad), 30. * u.arcmin, l_max)
l_norm = np.arange(l_max+1)
test_noise = test_noise * l_norm * (l_norm+1) / (2*np.pi)
test_spectra = spectra_dict[((('unlensed',angle), (w_inv, 30. * u.arcmin)),'foregrounds')]
test_spectra[:,4] += test_noise

test_likelihood = True
if test_likelihood == True:
    cov_angle = angle_array[2]
    print('angle covariance=',cov_angle)
    cov_key = (cov_angle, (w_inv, 30. * u.arcmin))

    likelihood_no_foregrounds = []
    likelihood_no_noise = []
    likelihood_noise = []
    likelihood_lens = []
    likelihood_lens_cov = []
    likelihood_test = []

    if compute_fisher == True :
        fisher_norm = fisher_dict[cov_key]
    for angle in angle_array:
        likelihood_no_noise.append( lib.likelihood(spectra_dict[ cov_key ], spectra_dict[angle] ) )
        likelihood_no_foregrounds.append( lib.likelihood(spectra_dict[ cov_key ], spectra_dict[(angle, (w_inv, 30. * u.arcmin))] ) )
        likelihood_noise.append(lib.likelihood(spectra_dict[ cov_key ], spectra_dict[((angle, (w_inv, 30. * u.arcmin)),'foregrounds')] ) )
        likelihood_lens_cov.append( lib.likelihood(spectra_dict[ ((('unlensed',cov_angle), (w_inv, 30. * u.arcmin)),'foregrounds') ], spectra_dict[((('unlensed',angle), (w_inv, 30. * u.arcmin)),'foregrounds')] ) )
        likelihood_lens.append( lib.likelihood(spectra_dict[ cov_key ], spectra_dict[((('unlensed',angle), (w_inv, 30. * u.arcmin)),'foregrounds')] ) )
        likelihood_test.append( lib.likelihood(spectra_dict[ cov_key ], test_spectra)  )

    likelihood_no_noise = likelihood_no_noise / max(likelihood_no_noise)
    likelihood_no_foregrounds = likelihood_no_foregrounds / max(likelihood_no_foregrounds)
    likelihood_noise = likelihood_noise / max(likelihood_noise)
    likelihood_lens = likelihood_lens / max(likelihood_lens)
    likelihood_lens_cov = likelihood_lens_cov / max(likelihood_lens_cov)
    likelihood_test = likelihood_test / max(likelihood_test)


    plt.plot(angle_array, likelihood_no_noise,'*',label='no noise')
    plt.plot(angle_array, likelihood_no_foregrounds,label='no foregrounds')
    plt.plot(angle_array, likelihood_noise,'--',label='all')
    plt.plot(angle_array, likelihood_lens,':',label='unlensed')
    plt.plot(angle_array, likelihood_lens_cov,label='unlensed_cov')
    plt.plot(angle_array, likelihood_test,'-.',label='test')




    plt.legend()
    plt.show()
    # print('likelihood=', likelihood)
# ((3*u.deg, (w_inv, 30. * u.arcmin)) , 'lensed_scalar')






""""
---------------------------cimetery-------------------------------------
"""
# lensing = False
# if lensing == True:
#     spectra_dict['lensed_scalar'] =lib.cl_rotation( powers['lensed_scalar'] ,0.*u.deg)
#     for k in beam_array:
#         for a in angle_array:
#             lib.spectra_addition(spectra_dict, (a, (w_inv, k * u.arcmin)) , 'lensed_scalar' )


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

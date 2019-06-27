import sys, platform, os
# import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import camb
from camb import model, initialpower
import lib_project as lib
import plot_project as plotpro
# import matplotlib.patches as mpatches
# import healpy as hp
from astropy import units as u
# import pysm
import copy

# l_max = 1734
# nside = 64
l_max = 1500
raw_cl = False
pars, results, powers = lib.get_basics(l_max, raw_cl)
for name in powers: print(name)

spectrum = 'total' #'unlensed_total'
unchanged_cl = powers[spectrum]


angle_array = np.linspace(0.4,0.53 , 10)
angle_array = np.insert(angle_array, 0, 0.)
# i = 0
# for angle in angle_array:
#     if angle < 0:
#         index = i
#     i +=1
# angle_array = np.insert(angle_array, index +1, 0.)
imposed_rotation_angle = 0.5
i = 0
for angle in angle_array:
    if angle < imposed_rotation_angle:
        index2 = i
    i +=1

angle_array = np.insert(angle_array, index2 +1, imposed_rotation_angle)
angle_array = angle_array * u.deg
print('angle_array',angle_array)

spectra_dict = lib.get_spectra_dict(unchanged_cl, angle_array, include_unchanged = False)
print('size rotation',spectra_dict[0*u.deg].nbytes)
unlensed_cl = powers['unlensed_total']
# unlensed_cl = lib.cl_normalisation(powers['unlensed_total'])
for angle in angle_array:
    spectra_dict[(angle,'unlensed')]=lib.cl_rotation(unlensed_cl, angle)
print('size rot+lens',spectra_dict[(0*u.deg,'unlensed')].nbytes)
# print(spectra_dict)
beam_array = np.linspace(0,30,4)
# w_inv_array = (np.array([0,2,4,6,8,10]) * u.uK * u.arcmin)**2
w_inv_array = (np.array([0,1,2]) * u.uK * u.arcmin)**2

print('beam_array',beam_array)
print('w_inv_array',w_inv_array)
for k in beam_array:
    for w_inv in w_inv_array:
        # w_inv = (2 * u.uK * u.arcmin)**2
        theta_fwhm = k * u.arcmin
        nl_spectra = lib.get_noise_spectra(w_inv, theta_fwhm, l_max, raw_output = False)
        spectra_dict[ (w_inv, theta_fwhm) ]=nl_spectra

for k in beam_array:
    for w_inv in w_inv_array:

        for a in angle_array:
            lib.spectra_addition(spectra_dict, a, (w_inv, k * u.arcmin))
            lib.spectra_addition(spectra_dict, (a,'unlensed'), (w_inv, k * u.arcmin))



foregrounds = 0
if foregrounds==True:
    # foreground_dict = lib.get_foreground_spectrum(nside, 150*u.GHz)
    # lib.spectra_addition(foreground_dict, 'dust','synchrotron')
    if l_max >= 3 * 512 :
        print('l_max >= 3*512, pysm uses nside=512 maps, foregrounds spectra of l>1500 are meaningless')
    dust_spectra = np.load('dust_nside1024_nu150.0_maskHFI_Mask_GalPlane-apo2_2048_R2_field2.npy')
    synchrotron_spectra = np.load('synchrotron_nside1024_nu150.0_maskHFI_Mask_GalPlane-apo2_2048_R2_field2.npy')

    spectra_dict['foregrounds'] = dust_spectra[:l_max+1] + synchrotron_spectra[:l_max+1]
    spectra_dict['foregrounds/10'] = (dust_spectra[:l_max+1] + synchrotron_spectra[:l_max+1])/10
    spectra_dict['foregrounds/100'] = (dust_spectra[:l_max+1] + synchrotron_spectra[:l_max+1])/100
    spectra_dict['foregrounds/1000'] = (dust_spectra[:l_max+1] + synchrotron_spectra[:l_max+1])/1000


    for k in beam_array:
        for w_inv in w_inv_array:
            for a in angle_array:
                lib.spectra_addition(spectra_dict, (a, (w_inv, k * u.arcmin)), 'foregrounds')
                lib.spectra_addition(spectra_dict, (a, (w_inv, k * u.arcmin)), 'foregrounds/10')
                lib.spectra_addition(spectra_dict, (a, (w_inv, k * u.arcmin)), 'foregrounds/100')
                lib.spectra_addition(spectra_dict, (a, (w_inv, k * u.arcmin)), 'foregrounds/1000')

                lib.spectra_addition(spectra_dict, ((a,'unlensed'), (w_inv, k * u.arcmin)), 'foregrounds')

compute_fisher = 1
if compute_fisher == True:
    angle_fisher_array = angle_array
    beam_fisher_array = beam_array
    w_inv = (2 * u.uK * u.arcmin)**2
    fisher_dict = lib.get_fisher_dict( spectra_dict ,angle_fisher_array , w_inv, beam_fisher_array, foregrounds= foregrounds)

compute_fisher_trunc = 0
if compute_fisher_trunc == True and compute_fisher == True:
    fisher_trunc_array_dict = lib.get_truncated_fisher_dict( spectra_dict, angle_fisher_array, w_inv, beam_fisher_array , foregrounds = foregrounds)

# for key, value in spectra_dict.items() :
#     print (key)

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
        plotpro.truncated_fisher_cumulative(fisher_trunc_array_dict, (imposed_rotation_angle*u.deg, 'no noise'))
        print('fisher trunc',fisher_trunc_array_dict[(angle_array[0], 'no noise')])
    plt.show()

plot_cumulative_3D = 1
if plot_cumulative_3D == True and compute_fisher == True:
    if foregrounds == True:
        surface, heatmap = plotpro.cumulative_error_3D(fisher_dict[((3*u.deg, (w_inv, 30. * u.arcmin)),'foregrounds')],((3*u.deg, (w_inv, 30. * u.arcmin)),'foregrounds'))
    else:
        surface, heatmap = plotpro.cumulative_error_3D(fisher_dict[(imposed_rotation_angle*u.deg, 'no noise')],(imposed_rotation_angle*u.deg, 'no noise'))

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




# test_noise = lib.get_noise_spectrum(4 * (u.uK*u.uK * u.rad*u.rad), 30. * u.arcmin, l_max)
# l_norm = np.arange(l_max+1)
# test_noise = test_noise * l_norm * (l_norm+1) / (2*np.pi)
# test_spectra = copy.deepcopy(spectra_dict[((('unlensed',angle_array[2]), (w_inv, 30. * u.arcmin)),'foregrounds')])
# # test_spectra[:,4] += test_noise

test_likelihood = 0
if test_likelihood == True:
    data_angle = imposed_rotation_angle * u.deg #angle_array[33]
    print('angle covariance=',data_angle)
    # data_key = (data_angle, (w_inv, 30. * u.arcmin))
    data_key =  ((data_angle, (w_inv, 30. * u.arcmin)),'foregrounds')
    # data_key =  data_angle


    likelihood_angle = []
    likelihood_angle_lensing = []
    likelihood_angle_lensing_noise = []
    likelihood_angle_lensing_noise_foregrounds = []
    likelihood_angle_lensing_noise_foregrounds10 = []
    likelihood_angle_lensing_noise_foregrounds100 = []
    likelihood_angle_lensing_noise_foregrounds1000 = []



    extra_noise = 0
    if extra_noise:
        likelihood_angle_lensing_noise_foregrounds_100uKnoise = []


        # test_noise = lib.get_noise_spectra(100 * (u.uK*u.uK * u.arcmin*u.arcmin), 30. * u.arcmin, l_max)
        noise_cst = np.ones(l_max+1) * 9e+5
        test_noise = np.array([np.array([noise_cst[k] / 2., noise_cst[k] , noise_cst[k] , \
                            noise_cst[k]/np.sqrt(2), noise_cst[k], noise_cst[k]/np.sqrt(2)]) \
                            for k in range(len(noise_cst))])
        # test_noise[:,0]=0
        # test_noise[:,1]=0
        # test_noise[:,2]=0
        # test_noise[:,3]=0
        # test_noise[:,4]=0
        # test_noise[:,5]=0
        test_spectra_cov = copy.deepcopy(spectra_dict[((data_angle ,(w_inv, 30. * u.arcmin)),'foregrounds')]) + test_noise

    plot_increment_spectra = 0
    if plot_increment_spectra:
        plotpro.spectra( {'CMB w rotation and lensing': spectra_dict[data_angle], 'noise':spectra_dict[(w_inv, 30. * u.arcmin)], 'foregrounds':spectra_dict['foregrounds'], 'extra noise':test_noise, 'all':test_spectra_cov},linear_cross = False )
        plt.show()

    likelihood_angle_lensing_noise_list = []
    for w_inv in w_inv_array:
        likelihood_angle_lensing_noise_list.append([])
    for angle in angle_array[1:]:
        likelihood_angle.append( lib.likelihood(spectra_dict[ (angle,'unlensed') ], spectra_dict[(data_angle,'unlensed')] ) )
        likelihood_angle_lensing.append( lib.likelihood(spectra_dict[ angle ], spectra_dict[data_angle] ) )
        noise_counter = 0
        for w_inv in w_inv_array:
            likelihood_angle_lensing_noise_list[noise_counter].append( lib.likelihood(spectra_dict[ (angle, (w_inv, 30 * u.arcmin)) ], spectra_dict[(data_angle, (w_inv, 30 * u.arcmin))] ) )
            noise_counter+=1
        likelihood_angle_lensing_noise_foregrounds.append( lib.likelihood(spectra_dict[ (angle, (w_inv, 30. * u.arcmin)) ], spectra_dict[((data_angle, (w_inv, 30. * u.arcmin)),'foregrounds')] ) )
        likelihood_angle_lensing_noise_foregrounds10.append( lib.likelihood(spectra_dict[ (angle, (w_inv, 30. * u.arcmin)) ], spectra_dict[((data_angle, (w_inv, 30. * u.arcmin)),'foregrounds/10')] ) )
        likelihood_angle_lensing_noise_foregrounds100.append( lib.likelihood(spectra_dict[ (angle, (w_inv, 30. * u.arcmin)) ], spectra_dict[((data_angle, (w_inv, 30. * u.arcmin)),'foregrounds/100')] ) )
        likelihood_angle_lensing_noise_foregrounds1000.append( lib.likelihood(spectra_dict[ (angle, (w_inv, 30. * u.arcmin)) ], spectra_dict[((data_angle, (w_inv, 30. * u.arcmin)),'foregrounds/1000')] ) )




        if extra_noise:
            test_spectra = copy.deepcopy(spectra_dict[((angle ,(w_inv, 30. * u.arcmin)),'foregrounds')]) + test_noise


        if extra_noise:
            likelihood_angle_lensing_noise_foregrounds_100uKnoise.append( lib.likelihood(spectra_dict[ ((angle, (w_inv, 30. * u.arcmin)),'foregrounds') ] , test_spectra_cov )  )
        # likelihood_test.append( lib.likelihood(spectra_dict[ cov_key ], test_spectra)  )

    # likelihood_angle =                                      lib.likelihood_normalisation(np.exp(-np.array(lib.likelihood_normalisation(likelihood_angle)))) #(likelihood_no_noise -min(likelihood_no_noise))/ (max(likelihood_no_noise)-min(likelihood_no_noise))
    # likelihood_angle =                                      lib.likelihood_normalisation(np.exp(-np.array(likelihood_angle))) #(likelihood_no_noise -min(likelihood_no_noise))/ (max(likelihood_no_noise)-min(likelihood_no_noise))
    test_lnL = 0
    if test_lnL == 0:
        likelihood_angle =                                     np.exp(-np.array(lib.likelihood_normalisation(likelihood_angle))) #(likelihood_no_noise -min(likelihood_no_noise))/ (max(likelihood_no_noise)-min(likelihood_no_noise))
        likelihood_angle_lensing =                             np.exp(-np.array(lib.likelihood_normalisation(likelihood_angle_lensing))) #(likelihood_no_foregrounds -min(likelihood_no_foregrounds))/ (max(likelihood_no_foregrounds)-min(likelihood_no_foregrounds))
        noise_counter = 0
        for i in range(len(w_inv_array)):
            likelihood_angle_lensing_noise_list[i] =           np.exp(-np.array(lib.likelihood_normalisation(likelihood_angle_lensing_noise_list[i]))) #(likelihood_noise -min(likelihood_noise))/ (max(likelihood_noise)-min(likelihood_noise))

        likelihood_angle_lensing_noise_foregrounds =           np.exp(-np.array(lib.likelihood_normalisation(likelihood_angle_lensing_noise_foregrounds)))
        likelihood_angle_lensing_noise_foregrounds10 =         np.exp(-np.array(lib.likelihood_normalisation(likelihood_angle_lensing_noise_foregrounds10)))
        likelihood_angle_lensing_noise_foregrounds100 =        np.exp(-np.array(lib.likelihood_normalisation(likelihood_angle_lensing_noise_foregrounds100)))
        likelihood_angle_lensing_noise_foregrounds1000 =        np.exp(-np.array(lib.likelihood_normalisation(likelihood_angle_lensing_noise_foregrounds1000)))


        if extra_noise:
            likelihood_angle_lensing_noise_foregrounds_100uKnoise =np.exp(-np.array(lib.likelihood_normalisation(likelihood_angle_lensing_noise_foregrounds_100uKnoise))) #(likelihood_test -min(likelihood_test))/ (max(likelihood_test)-min(likelihood_test))


    alpha_plot = 0.5
    plt.axes([0.1, 0.1, 0.6, 0.75])
    plt.plot(angle_array[1:], likelihood_angle,label='Primordial spectra',alpha = alpha_plot)
    print('size likelihood=',likelihood_angle.nbytes)
    plt.plot(angle_array[1:], likelihood_angle_lensing,label='Lensed spectra')
    for i in range(len(w_inv_array)):
        plt.plot(angle_array[1:], likelihood_angle_lensing_noise_list[i],label='+= intrumental noise, w_inv={}'.format(w_inv_array[i]),alpha = alpha_plot)
    # plt.plot(angle_array, likelihood_angle_lensing_noise_foregrounds,':',label='+= foregrounds')
    if extra_noise:
        plt.plot(angle_array, likelihood_angle_lensing_noise_foregrounds_100uKnoise,label='+= extra 100K noise')
    # plt.plot(angle_array, likelihood_test,'-.',label='test')
    plt.plot(angle_array[1:], likelihood_angle_lensing_noise_foregrounds,label='+=foregrounds in data not model',alpha = alpha_plot)
    plt.axvline(x = data_angle.value,linestyle=':' ,label='Input rotation angle',color = 'k')

    test_gaussian = 0
    if test_gaussian:
        fisher = lib.fisher_angle(spectra_dict[0*u.deg], data_angle, cl_rot= spectra_dict[data_angle],  return_elements = False) /( u.rad* u.rad)
        print('fisher rad =',fisher)
        fisher_unlensed_unlensed=lib.fisher_angle(spectra_dict[(0*u.deg,'unlensed')], data_angle, cl_rot= spectra_dict[(data_angle,'unlensed')],  return_elements = False) /( u.rad* u.rad)
        fisher_unlensed_lensed=lib.fisher_angle(spectra_dict[(0*u.deg,'unlensed')], data_angle, cl_rot= spectra_dict[data_angle],  return_elements = False) /( u.rad* u.rad)
        fisher_lensed_lensed=lib.fisher_angle(spectra_dict[0*u.deg], data_angle, cl_rot= spectra_dict[data_angle],  return_elements = False) /( u.rad* u.rad)
        fisher_lensed_unlensed = lib.fisher_angle(spectra_dict[0*u.deg], data_angle, cl_rot= spectra_dict[(data_angle,'unlensed')],  return_elements = False) /( u.rad* u.rad)
        print('fisher unlensed unlensed',fisher_unlensed_unlensed)
        print('fisher_unlensed_lensed',fisher_unlensed_lensed)
        print('fisher_lensed_lensed',fisher_lensed_lensed)
        print('fisher_lensed_unlensed',fisher_lensed_unlensed)
        fisher = fisher.to(1/(u.deg * u.deg)).value
        print('fisher deg =',fisher)

        # fisher = np.sum(fisher_dict[(data_angle,'no noise')])
        gaussian = np.exp( -(angle_array[1:].value - data_angle.value)**2 * fisher) #* np.sqrt(fisher / np.pi)
        # gaussian = lib.likelihood_normalisation(gaussian)

        plt.plot(angle_array[1:], gaussian, '-.', label='gaussian test')

    plt.legend(title = 'Included in model:', loc='upper left', fontsize = 'small', bbox_to_anchor=(1, 0.5 ))
    plt.title('Likelihood Estimation for different models and data')
    plt.xlabel('Rotation angle (degree)')
    plt.ylabel('Likelihood')

    # plt.savefig('test_save.png')
    plt.show()
    likelihood_presentation = 1
    if likelihood_presentation == 1:
        alpha_plot = 0.5
        plt.axes([0.1, 0.1, 0.6, 0.75])
        plt.plot(angle_array[1:], likelihood_angle,label='Primordial spectra',alpha = alpha_plot)
        plt.axvline(x = data_angle.value,linestyle=':' ,label='Input rotation angle',color = 'k')
        plt.legend(title = 'Included in model:', loc='upper left', fontsize = 'small', bbox_to_anchor=(1, 0.5 ))
        plt.title('Likelihood Estimation for different models and data')
        plt.xlabel('Rotation angle (degree)')
        plt.ylabel('Likelihood')
        plt.show()


        plt.axes([0.1, 0.1, 0.6, 0.75])
        plt.plot(angle_array[1:], likelihood_angle,label='Primordial spectra',alpha = alpha_plot)
        plt.plot(angle_array[1:], likelihood_angle_lensing,label='Lensed spectra')
        plt.axvline(x = data_angle.value,linestyle=':' ,label='Input rotation angle',color = 'k')
        plt.legend(title = 'Included in model:', loc='upper left', fontsize = 'small', bbox_to_anchor=(1, 0.5 ))
        plt.title('Likelihood Estimation for different models and data')
        plt.xlabel('Rotation angle (degree)')
        plt.ylabel('Likelihood')
        plt.show()


        plt.axes([0.1, 0.1, 0.6, 0.75])
        plt.plot(angle_array[1:], likelihood_angle,label='Primordial spectra',alpha = alpha_plot)
        plt.plot(angle_array[1:], likelihood_angle_lensing,label='Lensed spectra')
        plt.plot(angle_array[1:], likelihood_angle_lensing_noise_list[-1],label='Lensed + intrumental noise, w_inv={}'.format(w_inv_array[-1]),alpha = alpha_plot)
        plt.axvline(x = data_angle.value,linestyle=':' ,label='Input rotation angle',color = 'k')
        plt.legend(title = 'Included in model:', loc='upper left', fontsize = 'small', bbox_to_anchor=(1, 0.5 ))
        plt.title('Likelihood Estimation for different models and data')
        plt.xlabel('Rotation angle (degree)')
        plt.ylabel('Likelihood')
        plt.show()

        plt.axes([0.1, 0.1, 0.6, 0.75])
        plt.plot(angle_array[1:], likelihood_angle,label='Primordial spectra',alpha = alpha_plot)
        plt.plot(angle_array[1:], likelihood_angle_lensing,label='Lensed spectra')
        plt.plot(angle_array[1:], likelihood_angle_lensing_noise_list[-1],label='Lensed + intrumental noise, w_inv={}'.format(w_inv_array[-1]),alpha = alpha_plot)
        plt.plot(angle_array[1:], likelihood_angle_lensing_noise_foregrounds,label='Lensed + noise + foregrounds in data not model',alpha = alpha_plot)
        plt.axvline(x = data_angle.value,linestyle=':' ,label='Input rotation angle',color = 'k')
        plt.legend(title = 'Included in model:', loc='upper left', fontsize = 'small', bbox_to_anchor=(1, 0.5 ))
        plt.title('Likelihood Estimation for different models and data')
        plt.xlabel('Rotation angle (degree)')
        plt.ylabel('Likelihood')
        plt.show()

        plt.axes([0.1, 0.1, 0.6, 0.75])
        plt.plot(angle_array[1:], likelihood_angle,label='Primordial spectra',alpha = alpha_plot)
        plt.plot(angle_array[1:], likelihood_angle_lensing,label='Lensed spectra')
        plt.plot(angle_array[1:], likelihood_angle_lensing_noise_list[-1],label='Lensed + intrumental noise, w_inv={}'.format(w_inv_array[-1]),alpha = alpha_plot)
        plt.plot(angle_array[1:], likelihood_angle_lensing_noise_foregrounds,label='Lensed + noise + foregrounds in data not model',alpha = alpha_plot)
        plt.plot(angle_array[1:], likelihood_angle_lensing_noise_foregrounds10,label=r'Lensed + noise + $\frac{foregrounds}{10}$ in data not model',alpha = alpha_plot)
        plt.axvline(x = data_angle.value,linestyle=':' ,label='Input rotation angle',color = 'k')
        plt.legend(title = 'Included in model:', loc='upper left', fontsize = 'small', bbox_to_anchor=(1, 0.5 ))
        plt.title('Likelihood Estimation for different models and data')
        plt.xlabel('Rotation angle (degree)')
        plt.ylabel('Likelihood')
        plt.show()

        plt.axes([0.1, 0.1, 0.6, 0.75])
        plt.plot(angle_array[1:], likelihood_angle,label='Primordial spectra',alpha = alpha_plot)
        plt.plot(angle_array[1:], likelihood_angle_lensing,label='Lensed spectra')
        plt.plot(angle_array[1:], likelihood_angle_lensing_noise_list[-1],label='Lensed + intrumental noise, w_inv={}'.format(w_inv_array[-1]),alpha = alpha_plot)
        plt.plot(angle_array[1:], likelihood_angle_lensing_noise_foregrounds,label='Lensed + noise + foregrounds in data not model',alpha = alpha_plot)
        plt.plot(angle_array[1:], likelihood_angle_lensing_noise_foregrounds10,label=r'Lensed + noise + $\frac{foregrounds}{10}$ in data not model',alpha = alpha_plot)
        plt.plot(angle_array[1:], likelihood_angle_lensing_noise_foregrounds100,label=r'Lensed + noise + $\frac{foregrounds}{100}$ in data not model',alpha = alpha_plot)
        plt.axvline(x = data_angle.value,linestyle=':' ,label='Input rotation angle',color = 'k')
        plt.legend(title = 'Included in model:', loc='upper left', fontsize = 'small', bbox_to_anchor=(1, 0.5 ))
        plt.title('Likelihood Estimation for different models and data')
        plt.xlabel('Rotation angle (degree)')
        plt.ylabel('Likelihood')
        plt.show()

        plt.axes([0.1, 0.1, 0.6, 0.75])
        plt.plot(angle_array[1:], likelihood_angle,label='Primordial spectra',alpha = alpha_plot)
        plt.plot(angle_array[1:], likelihood_angle_lensing,label='Lensed spectra')
        plt.plot(angle_array[1:], likelihood_angle_lensing_noise_list[-1],label='Lensed + intrumental noise, w_inv={}'.format(w_inv_array[-1]),alpha = alpha_plot)
        plt.plot(angle_array[1:], likelihood_angle_lensing_noise_foregrounds,label='Lensed + noise + foregrounds in data not model',alpha = alpha_plot)
        plt.plot(angle_array[1:], likelihood_angle_lensing_noise_foregrounds10,label=r'Lensed + noise + $\frac{foregrounds}{10}$ in data not model',alpha = alpha_plot)
        plt.plot(angle_array[1:], likelihood_angle_lensing_noise_foregrounds100,label=r'Lensed + noise + $\frac{foregrounds}{100}$ in data not model',alpha = alpha_plot)
        plt.plot(angle_array[1:], likelihood_angle_lensing_noise_foregrounds1000,label=r'Lensed + noise + $\frac{foregrounds}{1000}$ in data not model',alpha = alpha_plot)

        plt.axvline(x = data_angle.value,linestyle=':' ,label='Input rotation angle',color = 'k')
        plt.legend(title = 'Included in model:', loc='upper left', fontsize = 'small', bbox_to_anchor=(1, 0.5 ))
        plt.title('Likelihood Estimation for different models and data')
        plt.xlabel('Rotation angle (degree)')
        plt.ylabel('Likelihood')
        plt.show()

spectra_presentation = 0
if spectra_presentation:
    # beam_array = np.linspace(0,30,4) *u.arcmin
    # w_inv_array = (np.array([0,1,2]) * u.uK * u.arcmin)**2
    # spectra_dict[ (w_inv, theta_fwhm) ]
    # plotpro.spectra({((0*(u.uK * u.arcmin)**2,30*u.arcmin)):spectra_dict[(0*(u.uK * u.arcmin)**2,30*u.arcmin)],\
    # ((1*(u.uK * u.arcmin)**2,30*u.arcmin)):spectra_dict[(1*(u.uK * u.arcmin)**2,30*u.arcmin)],\
    # ((4*(u.uK * u.arcmin)**2,30*u.arcmin)):spectra_dict[(4*(u.uK * u.arcmin)**2,30*u.arcmin)]})
    # plt.show()
    plotpro.spectra({'normal power spectra':spectra_dict[(0*u.deg,'unlensed')]})
    plt.show()
    plotpro.spectra({'power spectra rotated by {}'.format(imposed_rotation_angle*u.deg):spectra_dict[(imposed_rotation_angle*u.deg,'unlensed')]})
    plt.show()
    plotpro.spectra({'normal power spectra':spectra_dict[(0*u.deg,'unlensed')],'power spectra rotated by {}'.format(imposed_rotation_angle*u.deg):spectra_dict[(imposed_rotation_angle*u.deg,'unlensed')]})
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



# likelihood_lens_cov.append( lib.likelihood(spectra_dict[ (((angle,'unlensed'), (w_inv, 30. * u.arcmin)),'foregrounds') ], spectra_dict[(((data_angle,'unlensed'), (w_inv, 30. * u.arcmin)),'foregrounds')] ) )

# test_noise = lib.get_noise_spectrum(100 * (u.uK*u.uK * u.arcmin*u.arcmin), 30. * u.arcmin, l_max)
# l_norm = np.arange(l_max+1)
# test_noise = test_noise * l_norm * (l_norm+1) / (2*np.pi)
# test_noise = lib.get_noise_spectra(4 * (u.uK*u.uK * u.rad*u.rad), 30. * u.arcmin, l_max)


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





# test_spectra = copy.deepcopy(spectra_dict[((angle, (w_inv, 30. * u.arcmin)),'foregrounds')])
# test_spectra[:,4] =copy.deepcopy(spectra_dict[((angle, (w_inv, 30. * u.arcmin)),'foregrounds')])[:,4]  + test_noise

# print('angle',angle)
# print('test_spectra',test_spectra)
# print('spectra_dict[(angle, (w_inv, 30. * u.arcmin))]',spectra_dict[(angle, (w_inv, 30. * u.arcmin))])
# for k in range(6):
#     test_spectra[:,k] += test_noise

# likelihood_angle =                                     likelihood_angle #(likelihood_no_noise -min(likelihood_no_noise))/ (max(likelihood_no_noise)-min(likelihood_no_noise))
# likelihood_angle_lensing =                             likelihood_angle_lensing #(likelihood_no_foregrounds -min(likelihood_no_foregrounds))/ (max(likelihood_no_foregrounds)-min(likelihood_no_foregrounds))


# likelihood_angle_lensing =                              lib.likelihood_normalisation(np.exp(-np.array(lib.likelihood_normalisation(likelihood_angle_lensing)))) #(likelihood_no_foregrounds -min(likelihood_no_foregrounds))/ (max(likelihood_no_foregrounds)-min(likelihood_no_foregrounds))

# OLD VERSION :
# likelihood_angle_lensing_noise =                        lib.likelihood_normalisation(np.exp(-np.array(lib.likelihood_normalisation(likelihood_angle_lensing_noise)))) #(likelihood_noise -min(likelihood_noise))/ (max(likelihood_noise)-min(likelihood_noise))
# likelihood_angle_lensing_noise_foregrounds =            lib.likelihood_normalisation(np.exp(-np.array(lib.likelihood_normalisation(likelihood_angle_lensing_noise_foregrounds)))) #(likelihood_lens -min(likelihood_lens))/ (max(likelihood_lens)-min(likelihood_lens))
# likelihood_angle_lensing_noise_foregrounds_100uKnoise = lib.likelihood_normalisation(np.exp(-np.array(lib.likelihood_normalisation(likelihood_angle_lensing_noise_foregrounds_100uKnoise)))) #(likelihood_test -min(likelihood_test))/ (max(likelihood_test)-min(likelihood_test))

# likelihood_lens_cov =       np.exp(-np.array(lib.likelihood_normalisation(likelihood_lens_cov))) #(likelihood_lens_cov -min(likelihood_lens_cov))/ (max(likelihood_lens_cov)-min(likelihood_lens_cov))

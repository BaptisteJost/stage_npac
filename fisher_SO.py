from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt
import plot_project as plotpro
import lib_project as lib
import V3calc as V3
from pandas import DataFrame as df
import pandas as pd
import copy

pars, results, powers = lib.get_basics(l_max=10000, raw_cl=True, ratio=0.00)

b_angle = 0.0 * u.deg
SAT = 1
LAT = 1
if SAT:
    l_max_SAT = 300
    l_max_SAT_ = 300
    l_min_SAT = 30

    SAT_pure_ps = powers['total'][:l_max_SAT]

    fsky_SAT = 0.1
    telescope = 'SAT'
    SAT_yrs_LF = 1

if LAT:
    l_max_SAT = 5000
    l_min_SAT = 100

    SAT_pure_ps = powers['total'][:l_max_SAT]

    fsky_SAT = 0.4
    telescope = 'LAT'

SAT_noise_dict = {}
for i in [0, 1, 2]:
    if SAT:
        print('SAT')
        for j in [0, 1, 2]:
            print('j', j)
            noise_str = '{}'.format(i)+'{}'.format(j)
            print('le noise string = ', len(noise_str))
            noise_nl = V3.so_V3_SA_noise(i, j, SAT_yrs_LF, fsky_SAT,
                                         l_max_SAT)[1]
            # except ValueError:
            # print('oups')

            # else:
            #     print('Not a telescope')
            noise_cl = lib.get_cl_noise(noise_nl)[0, 0]
            SAT_noise_dict[noise_str] = np.append([0, 0], noise_cl)
    if LAT:
        print('LAT')

        noise_str = '{}'.format(i)
        print('d')
        # try:
        noise_nl = V3.so_V3_LA_noise(i, fsky_SAT, l_max_SAT)[2]
        print('Shape noise_nl = ', np.shape(noise_nl))
        noise_cl = lib.get_cl_noise(noise_nl)[0, 0]

        SAT_noise_dict[noise_str] = np.append([0, 0], noise_cl)


del noise_nl

ls = np.arange(SAT_noise_dict[
    list(SAT_noise_dict.keys())[0]].shape[0])
for key in SAT_noise_dict:
    plt.plot(ls, lib.get_normalised_cl(SAT_noise_dict[key]), label=key)
plt.plot(ls, lib.get_normalised_cl(SAT_pure_ps[:, 1]), label='pure EE')
plt.yscale('log')
plt.legend()
plt.show()


SAT_rot = lib.cl_rotation(SAT_pure_ps, b_angle)

truncated_fisher, truncated_fisher_element_array =\
    lib.truncated_fisher_angle(SAT_pure_ps, b_angle, SAT_rot, f_sky=fsky_SAT,
                               return_elements=True, raw_cl=True,
                               raw_cl_rot=True)


fisher_element_dict = {'no noise': truncated_fisher_element_array}


noisy_spectra_dict = {'no noise': SAT_rot}


for key in SAT_noise_dict:

    noisy_cmb_spectra = copy.deepcopy(SAT_rot.T)

    noisy_cmb_spectra[0] += copy.deepcopy(SAT_noise_dict[key])/2
    noisy_cmb_spectra[1] += copy.deepcopy(SAT_noise_dict[key])
    noisy_cmb_spectra[2] += copy.deepcopy(SAT_noise_dict[key])
    noisy_spectra_dict[key] = copy.deepcopy(noisy_cmb_spectra.T)

    noisy_cmb_spectra = 0

    truncated_fisher, fisher_element_dict[key] =\
        copy.deepcopy(lib.truncated_fisher_angle(SAT_pure_ps, b_angle,
                                                 noisy_spectra_dict[key],
                                                 f_sky=fsky_SAT,
                                                 return_elements=True,
                                                 raw_cl=True, raw_cl_rot=True))


ls = np.arange(fisher_element_dict[
    list(fisher_element_dict.keys())[0]].shape[1])

for key in SAT_noise_dict:
    plt.plot(ls, fisher_element_dict[key][4], label=key)
plt.plot(ls, lib.get_normalised_cl(SAT_pure_ps).T[1], label='pure EE')
plt.title('fisher EB element with noise ')
plt.legend()
plt.xlabel('ell')
plt.ylabel('C_ell')
plt.xscale('log')
plt.yscale('log')
plt.show()

for key in SAT_noise_dict:
    plt.plot(ls, noisy_spectra_dict[key].T[4] / SAT_noise_dict[key], label=key)
plt.title('signal sur bruit EB')
plt.legend()
plt.xlabel('ell')
plt.ylabel('C_ell')
plt.xscale('log')
plt.yscale('log')
plt.show()


sigma_dict_EB = {'no noise': 1/np.sqrt(sum(
    fisher_element_dict['no noise'][4][l_min_SAT:]))}

sigma_dict_EE = {'no noise': 1/np.sqrt(sum(
    fisher_element_dict['no noise'][4][l_min_SAT:]))}

sigma_dict_BB = {'no noise': 1/np.sqrt(sum(
    fisher_element_dict['no noise'][4][l_min_SAT:]))}

sigma_element_EB = {'no noise': 1/np.sqrt(fisher_element_dict['no noise'][4])}
sigma_element_EE = {'no noise': 1/np.sqrt(fisher_element_dict['no noise'][1])}
sigma_element_BB = {'no noise': 1/np.sqrt(fisher_element_dict['no noise'][2])}


for key in SAT_noise_dict:
    fisher_sum_EB = sum(fisher_element_dict[key][4][l_min_SAT:])
    sigma_dict_EB[key] = 1/np.sqrt(fisher_sum_EB)
    sigma_element_EB[key] = 1/np.sqrt(fisher_element_dict[key][4])

    fisher_sum_EE = sum(fisher_element_dict[key][1][l_min_SAT:])
    sigma_dict_EE[key] = 1/np.sqrt(fisher_sum_EE)
    sigma_element_EE[key] = 1/np.sqrt(fisher_element_dict[key][1])

    fisher_sum_BB = sum(fisher_element_dict[key][2][l_min_SAT:])
    sigma_dict_BB[key] = 1/np.sqrt(fisher_sum_BB)
    sigma_element_BB[key] = 1/np.sqrt(fisher_element_dict[key][2])


ls = np.arange(sigma_element_EB[
    list(sigma_element_EB.keys())[0]].shape[0])

for key in sigma_element_EB:
    plt.plot(ls, sigma_element_EB[key], label=key)
plt.title('sigma element EB')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlabel('ell')
plt.ylabel('1/sqrt(fisher_ell)')
plt.show()

for key in fisher_element_dict:
    print('KEY = ', key)
    if len(key) == 2:
        print('KEY 2')
        plotpro.cumulative_error(fisher_element_dict[key][4][:l_max_SAT_],
                                 label=key, l_min=l_min_SAT)
    else:
        print('KEY ELSE')
        plotpro.cumulative_error(fisher_element_dict[key][4],
                                 label=key, l_min=l_min_SAT)
plt.title('cumulative error on alpha={} with {} model noises'.format(
    b_angle, telescope))
plt.show()


fig, ax = plt.subplots(2, 2, figsize=(12, 12))
ax[0, 0].set_title(r'$EE$')
ax[1, 1].set_title(r'$BB$')
ax[0, 1].set_title(r'$EB$')
for key in sigma_element_EB:
    color = next(ax[1, 1]._get_lines.prop_cycler)['color']

    ax[0, 0].plot(ls, sigma_element_EE[key], label=key, color=color)
    ax[1, 1].plot(ls, sigma_element_BB[key], label=key, color=color)
    ax[0, 1].plot(ls, sigma_element_EB[key], label=key, color=color)
for ax_ in ax.reshape(-1):
    ax_.set_xscale('log')
    ax_.set_yscale('log')
    ax_.set_xlabel(r'$\ell$')
    ax_.set_ylabel(r'$C_{\ell}$')
ax[1, 1].legend(borderaxespad=0., loc='upper left',
                bbox_to_anchor=(-0.2, 1.5))
plt.show()


if SAT:
    sensitivity_mode = ['threshold', 'baseline', 'goal']
    one_over_f_mode = ['pessimistic', 'optimistic', 'none']
    sigma_array = []
    sigma_array = [[sigma_dict_EB['{}'.format(i)+'{}'.format(j)]
                    for i in [0, 1, 2]] for j in [0, 1, 2]]
    sigma_array = np.array(sigma_array)

    pd.options.display.float_format = '{:.4e}'.format
    sigma_df = df(sigma_array, columns=sensitivity_mode, index=one_over_f_mode)
    sigma_df = sigma_df.rename_axis('sensitivity_mode', axis='columns')
    sigma_df = sigma_df.rename_axis('one_over_f_mode', axis='index')
    print('\n sigma_df = ')
    print(sigma_df, '\n')

if LAT:
    sensitivity_mode = ['threshold', 'baseline', 'goal']
    sigma_array = []
    sigma_array = [sigma_dict_EB['{}'.format(i)]
                   for i in [0, 1, 2]]
    sigma_array = np.array(sigma_array)

    pd.options.display.float_format = '{:.4e}'.format
    sigma_df = df(sigma_array, index=sensitivity_mode)
    sigma_df = sigma_df.rename_axis('sensitivity_mode', axis='index')

    print('\n sigma_df = ')
    print(sigma_df, '\n')

fisher_element_dict_T = {}
for key in fisher_element_dict:
    fisher_element_dict_T[key] = fisher_element_dict[key][:-1].T


plotpro.spectra(fisher_element_dict_T)
plt.title('fisher element dict')
plt.show()
print('END')


"""===================================PURGATORY============================="""
'''

cl_deriv = lib.cl_rotation_derivative(SAT_pure_ps, b_angle)

for key in SAT_noise_dict:
    plt.plot(SAT_noise_dict[key], label=key)
plt.title('SAT noise')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()key

%%%%%%%%%%%%%%%%%%%

cl_number = len(SAT_pure_ps[:, 0])
SAT_4_5_zero = np.append(SAT_pure_ps.T,
                         np.array([[0]*cl_number, [0]*cl_number]), 0).T

%%%%%%%%%%%%%%%%%%%

fish_m = lib.fisher_manual(cl_rot, f_sky=fsky_SAT, raw_cl=True)

fish_b, total_fisher_element = lib.fisher_angle(
    cl_rot, b_angle, cl_rot=cl_rot,
    f_sky=fsky_SAT,
    return_elements=True,
    raw_cl=True, raw_cl_rot=True)

ls = np.arange(total_fisher_element.shape[0])

plt.plot(ls, total_fisher_element)
plt.title('total fisher no noise')
plt.xscale('log')
plt.yscale('log')
plt.show()

print('fish_m = ', fish_m)
print('fish_b = ', fish_b)
print('diff = ', fish_m - fish_b)

print('fish_m = ', 1/np.sqrt(fish_m))
print('fish_b = ', 1/np.sqrt(fish_b))

%%%%%%%%%%%%%%%%%%%

for key in SAT_noise_dict:
    plt.plot(SAT_noise_dict[key], label=key)
plt.title('SAT noise')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()

%%%%%%%%%%%%%%%%%%%%%

plotpro.spectra(noisy_spectra_dict)
plt.title('spectra with noise')
plt.show()


# exit()
trunc_cov_plot = lib.get_truncated_covariance_matrix(
    noisy_spectra_dict['00'])[4]
plt.plot(trunc_cov_plot[0, 0], label='EE')
plt.plot(trunc_cov_plot[1, 1], label='BB')
plt.plot(trunc_cov_plot[0, 1], label='EB')
plt.legend()
plt.title('covariance matrix')
plt.xscale('log')
plt.yscale('log')
plt.show()

trunc_cov_inv_plot = np.linalg.inv(trunc_cov_plot.T[2:]).T
print('shape trunc_cov_inv_plot', np.shape(trunc_cov_inv_plot))
plt.plot(trunc_cov_inv_plot[0, 0], label='EE')
plt.plot(trunc_cov_inv_plot[1, 1], label='BB')
plt.plot(trunc_cov_inv_plot[0, 1], label='EB')
plt.plot(-trunc_cov_inv_plot[0, 0], '--', label='EE')
plt.plot(-trunc_cov_inv_plot[1, 1], '--', label='BB')key
plt.plot(-trunc_cov_inv_plot[0, 1], '--', label='EB')
plt.legend()
plt.title('inverse covariance matrix')
plt.xscale('log')
plt.yscale('log')
plt.show()

deriv_cl = lib.cl_rotation_derivative(SAT_pure_ps, b_angle)
trunc_deriv_plot = lib.get_truncated_covariance_matrix(deriv_cl)[4]
plt.plot(trunc_deriv_plot[0, 0], label='EE')
plt.plot(trunc_deriv_plot[1, 1], label='BB')
plt.plot(trunc_deriv_plot[0, 1], label='EB')
# plt.plot(-trunc_deriv_plot[0, 0], '--', label='EE')
# plt.plot(-trunc_deriv_plot[1, 1], '--', label='BB')
# plt.plot(-trunc_deriv_plot[0, 1], '--', label='EB')
plt.legend()
plt.title('derivativ matrix')
plt.xscale('log')
plt.yscale('log')
plt.show()


%%%%%%%%%%%%%%%%%%%%%%%%%%

fisher_element_noise = {'no noise': total_fisher_element}

for key in SAT_noise_dict:
    fisher_noise, fisher_element_noise[key] = \
        copy.deepcopy(lib.fisher_angle(SAT_pure_ps, b_angle,
                                       cl_rot=noisy_spectra_dict[key],
                                       f_sky=fsky_SAT, return_elements=True,
                                       raw_cl=True, raw_cl_rot=True))

ls = np.arange(fisher_element_noise[
        list(fisher_element_noise.keys())[0]].shape[0])
# for key in fisher_element_noise:
plt.title('fisher element noise 00 TEB')
plt.plot(ls, fisher_element_noise['00'])
plt.plot(ls, -fisher_element_noise['00'], '--')
plt.xscale('log')
plt.yscale('log')
plt.show()

%%%%%%%%%%%%%%%%%%%%%%%%%

plotpro.spectra({'normal': lib.get_normalised_cl(SAT_pure_ps)})
plt.title('cmb spectra, no noise')
plt.show()

%%%%%%%%%%%%%%%%%%%%%%%%

fisher_dict_EB = {'no noise': sum(
    fisher_element_dict['no noise'][4][l_min_SAT:])}
fisher_dict_EB[key] = sum(fisher_element_dict[key][4][l_min_SAT:])

%%%%%%%%%%%%%%%%%%%%%%
'''

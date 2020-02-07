
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt
import plot_project as plotpro
import lib_project as lib
import V3calc as V3
import time
from pandas import DataFrame as df
import pandas as pd
import copy

pars, results, powers = lib.get_basics(l_max=2500, raw_cl=True, ratio=0.07)

l_max_SAT = 1000

SAT_pure_ps = powers['total'][:l_max_SAT]

fsky_SAT = 0.1

b_angle = 0.0 * u.deg

print('SAT_pure_ps shape =', np.shape(SAT_pure_ps))


SAT_noise_dict = {}
for i in [0, 1, 2]:
    for j in [0, 1, 2]:
        noise_str = '{}'.format(i)+'{}'.format(j)
        print('noise_str = ', noise_str)
        print('i;j = ', i, j)
        noise_nl = V3.so_V3_SA_noise(i, j, 1, fsky_SAT, l_max_SAT)[1]
        print('noise spectrun shape', np.shape(noise_nl))
        noise_cl = lib.get_cl_noise(noise_nl)[0, 0]
        SAT_noise_dict[noise_str] = np.append([0, 0], noise_cl)
        # SAT_noise_dict[noise_str] = np.array([np.append([0, 0],
        #                                                 noise[k])
        #                                       for k in [0, 1, 2, 3, 4, 5]]).T
print('noise_cl shape = ', np.shape(noise_cl))

del noise_nl


# WARNING: check the arguments of sat noise
for key in SAT_noise_dict:
    plt.plot(SAT_noise_dict[key], label=key)
plt.title('SAT noise')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()


print('SAT_noise_dict[00] type = ',
      type(SAT_noise_dict['00']))


print('SAT_noise_dict[00][30:] shape = ',
      np.shape(SAT_noise_dict['00']))

print('SAT_noise_dict[00][0] shape = ',
      np.shape(SAT_noise_dict['00'][0]))

print('SAT_noise_dict[00][0][30:] shape = ',
      np.shape(SAT_noise_dict['00'][30:]))


cl_rot = lib.cl_rotation(SAT_pure_ps, b_angle)

print('cl_rot SHAPE %%%%% = ', np.shape(cl_rot))

cl_deriv = lib.cl_rotation_derivative(SAT_pure_ps, b_angle)


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

SAT_rot = lib.cl_rotation(SAT_pure_ps, b_angle)

print('shape (SAT_pure_ps)', np.shape(SAT_pure_ps))
cl_number = len(SAT_pure_ps[:, 0])

start_zero = time.time()
SAT_4_5_zero = np.append(SAT_pure_ps.T,
                         np.array([[0]*cl_number, [0]*cl_number]), 0).T
end_zero = time.time()
print('TIME ZERO = ', end_zero - start_zero)

print('SAT_4_5_zero shape = ', np.shape(SAT_4_5_zero))
truncated_fisher, truncated_fisher_element_array =\
    lib.truncated_fisher_angle(SAT_pure_ps, b_angle, SAT_4_5_zero,
                               f_sky=fsky_SAT, return_elements=True,
                               raw_cl=True, raw_cl_rot=True)
print('===================================================================')
print('truncater_fisher shape = ', np.shape(truncated_fisher))
print('truncated_fisher_element_array shape = ',
      np.shape(truncated_fisher_element_array))
print("truncated_fisher[-1]", truncated_fisher[-1])
print('DIFF au cas ou = ', truncated_fisher[-1] - fish_m)
print('DIFF au cas ou 2= ', truncated_fisher[-1] - fish_b)

# fisher_dict = {'no noise': truncated_fisher}
fisher_element_dict = {'no noise': truncated_fisher_element_array}

fisher_element_noise = {'no noise': total_fisher_element}


noisy_spectra_dict = {'no noise': SAT_4_5_zero}

for key in SAT_noise_dict:
    print('THE KEEEEEYYYYY : ', key)
    noisy_cmb_spectra = cl_rot.T
    print('noisy_spectra_dict shape before =', np.shape(noisy_cmb_spectra))
    noisy_cmb_spectra[0] = noisy_cmb_spectra[0] + SAT_noise_dict[key]/2
    noisy_cmb_spectra[1] = noisy_cmb_spectra[1] + SAT_noise_dict[key]
    noisy_cmb_spectra[2] = noisy_cmb_spectra[2] + SAT_noise_dict[key]
    noisy_spectra_dict[key] = copy.deepcopy(noisy_cmb_spectra.T)
    print('noisy_spectra_dict shape before =', np.shape(noisy_cmb_spectra))
    del noisy_cmb_spectra
    print('noisy_spectra_dict', noisy_spectra_dict)
    print('shape noisy_spectra_dict[key] after =', np.shape(noisy_spectra_dict[key]))
    print(key)
    print('noisy_spectra_dict[key] shape = ',
          np.shape(noisy_spectra_dict[key]))
    print('noisy_spectra_dict[key] type = ',
          type(noisy_spectra_dict[key]))
    print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
    print('COVARIANCE', key)

    truncated_fisher, fisher_element_dict[key] =\
        copy.deepcopy(lib.truncated_fisher_angle(SAT_pure_ps, b_angle,
                                                 noisy_spectra_dict[key], f_sky=fsky_SAT,
                                                 return_elements=True, raw_cl=True,
                                                 raw_cl_rot=True))
    print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')
    fisher_noise, fisher_element_noise[key] = \
        copy.deepcopy(lib.fisher_angle(SAT_pure_ps, b_angle, cl_rot=noisy_spectra_dict[key],
                                       f_sky=fsky_SAT,
                                       return_elements=True,
                                       raw_cl=True, raw_cl_rot=True))

print('OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO')
print('cl_rot[500][2]', cl_rot[30][2])
print('noisy_spectra_dict[00][500][2]', noisy_spectra_dict['00'][30][2])
print('noisy_spectra_dict[22][500][2]', noisy_spectra_dict['22'][30][2])
print('SAT_noise_dict[00]', SAT_noise_dict['00'][30])
print('SAT_noise_dict[22]', SAT_noise_dict['22'][30])

# exit()
trunc_cov_plot = lib.get_truncated_covariance_matrix(noisy_spectra_dict['00'])[4]
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
plt.plot(-trunc_cov_inv_plot[1, 1], '--', label='BB')
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

# exit()

ls = np.arange(fisher_element_dict[list(fisher_element_dict.keys())[0]].shape[1])
for key in SAT_noise_dict:
    plt.plot(ls, fisher_element_dict['00'][-1], label=key)
plt.title('fisher element with noise TEB')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.show()

ls = np.arange(fisher_element_noise[list(fisher_element_noise.keys())[0]].shape[0])
# for key in fisher_element_noise:

plt.plot(ls, fisher_element_noise['00'])
plt.plot(ls, -fisher_element_noise['00'], '--')
plt.title('fisher element noise 00 ')
plt.xscale('log')
plt.yscale('log')
plt.show()

print('test')

print("truncated_fisher", truncated_fisher)
print('len fisher_element_dict =', len(fisher_element_dict))
print('shape fisher_element_dict[00] =', np.shape(fisher_element_dict['00']))
print('IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII')
plotpro.spectra(noisy_spectra_dict)
plt.title('spectra with noise')
plt.show()
print("TSET PRINT")
# fig, ax = plt.subplots(3, 2, figsize=(12, 12))
# plotpro.one_spectrum({'normal': SAT_4_5_zero}, 'TT', ax)
plotpro.spectra({'normal': lib.get_normalised_cl(SAT_pure_ps)})
plt.title('cmb spectra, no noise')
plt.show()
print('QWERT')
fisher_dict_EB = {'no noise': sum(fisher_element_dict['no noise'][4][30:])}
sigma_dict_EB = {'no noise': 1/np.sqrt(fisher_dict_EB['no noise'])}
for key in SAT_noise_dict:
    fisher_dict_EB[key] = sum(fisher_element_dict[key][4][30:])
    sigma_dict_EB[key] = 1/np.sqrt(fisher_dict_EB[key])

print(fisher_dict_EB)
print(sigma_dict_EB)

# sensitivity_mode = {0: 'threshold', 1: 'baseline', 2: 'goal'}
# one_over_f_mode = {0: 'pessimistic', 1: 'optimistic', 2: 'none'}
sensitivity_mode = ['threshold', 'baseline', 'goal']
one_over_f_mode = ['pessimistic', 'optimistic', 'none']
sigma_array = []
sigma_array = [[sigma_dict_EB['{}'.format(i)+'{}'.format(j)]
                for i in [0, 1, 2]] for j in [0, 1, 2]]
sigma_array = np.array(sigma_array)
print('SIGMA ARRAY=', sigma_array)

pd.options.display.float_format = '{:.3e}'.format
sigma_df = df(sigma_array, columns=sensitivity_mode, index=one_over_f_mode)
sigma_df = sigma_df.rename_axis('sensitivity_mode', axis='columns')
sigma_df = sigma_df.rename_axis('one_over_f_mode', axis='index')
print('sigma_df = ')
print(sigma_df)

fisher_element_dict_T = {}
for key in fisher_element_dict:
    fisher_element_dict_T[key] = fisher_element_dict[key][:-1].T
    # fisher_element_dict_T[key] = fisher_dict_EB[key].T

    print('shape fisher_element_dict[key]', np.shape(fisher_element_dict[key][4]))
    print('shape fisher_element_dict_T[key]',
          np.shape(fisher_element_dict_T[key]))

plotpro.spectra(fisher_element_dict_T)
plt.title('fisher element dict')
plt.show()
print('END')

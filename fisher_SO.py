from fgbuster import visualization as visu
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt
import plot_project as plotpro
import lib_project as lib
import V3calc as V3
from pandas import DataFrame as df
import pandas as pd
import copy
from matplotlib.cm import get_cmap
import matplotlib.text as mtext
import matplotlib
import IPython

r = 0

pars, results, powers = lib.get_basics(l_max=10000, raw_cl=True, ratio=r)

b_angle = 0 * u.rad
SAT = 1
LAT = 1
if SAT:
    l_max_SAT = 300
    l_max_SAT_ = 300

    l_min_SAT = 30
    l_min_SAT_ = 30

    SAT_pure_ps = powers['total'][:l_max_SAT]

    fsky_SAT = 0.1
    fsky_SAT_ = 0.1

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
            print('i', i)
            print('j', j)
            noise_str = '{}'.format(i)+'{}'.format(j)
            print('le noise string = ', len(noise_str))
            noise_nl = V3.so_V3_SA_noise(i, j, SAT_yrs_LF, fsky_SAT_,
                                         l_max_SAT)[1]
            ell = V3.so_V3_SA_noise(i, j, SAT_yrs_LF, fsky_SAT_,
                                    l_max_SAT)[0]
            print('SAT_yrs_LF', SAT_yrs_LF)
            print('fsky_SAT', fsky_SAT)
            print('l_max_SAT', l_max_SAT)
            print('noise nl = ', noise_nl)
            print('ell =', ell)
            print('shape noise nl', np.shape(noise_nl))
            noise_cl = lib.get_cl_noise(noise_nl, instrument='SAT')[0, 0]
            print('shape noise cl = ', np.shape(noise_cl))
            SAT_noise_dict[noise_str] = np.append([0, 0], noise_cl)
    if LAT:
        print('LAT')

        noise_str = '{}'.format(i)
        print('d')
        # try:
        noise_nl = V3.so_V3_LA_noise(i, fsky_SAT, l_max_SAT)[2]
        print('Shape noise_nl = ', np.shape(noise_nl))
        noise_cl = lib.get_cl_noise(noise_nl, instrument='LAT')[0, 0]

        SAT_noise_dict[noise_str] = np.append([0, 0], noise_cl)

plt.plot(lib.get_normalised_cl(SAT_pure_ps[:, 2]))
plt.xscale('log')
plt.yscale('log')
plt.show()

del noise_nl
print('NOISE SAT =', SAT_noise_dict['00'])
print(' shape NOISE SAT = ', np.shape(SAT_noise_dict['00']))
# exit()
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
    if len(key) == 2:
        print('coucou')
        print(key)

        truncated_fisher, fisher_element_dict[key] =\
            copy.deepcopy(lib.truncated_fisher_angle(SAT_pure_ps, b_angle,
                                                     noisy_spectra_dict[key],
                                                     f_sky=fsky_SAT_,
                                                     return_elements=True,
                                                     raw_cl=True, raw_cl_rot=True))
        if key == '00':
            arg1 = copy.deepcopy(SAT_pure_ps)
            arg2 = b_angle,
            arg3 = copy.deepcopy(noisy_spectra_dict[key])
            arg4 = fsky_SAT_
            arg5 = True
            arg6 = True
            arg7 = True
            save_fishel = copy.deepcopy(fisher_element_dict[key])
            print('diff 0 =', save_fishel[4] - fisher_element_dict['00'][4])

    elif key == 'no noise':
        truncated_fisher, fisher_element_dict[key] =\
            copy.deepcopy(lib.truncated_fisher_angle(SAT_pure_ps, b_angle,
                                                     noisy_spectra_dict[key],
                                                     f_sky=1,
                                                     return_elements=True,
                                                     raw_cl=True, raw_cl_rot=True))

    else:
        print('else key', key)
        truncated_fisher, fisher_element_dict[key] =\
            copy.deepcopy(lib.truncated_fisher_angle(SAT_pure_ps, b_angle,
                                                     noisy_spectra_dict[key],
                                                     f_sky=fsky_SAT,
                                                     return_elements=True,
                                                     raw_cl=True, raw_cl_rot=True))

print('diff 0.5 =', save_fishel[4] - fisher_element_dict['00'][4])

ls = np.arange(fisher_element_dict[
    list(fisher_element_dict.keys())[0]].shape[1])

# for key in SAT_noise_dict:
#     plt.plot(ls, fisher_element_dict[key][4], label=key)
# print('diff 1 =', save_fishel[4] - fisher_element_dict['00'][4])
# plt.plot(ls, lib.get_normalised_cl(SAT_pure_ps).T[1], label='pure EE')
# plt.title('fisher EB element with noise ')
# plt.legend()
# plt.xlabel('ell')
# plt.ylabel('C_ell')
# plt.xscale('log')
# plt.yscale('log')
# plt.show()

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
print('diff 2 =', save_fishel[4] - fisher_element_dict['00'][4])


for key in SAT_noise_dict:
    if len(key) == 2:
        fisher_sum_EB = sum(fisher_element_dict[key][4][l_min_SAT_:l_max_SAT_])
        fisher_sum_EE = sum(fisher_element_dict[key][1][l_min_SAT_:l_max_SAT_])
        fisher_sum_BB = sum(fisher_element_dict[key][2][l_min_SAT_:l_max_SAT_])
        print('diff 3 =', save_fishel[4] - fisher_element_dict['00'][4])

    else:
        fisher_sum_EB = sum(fisher_element_dict[key][4][l_min_SAT:])
        fisher_sum_EE = sum(fisher_element_dict[key][1][l_min_SAT:])
        fisher_sum_BB = sum(fisher_element_dict[key][2][l_min_SAT:])
        print('diff 4 =', save_fishel[4] - fisher_element_dict['00'][4])

    sigma_dict_EB[key] = 1/np.sqrt(fisher_sum_EB)
    sigma_element_EB[key] = 1/np.sqrt(fisher_element_dict[key][4])
    print('diff 5 =', save_fishel[4] - fisher_element_dict['00'][4])

    sigma_dict_EE[key] = 1/np.sqrt(fisher_sum_EE)
    sigma_element_EE[key] = 1/np.sqrt(fisher_element_dict[key][1])

    sigma_dict_BB[key] = 1/np.sqrt(fisher_sum_BB)
    sigma_element_BB[key] = 1/np.sqrt(fisher_element_dict[key][2])
    print('diff 6 =', save_fishel[4] - fisher_element_dict['00'][4])


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

SAT_map = get_cmap('autumn')
LAT_map = get_cmap('viridis')
SAT_counter = 0
LAT_counter = 0


class LegendTitle(object):
    def __init__(self, text_props=None):
        self.text_props = text_props or {}
        super(LegendTitle, self).__init__()

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        title = mtext.Text(
            x0, y0,  orig_handle, usetex=False, **self.text_props)
        handlebox.add_artist(title)
        return title


for key in fisher_element_dict:
    if len(key) == 2:
        color = SAT_map(SAT_counter/10)
        plotpro.cumulative_error(fisher_element_dict[key][4][:l_max_SAT_],
                                 label='sensitivity mode :'+key[0] +
                                 ' and 1/f :' + key[1],
                                 l_min=l_min_SAT_, dotted=True,
                                 color=color)
        print('diff 7 =', save_fishel[4] - fisher_element_dict['00'][4])

        SAT_counter += 1
    elif key == 'no noise':
        plotpro.cumulative_error(fisher_element_dict[key][4],
                                 label=key, l_min=l_min_SAT_, color='black')
    else:
        color = LAT_map(1/2+1/6*LAT_counter)
        plotpro.cumulative_error(fisher_element_dict[key][4],
                                 label='sensitivity mode :'+key[0],
                                 l_min=l_min_SAT, color=color)
        LAT_counter += 1
print('diff 8 =', save_fishel[4] - fisher_element_dict['00'][4])

plt.title(r'Cumulative error on $ \alpha = $ {}$^\circ $ with SAT and LAT, and $ r = ${}'.
          format(b_angle.value, r))
handles, labels = plt.gca().get_legend_handles_labels()
order = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 4, 8, 12]
print([labels[idx] for idx in order])
new_handles = [handles[idx] for idx in order]
new_handles.insert(1, 'SAT')
new_handles.insert(11, 'LAT')
print('new_handles', new_handles)
new_labels = [labels[idx] for idx in order]
new_labels.insert(1, '')
new_labels.insert(11, '')
print('new_labels', new_labels)

# new_handles.append('Title 1')
# new_labels.append('')
# plt.legend()
# print(plt.legend.get_legend_handler_map)
print('')
print('HANDLER MAP=',
      matplotlib.legend.Legend.get_legend_handler_map(plt.legend()))
plt.legend(new_handles, new_labels,
           handler_map={str: LegendTitle({'fontsize': 11})})
plt.grid(b=True, linestyle=':')
plt.show()
print('handles', handles)
print('labels', labels)


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
    print(sigma_df.to_latex())

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
    print(sigma_df.to_latex())

fisher_element_dict_T = {}
for key in fisher_element_dict:
    fisher_element_dict_T[key] = fisher_element_dict[key][: -1].T
print('diff 9 =', save_fishel[4] - fisher_element_dict['00'][4])


plotpro.spectra(fisher_element_dict_T)
plt.title('fisher element dict')
plt.show()


pars, results, powers_r1 = lib.get_basics(l_max=10000, raw_cl=True, ratio=1)
cl_r1 = powers_r1['unlensed_total'][:l_max_SAT]
key_corner = '00'
cov_matrix = np.array(
    [[noisy_spectra_dict[key_corner][l_min_SAT_:l_max_SAT_, 1], noisy_spectra_dict[key_corner][l_min_SAT_:l_max_SAT_, 4]],
     [noisy_spectra_dict[key_corner][l_min_SAT_:l_max_SAT_, 4], noisy_spectra_dict[key_corner][l_min_SAT_:l_max_SAT_, 2]]])

deriv1 = lib.cl_rotation_derivative(SAT_pure_ps, b_angle)

deriv_matrix1 = np.array(
    [[deriv1[l_min_SAT_:l_max_SAT_, 1], deriv1[l_min_SAT_:l_max_SAT_, 4]],
     [deriv1[l_min_SAT_:l_max_SAT_, 4], deriv1[l_min_SAT_:l_max_SAT_, 2]]])

deriv_matrix2 = lib.get_dr_cov_bir_EB(cl_r1, b_angle)

fisher_alpha_err = np.array([
    [lib.fisher(cov_matrix, deriv_matrix1, 0.1), lib.fisher(
        cov_matrix, deriv_matrix1, 0.1, cov2=cov_matrix, deriv2=deriv_matrix2)],
    [lib.fisher(cov_matrix, deriv_matrix1, 0.1, cov2=cov_matrix, deriv2=deriv_matrix2),
     lib.fisher(cov_matrix, deriv_matrix2, 0.1)]
])

sigma2 = np.linalg.inv(fisher_alpha_err)
visu.corner_norm([b_angle.value, r], sigma2, labels=[r'$\alpha$', r'$r$'])
# plt.title(r'True $\alpha =${} $r =${}'.format(b_angle, r))
plt.show()
IPython.embed()


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

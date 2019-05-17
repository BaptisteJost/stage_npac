import sys, platform, os
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import camb
from camb import model, initialpower
import lib_project as lib
from mpl_toolkits.mplot3d import axes3d
import matplotlib.patches as mpatches
from matplotlib import cm
from numpy import inf
import matplotlib.ticker as mticker


def cl_comparison(cl_unchanged, cl_rot, angle):
    #commit test
    ls = np.arange(cl_unchanged.shape[0])
    fig, ax = plt.subplots(3,2, figsize = (12,12))

    ax[0,0].plot(ls,cl_unchanged[:,0] , color='k', label = 'original power spectra')
    ax[0,0].plot(ls,cl_rot[:,0], color='b',label='power spectra rotated by {}'.format(angle))
    ax[0,0].set_title('TT')
    ax[0,0].legend( borderaxespad=0.,loc='upper center', bbox_to_anchor=(0.5, 1.5 ))

    ax[0,1].plot(ls,cl_unchanged[:,1] , color='k', linewidth = 2)
    ax[0,1].plot(ls,cl_rot[:,1] , color='b')
    ax[0,1].set_title(r'$EE$')

    ax[1,0].plot(ls,cl_unchanged[:,2] , color='k', linewidth = 2)
    ax[1,0].plot(ls,cl_rot[:,2] , color='b')
    ax[1,0].set_title(r'$BB$')

    ax[1,1].plot(ls,cl_unchanged[:,3] , color='k', linewidth = 2)
    ax[1,1].plot(ls,-cl_unchanged[:,3] , '--', color='k', linewidth = 2)
    ax[1,1].plot(ls,cl_rot[:,3] , color='b')
    ax[1,1].plot(ls,-cl_rot[:,3] , '--', color='b')
    ax[1,1].set_title(r'$TE$');

    ax[2,0].plot(ls,cl_rot[:,4] , color='b')
    ax[2,0].set_title(r'$EB$');

    ax[2,1].plot(ls,cl_rot[:,5] , color='b')
    ax[2,1].plot(ls,-cl_rot[:,5] ,  '--',color='b')
    ax[2,1].set_title(r'$TB$');
    for ax_ in ax.reshape(-1): ax_.set_xlim([2,cl_unchanged.shape[0]]);\
                                ax_.set_xscale('log');\
                                ax_.set_yscale('log'); \

    return fig, ax

def new_angle(cl_rot, angle, fig, ax):
    ls = np.arange(cl_rot.shape[0])
    ax[0,0].plot(ls,cl_rot[:,0],label='power spectra rotated by {}'.format(angle))
    ax[0,0].legend( borderaxespad=0.,loc='upper center', bbox_to_anchor=(0.5, 1.5 ))
    ax[0,1].plot(ls,cl_rot[:,1])
    ax[1,0].plot(ls,cl_rot[:,2])
    color = next(ax[1,1]._get_lines.prop_cycler)['color']
    ax[1,1].plot(ls,cl_rot[:,3], color = color)
    ax[1,1].plot(ls,-cl_rot[:,3], '--', color = color)
    ax[2,0].plot(ls,cl_rot[:,4])
    ax[2,1].plot(ls,cl_rot[:,5], color = color)
    ax[2,1].plot(ls,-cl_rot[:,5],  '--', color = color)
    for ax_ in ax.reshape(-1): ax_.set_xlim([2,cl_rot.shape[0]]); \
                                ax_.set_xscale('log');\
                                ax_.set_yscale('log');\
    return fig, ax


def error_on_angle_wrt_scale(fisher_element, label = None):
    ls = np.arange(2,len(fisher_element))
    print('1 / np.sqrt(fisher_element[2:]', 1/np.sqrt(fisher_element[2:]))
    plt.plot(ls , 1 / np.sqrt(fisher_element[2:]), label=label)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\ell$',fontsize = 16)
    plt.ylabel(r'$ \frac{1}{\sqrt{F_{\ell}}} $',fontsize = 16)
    plt.legend()

    return 0

def cumulative_error(fisher_element, label = None):
    ls = np.arange(2,len(fisher_element))

    cumulative = np.array( [ 1 / np.sqrt( sum( fisher_element[:k]) ) for k in range(len(fisher_element)) ] )
    plt.plot(ls, cumulative[2:], label=label)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\ell$',fontsize = 16)
    plt.ylabel(r'$ \frac{1}{\sqrt{ \sum_{ \ell_{min} = 2 }^{\ell}{F_{\ell}}}} $',fontsize = 16)
    plt.legend()
    return 0


def white_noise_vs_lensing(nl_spectra, powers):
    fig = plt.figure()
    ls = np.arange(powers['lensed_scalar'].shape[0])

    plt.plot(ls, powers['lensed_scalar'][:,2], label=r'$C_{\ell}^{BB}$ lensed')
    plt.plot(ls, nl_spectra[:,1], label=r'$N_{\ell}$')
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$\frac{ \ell (\ell +1) C_{\ell} }{2 \pi}$')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(2,1000)
    plt.ylim(1E-6,1E5)
    plt.legend()
    return fig



def spectra(cl_dict):
    ls = np.arange(cl_dict[ list(cl_dict.keys())[0] ].shape[0])
    fig, ax = plt.subplots(3,2, figsize = (12,12))
    ax[0,0].set_title('TT')
    ax[0,1].set_title(r'$EE$')
    ax[1,0].set_title(r'$BB$')
    ax[1,1].set_title(r'$TE$');
    ax[2,0].set_title(r'$EB$');
    ax[2,1].set_title(r'$TB$');

    for key,value in cl_dict.items():
        color = next(ax[1,1]._get_lines.prop_cycler)['color']
        ax[0,0].plot(ls,cl_dict[key][:,0],label='{}'.format(key), color = color)
        ax[0,1].plot(ls,cl_dict[key][:,1],label='{}'.format(key), color = color)
        ax[1,0].plot(ls,cl_dict[key][:,2],label='{}'.format(key), color = color)
        ax[1,1].plot(ls,cl_dict[key][:,3],label='{}'.format(key), color = color)
        ax[1,1].plot(ls,-cl_dict[key][:,3], '--', color = color)
        ax[2,0].plot(ls,cl_dict[key][:,4],label='{}'.format(key), color = color)
        ax[2,1].plot(ls,cl_dict[key][:,5],label='{}'.format(key), color = color)
        ax[2,1].plot(ls,-cl_dict[key][:,5], '--', color = color)

    for ax_ in ax.reshape(-1):
        ax_.set_xlim([2,cl_dict[ list(cl_dict.keys())[0] ].shape[0]]); \
        ax_.set_xscale('log');\
        ax_.set_yscale('log');\
        ax_.set_xlabel(r'$\ell$')
        ax_.set_ylabel(r'$C_{\ell} \frac{\ell (\ell +1)}{2 \pi}$')
    ax[0,0].legend( borderaxespad=0.,loc='upper left', bbox_to_anchor=(-0.2, 1.5 ))

    return 0

def relative_derivative(spectra_dict, key_rot):

    cl_normalised = lib.cl_normalisation( spectra_dict[key_rot] )
    cl_derivative = lib.cl_rotation_derivative(cl_normalised, key_rot )

    ls = np.arange(cl_normalised.shape[0])

    label_list = ['TT', 'EE', 'BB', 'TE', 'EB', 'TB']
    for i in range(cl_normalised.shape[1]):
        dcl_o_cl = cl_derivative[:,i] / cl_normalised[:,i]

        plt.plot(ls, dcl_o_cl , label = label_list[i] )
    plt.xlabel(r'$\ell$',fontsize = 16)
    plt.ylabel(r'$\frac{\partial_{\alpha}C_{\ell}}{C_{\ell}}$',fontsize = 16)
    plt.title( r'$ \frac{ \partial C_{\ell}} {\partial \alpha} . \frac{1}{ C_{\ell} }$ for $\alpha =$ %s as a function of $\ell$' %key_rot)

    plt.legend()
    return 0

def truncated_fisher_cumulative(fisher_trunc_array_dict, key):
    count = 0
    label_list = ['T', 'E', 'B', 'T,E', 'E,B', 'T,B','T,E,B']
    for fisher_element in fisher_trunc_array_dict[key]:
        cumulative_error(fisher_element, label = label_list[count])
        count += 1
    if key[1] == 'no noise':
        plt.title( 'cumulative error for rotation {}, and NO noise'.format(key[0]) )
    else :
        plt.title( 'cumulative error for rotation {}, and noise beam {}'.format(key[0], key[1][1]) )
    return 0

def cumulative_error_3D(fisher_element, key):

    l_min_array = np.arange(2,len(fisher_element))
    l_max_array = np.arange(2,len(fisher_element))

    slice_list = [ [ sum( fisher_element[l_min:l_max+1] )  for l_max in l_max_array ] for l_min in l_min_array]
    l_min_array , l_max_array = np.meshgrid(l_min_array , l_max_array, indexing = 'ij')

    cumulative = np.array(  1/np.sqrt(slice_list)  )
    cumulative [ cumulative == inf ] = 0
    cumulative = np.array(cumulative)

    fig1 = plt.figure()
    ax = fig1.gca(projection='3d')

    log_cumu = np.log10(cumulative)
    log_cumu [ log_cumu == inf ] = None
    log_cumu [ log_cumu == -inf ] = None
    log_cumu = np.array(log_cumu)

    surface = ax.plot_surface( np.log10(l_min_array), np.log10(l_max_array), log_cumu, cmap = cm.viridis, vmin = np.nanmin(log_cumu),vmax = np.nanmax(log_cumu), rcount = 50, ccount = 50)

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
    ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))

    ax.set_xlabel(r'$\ell_{min}$',fontsize = 16)
    ax.set_ylabel(r'$\ell_{max}$',fontsize = 16)
    ax.set_zlabel(r'$ \frac{1}{\sqrt{ \sum_{ \ell_{min} }^{\ell_{max}}{F_{\ell}}}} $',fontsize = 16)
    cbar = fig1.colorbar(surface)
    cbar.ax.set_ylabel(r'$ \frac{1}{\sqrt{ \sum_{ \ell_{min} }^{\ell_{max}}{F_{\ell}}}} $',fontsize = 16)

    fig2, ax2 = plt.subplots()
    contour = ax2.contourf(l_min_array, l_max_array, log_cumu, cmap = cm.viridis, zdir='z', offset=np.nanmin(log_cumu)-1)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel(r'$\ell_{min}$',fontsize = 16)
    ax2.set_ylabel(r'$\ell_{max}$',fontsize = 16)
    cbar = fig2.colorbar(contour)
    cbar.ax.set_ylabel(r'$ \frac{1}{\sqrt{ \sum_{ \ell_{min} }^{\ell_{max}}{F_{\ell}}}} $',fontsize = 16)

    if len(key[0])==2 :
        if key[1] == 'no noise':
            ax.set_title( 'surface plot of cumulative error for rotation {}, and NO noise, with lensing'.format(key[0][0]) )
            ax2.set_title( 'heatmap of cumulative error for rotation {}, and NO noise, with lensing'.format(key[0][0]) )
        else:
            ax.set_title( 'surface plot of cumulative error for rotation {}, and noise beam {}, with lensing'.format(key[0][0], key[0][1][1]) )
            ax2.set_title( 'heatmap of cumulative error for rotation {}, and noise beam {}, with lensing'.format(key[0][0], key[0][1][1]) )
    else:
        if key[1] == 'no noise':
            ax.set_title( 'surface plot of cumulative error for rotation {}, and NO noise'.format(key[0]) )
            ax2.set_title( 'heatmap of cumulative error for rotation {}, and NO noise'.format(key[0]) )
        else :
            ax.set_title( 'surface plot of cumulative error for rotation {}, and noise beam {}'.format(key[0], key[1][1]) )
            ax2.set_title( 'heatmap of cumulative error for rotation {}, and noise beam {}'.format(key[0], key[1][1]) )

    return fig1, fig2

def log_tick_formatter(val, pos=None):
    return "{:.2e}".format(10**val)

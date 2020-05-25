import likelihood_SO as l_SO
import class_faraday as cf
import numpy as np
import healpy as hp
from pysm import convert_units
import lib_project as lib
import copy
# import pysm
import pymaster as nmt
import IPython
import matplotlib.pyplot as plt
import time
from astropy import units as u


def binning_definition(nside, lmin=2, lmax=200, nlb=[], custom_bins=False):
    if custom_bins:
        ells = np.arange(3*nside, dtype='int32')  # Array of multipoles
        weights = (1.0/nlb)*np.ones_like(ells)  # Array of weights
        bpws = -1+np.zeros_like(ells)  # Array of bandpower indices
        i = 0
        while (nlb+1)*(i+1)+lmin < lmax:
            bpws[(nlb+1)*i+lmin:(nlb+1)*(i+1)+lmin] = i
            i += 1
        # adding a trash bin 2<=ell<=lmin
        # bpws[lmin:(nlb+1)*i+lmin] += 1
        # bpws[2:lmin] = 0
        # weights[2:lmin]= 1.0/(lmin-2-1)
        b = nmt.NmtBin(nside, bpws=bpws, ells=ells, weights=weights)
    else:
        b = nmt.NmtBin(nside, nlb=int(1./self.config['fsky']))
    return b


def get_field(mp_q, mp_u, mask_apo, purify_e=False, purify_b=True):
    # This creates a spin-2 field with both pure E and B.
    f2y = nmt.NmtField(mask_apo, [mp_q, mp_u], purify_e=purify_e, purify_b=purify_b)
    return f2y


def compute_master(f_a, f_b, wsp):
    cl_coupled = nmt.compute_coupled_cell(f_a, f_b)
    cl_decoupled = wsp.decouple_cell(cl_coupled)
    return cl_decoupled


def main():
    nsim = 1
    purify_e = False
    rotation_angle = (0.0 * u.deg).to(u.rad)
    r = 0.0
    pysm_model = 'c1s0d0'
    f_sky_SAT = 0.1

    # from config.yml
    nside = 512
    lmin = 30
    lmax = 500
    nlb = 10
    custom_bins = True
    aposize = 8.0
    apotype = 'C1'
    purify_b = True

    spectra_cl = cf.power_spectra_operation(r=r,
                                            rotation_angle=rotation_angle, powers_name='total')
    spectra_cl.get_spectra()
    spectra_cl.spectra_rotation()
    spectra_cl.get_frequencies()

    true_spectra = cf.power_spectra_obj(spectra_cl.cl_rot.spectra,
                                        spectra_cl.cl_rot.ell)
    true_spectra.ell = np.arange(lmin, lmax)

    sky_map = l_SO.sky_map(nside=nside, sky_model=pysm_model)
    sky_map.get_pysm_sky()
    sky_map.get_frequency()

    # IPython.embed()
    # sim_maps = hp.synfast(true_spectra.spectra.T, nside, lmax=lmax)
    # sim_cmb_freqmaps = np.array([sim_maps/len(spectra_cl.frequencies)
    #                              for i in range(len(spectra_cl.frequencies))])

    wsp = nmt.NmtWorkspace()
    b = binning_definition(nside, lmin=lmin, lmax=lmax,
                           nlb=nlb, custom_bins=custom_bins)
    # wsp.wsp.ncls = 4

    print('building mask ... ')
    mask = hp.read_map("outputs_22VTMDJN55_00000_00000/binary_mask_cut.fits")

    mask_apo = nmt.mask_apodization(
        mask, aposize, apotype=apotype)

    cltt, clee, clbb, clte = hp.read_cl(
        "outputs_22VTMDJN55_00000_00000/Cls_Planck2018_lensed_scalar.fits")[:, :4000]
    mp_t_sim, mp_q_sim, mp_u_sim = hp.synfast(
        [cltt, clee, clbb, clte], nside=nside, new=True, verbose=False)
    f2y0 = get_field(mp_q_sim, mp_u_sim, mask_apo)
    wsp.compute_coupling_matrix(f2y0, f2y0, b)
    #
    # Cl_cmb_freq = []
    # for f in range(len(spectra_cl.frequencies)):
    #     print('f', f)
    #     fn = get_field(mask*sim_cmb_freqmaps[f, 1, :],
    #                    mask*sim_cmb_freqmaps[f, 2, :], mask_apo,
    #                    purify_e=purify_e, purify_b=purify_b)
    #     # IPython.embed()
    #
    #     Cl_cmb_freq.append(compute_master(fn, fn, wsp))
    #
    # Cl_cmb_freq = np.array(Cl_cmb_freq)

    Cl_cmb = []

    Cl_dust = []
    Cl0_dust = []
    Cl_sync = []
    Cl_cmb_dust = []
    Cl_cmb_sync = []
    Cl_cmb_dust_sync = []

    start_10sim = time.time()
    for i in range(nsim):
        start_1sim = time.time()
        print('i', i)
        cmb_map = hp.synfast(spectra_cl.cl_rot.spectra.T, nside, new=True)  # * \
        # convert_units('K_RJ', 'K_CMB', i)

        # f0_cmb = nmt.NmtField(mask_apo, [mask*cmb_map[0, :]])
        f2_cmb = get_field(mask*cmb_map[1, :],
                           mask*cmb_map[2, :], mask_apo,
                           purify_e=purify_e, purify_b=purify_b)
        Cl_cmb.append(compute_master(f2_cmb, f2_cmb, wsp))

    for i in sky_map.frequencies:

        dust_maps_ = sky_map.sky.dust(i) * \
            convert_units('K_RJ', 'K_CMB', i)
        dust_maps = lib.map_rotation(dust_maps_, rotation_angle)

        f0_dust = nmt.NmtField(mask_apo, [mask*dust_maps[0, :]])

        f2_dust = get_field(mask*dust_maps[1, :],
                            mask*dust_maps[2, :], mask_apo,
                            purify_e=purify_e, purify_b=purify_b)

        sync_maps_ = sky_map.sky.synchrotron(i) * \
            convert_units('K_RJ', 'K_CMB', i)
        sync_maps = lib.map_rotation(sync_maps_, rotation_angle)
        # f0_sync = nmt.NmtField(mask_apo, [mask*sync_maps[0, :]])

        f2_sync = get_field(mask*sync_maps[1, :],
                            mask*sync_maps[2, :], mask_apo,
                            purify_e=purify_e, purify_b=purify_b)

        cmb_dust_map = (cmb_map/6) + dust_maps
        f2_cmb_dust = get_field(mask*cmb_dust_map[1, :],
                                mask*cmb_dust_map[2, :], mask_apo,
                                purify_e=purify_e, purify_b=purify_b)

        cmb_sync_map = (cmb_map/6) + sync_maps
        f2_cmb_sync = get_field(mask*cmb_sync_map[1, :],
                                mask*cmb_sync_map[2, :], mask_apo,
                                purify_e=purify_e, purify_b=purify_b)

        cmb_dust_sync_map = (cmb_map/6) + sync_maps + dust_maps
        f2_cmb_dust_sync = get_field(mask*cmb_dust_sync_map[1, :],
                                     mask*cmb_dust_sync_map[2, :], mask_apo,
                                     purify_e=purify_e, purify_b=purify_b)
        # IPython.embed()
        Cl_cmb_dust.append(compute_master(f2_cmb_dust, f2_cmb_dust, wsp))
        Cl_cmb_sync.append(compute_master(f2_cmb_sync, f2_cmb_sync, wsp))
        Cl_cmb_dust_sync.append(compute_master(f2_cmb_dust_sync, f2_cmb_dust_sync, wsp))
        Cl_dust.append(compute_master(f2_dust, f2_dust, wsp))
        Cl_sync.append(compute_master(f2_sync, f2_sync, wsp))

        Cl0_dust.append(nmt.compute_full_master(f0_dust, f0_dust, b))
        print('time 1 sim = ', time.time() - start_1sim)
    time10 = time.time() - start_10sim
    print('time 10 sim = ', time10)
    print('average time over {} sim = '.format(nsim), time10/nsim)

    Cl_cmb = np.array(Cl_cmb)
    Cl_dust = np.array(Cl_dust)
    Cl_sync = np.array(Cl_sync)
    Cl_cmb_dust = np.array(Cl_cmb_dust)
    Cl_cmb_sync = np.array(Cl_cmb_sync)
    Cl_cmb_dust_sync = np.array(Cl_cmb_dust_sync)

    Cl0_dust = np.array(Cl0_dust)

    Cl0_mean_dust = np.mean(Cl0_dust, axis=0)
    Cl_mean_dust = np.mean(Cl_dust, axis=0)
    true_cl_bined = b.bin_cell(spectra_cl.cl_rot.spectra[:3*nside].T)
    prim_cl_bined = b.bin_cell(spectra_cl.spectra.spectra[:3*nside].T)

    # simple_anafast = hp.anafast(sim_maps*mask)
    ells = [int(i) for i in b.get_effective_ells()]

    lensed_cl = cf.power_spectra_operation(r=r,
                                           rotation_angle=rotation_angle, powers_name='lensed_scalar')
    lensed_cl.get_spectra()

    # lensed_cl_bined = b.bin_cell(lensed_cl.cl_rot.spectra[:3*nside].T)
    # IPython.embed()
    '''------------------------------'''

    rotation_angle_plot = 0.1*u.rad
    spectra_cl_plot = cf.power_spectra_operation(r=r,
                                                 rotation_angle=rotation_angle_plot, powers_name='total')
    spectra_cl_plot.get_spectra()
    spectra_cl_plot.spectra_rotation()

    true_spectra = cf.power_spectra_obj(spectra_cl_plot.cl_rot.spectra,
                                        spectra_cl_plot.cl_rot.ell)
    true_spectra.ell = np.arange(lmin, lmax)
    # IPython.embed()
    # EE,EB?,BE?,BB
    # plt.plot(b.get_effective_ells(), simple_anafast[1, ells], label='Anafast method')

    fig, ax = plt.subplots()
    for i in range(len(sky_map.frequencies)):
        color = next(ax._get_lines.prop_cycler)['color']
        ax.plot(b.get_effective_ells(), Cl_dust[i, 1].T, alpha=1,
                label='Dust {} GHz'.format(sky_map.frequencies[i]), color=color)
        # ax.plot(b.get_effective_ells(), Cl_cmb_dust[i, 0].T, alpha=1,
        #         label='CMB + Dust {} GHz'.format(sky_map.frequencies[i]), color=color)
        # ax.plot(b.get_effective_ells(), Cl_sync[i, 1].T, '--', alpha=1,
        #         label='Sync {} GHz'.format(sky_map.frequencies[i]), color=color)
        ax.plot(b.get_effective_ells(), 0.01*np.sqrt(Cl_dust[i, 1].T * Cl_dust[i, 3].T), ':', alpha=1,
                label='Dust sqrt(EE*BB) {} GHz'.format(sky_map.frequencies[i]), color=color)

    # plt.plot(b.get_effective_ells(), Cl_mean.T[:, 0], label='Namaster method')

    color = next(ax._get_lines.prop_cycler)['color']

    rotation_angle_plot = 0.1*u.rad

    spectra_cl_plot.spectra_rotation(rotation_angle_plot)
    true_cl_bined = b.bin_cell(spectra_cl_plot.cl_rot.spectra[:3*nside].T)
    ax.plot(b.get_effective_ells(), true_cl_bined[4],
            '*', label='CAMB alpha = {}'.format(rotation_angle_plot), color=color)

    spectra_cl_plot.spectra_rotation(-rotation_angle_plot)
    true_cl_bined = b.bin_cell(spectra_cl_plot.cl_rot.spectra[:3*nside].T)
    ax.plot(b.get_effective_ells(), true_cl_bined[4],
            '.', label='CAMB alpha = -{}'.format(rotation_angle_plot), color=color)
    color = next(ax._get_lines.prop_cycler)['color']

    rotation_angle_plot = 0.5*u.rad

    spectra_cl_plot.spectra_rotation(rotation_angle_plot)
    true_cl_bined = b.bin_cell(spectra_cl_plot.cl_rot.spectra[:3*nside].T)
    ax.plot(b.get_effective_ells(), true_cl_bined[4],
            '*', label='CAMB alpha = {}'.format(rotation_angle_plot), color=color)

    spectra_cl_plot.spectra_rotation(-rotation_angle_plot)
    true_cl_bined_minus = b.bin_cell(spectra_cl_plot.cl_rot.spectra[:3*nside].T)
    ax.plot(b.get_effective_ells(), true_cl_bined_minus[4],
            '.', label='CAMB alpha = -{}'.format(rotation_angle_plot), color=color)
    # plt.plot(b.get_effective_ells(), prim_cl_bined[1], '--', label='Primordial spectrum from CAMB')
    plt.fill_between(b.get_effective_ells(), true_cl_bined[4], true_cl_bined_minus[4], alpha=0.2)
    # plt.plot(b.get_effective_ells(), Cl_sync.T[:, 0], label='namaster pysm sync')

    ax.set_xscale('log')
    # plt.yscale('log')
    ax.set_xlabel('ell')
    ax.set_ylabel('Cell EB')
    ax.set_title('Dust and synchrotron EB spectra recovered from PySM maps')
    ax.legend(loc='upper right')
    plt.show()

    """____________________plot cmb + dust power spectra____________________"""
    fig, ax = plt.subplots()
    dust_band = 2
    spectrum_num_namaster = 0

    dico_namaster2camb = {0: 1, 1: 4, 2: 4, 3: 2}
    dico_namaster2names = {0: 'EE', 1: 'EB', 2: 'EB', 3: 'BB'}
    spectrum_name = dico_namaster2names[spectrum_num_namaster]
    spectrum_num_camb = dico_namaster2camb[spectrum_num_namaster]

    ax.plot(b.get_effective_ells(), Cl_cmb[0, spectrum_num_namaster].T/36,
            '*', label='CMB')
    # ax.plot(b.get_effective_ells(), Cl_cmb_dust[dust_band, spectrum_num_namaster].T,
    #         alpha=1,
    #         label='CMB + Dust in map {} GHz'.format(sky_map.frequencies[dust_band]))
    # ax.plot(b.get_effective_ells(), Cl_cmb[0, spectrum_num_namaster].T/6 +
    #         Cl_dust[dust_band, spectrum_num_namaster].T,
    #         alpha=1,
    #         label='CMB + Dust in Cell {} GHz'.format(sky_map.frequencies[dust_band]))
    ax.plot(b.get_effective_ells(), Cl_dust[dust_band, spectrum_num_namaster].T,
            label='Dust {} GHz'.format(sky_map.frequencies[dust_band]))
    ax.plot(b.get_effective_ells(), Cl_sync[dust_band, spectrum_num_namaster].T,
            alpha=1,
            label='Synchrotron {} GHz'.format(sky_map.frequencies[dust_band]))
    model_cl_bined = b.bin_cell(spectra_cl.cl_rot.spectra[:3*nside].T).T
    # ax.plot(b.get_effective_ells(), model_cl_bined[:, spectrum_num_camb]/6,
    #         label='camb binned/#frequencies')
    ax.plot(b.get_effective_ells(), model_cl_bined[:, spectrum_num_camb]/36,
            label='camb binned')
    ax.set_xscale('log')
    # plt.yscale('log')
    ax.set_xlabel('ell')
    ax.set_ylabel('Cell {}'.format(spectrum_name))
    # ax.set_title('Dust and synchrotron {} spectra recovered from PySM maps'.format(spectrum_name))
    ax.legend(loc='upper left')
    plt.show()

    # IPython.embed()

    """________________________likelihood estimation________________________"""
    min_angle = -0.02  # rotation.value - 5*(1/np.sqrt(fishaa))
    max_angle = 0.02  # rotation.value + 5*(1/np.sqrt(fishaa))
    nstep_angle = 1000
    angle_grid = np.arange(min_angle, max_angle,
                           (max_angle - min_angle)/nstep_angle)*u.radian
    # idx1 = (np.abs(angle_grid.value - rotation.value)).argmin()
    # print('angle_grid check ', angle_grid[idx1])
    """dust bands = [27,39,93,145,225,280]"""
    dust_band = 2

    data_spectra_cmb = Cl_cmb[0].T
    data_matrix_cmb = cf.power_spectra_obj(
        np.array([[data_spectra_cmb[:, 0], data_spectra_cmb[:, 2]],
                  [data_spectra_cmb[:, 1], data_spectra_cmb[:, 3]]]).T,
        b.get_effective_ells())

    data_spectra_cmb_dust = Cl_cmb_dust[dust_band].T
    data_matrix_cmb_dust = cf.power_spectra_obj(
        np.array([[data_spectra_cmb_dust[:, 0], data_spectra_cmb_dust[:, 2]],
                  [data_spectra_cmb_dust[:, 1], data_spectra_cmb_dust[:, 3]]]).T,
        b.get_effective_ells())

    data_spectra_cmb_sync = Cl_cmb_sync[dust_band].T
    data_matrix_cmb_sync = cf.power_spectra_obj(
        np.array([[data_spectra_cmb_sync[:, 0], data_spectra_cmb_sync[:, 2]],
                  [data_spectra_cmb_sync[:, 1], data_spectra_cmb_sync[:, 3]]]).T,
        b.get_effective_ells())

    data_spectra_cmb_dust_sync = Cl_cmb_dust_sync[dust_band].T
    data_matrix_cmb_dust_sync = cf.power_spectra_obj(
        np.array([[data_spectra_cmb_dust_sync[:, 0], data_spectra_cmb_dust_sync[:, 2]],
                  [data_spectra_cmb_dust_sync[:, 1], data_spectra_cmb_dust_sync[:, 3]]]).T,
        b.get_effective_ells())
    # model_spectra = cf.power_spectra_operation()
    # model_spectra.spectra = cf.power_spectra_obj(
    #     np.array([copy.deepcopy(data_spectra[:, 0]*0),
    #               copy.deepcopy(data_spectra[:, 0]),
    #               copy.deepcopy(data_spectra[:, 3]),
    #               copy.deepcopy(data_spectra[:, 0])*0,
    #               copy.deepcopy(data_spectra[:, 1])*0,
    #               copy.deepcopy(data_spectra[:, 0])*0]).T,
    #     b.get_effective_ells())

    spectra_cl = cf.power_spectra_operation(r=r,
                                            rotation_angle=rotation_angle, powers_name='total')
    spectra_cl.get_spectra()
    spectra_cl.spectra_rotation()
    spectra_cl.get_frequencies()

    # true_spectra = cf.power_spectra_obj(spectra_cl.cl_rot.spectra,
    #                                     spectra_cl.cl_rot.ell)
    # true_spectra.ell = np.arange(lmin, lmax)
    model_cl_bined = b.bin_cell(spectra_cl.cl_rot.spectra[:3*nside].T).T

    # data_spectra_cmb_CAMB = Cl_cmb[0].T
    data_matrix_cmb_CAMB = cf.power_spectra_obj(
        np.array([[model_cl_bined[:, 1], model_cl_bined[:, 4]],
                  [model_cl_bined[:, 4], model_cl_bined[:, 2]]]).T,
        b.get_effective_ells())
    # model_matrix = cf.power_spectra_obj(
    #     np.array([[model_spectra.spectra.spectra[:, 0], model_spectra.spectra.spectra[:, 2]*0],
    #               [model_spectra.spectra.spectra[:, 1]*0, model_spectra.spectra.spectra[:, 3]]]).T,
    #     b.get_effective_ells())
    # IPython.embed()

    likelihood_values_cmb_CAMB = []
    likelihood_values_cmb = []
    likelihood_values_cmb_dust = []
    likelihood_values_cmb_sync = []
    likelihood_values_cmb_dust_sync = []

    for angle in angle_grid:
        # model_spectra.spectra_rotation(angle)
        spectra_cl.spectra_rotation(angle)
        model_cl_bined = b.bin_cell(spectra_cl.cl_rot.spectra[:3*nside].T).T
        # model_cl_bined = spectra_cl.cl_rot.spectra[:3*nside]
        # model_cl_bined = model_cl_bined[ells]

        # model.get_instrument_spectra()
        # model_spectra_rot = model_spectra.cl_rot.spectra

        model_matrix = cf.power_spectra_obj(np.array(
            [[model_cl_bined[:, 1], model_cl_bined[:, 4]],
             [model_cl_bined[:, 4], model_cl_bined[:, 2]]]).T,
            b.get_effective_ells())

        """CMB CAMB"""
        likelihood_val_cmb_CAMB = cf.likelihood_pws(model_matrix, data_matrix_cmb_CAMB,
                                                    f_sky=f_sky_SAT)
        likelihood_values_cmb_CAMB.append(likelihood_val_cmb_CAMB)

        """CMB"""
        likelihood_val_cmb = cf.likelihood_pws(model_matrix, data_matrix_cmb,
                                               f_sky=f_sky_SAT)
        likelihood_values_cmb.append(likelihood_val_cmb)

        """CMB + Dust"""
        likelihood_val_cmb_dust = cf.likelihood_pws(model_matrix, data_matrix_cmb_dust,
                                                    f_sky=f_sky_SAT)
        likelihood_values_cmb_dust.append(likelihood_val_cmb_dust)

        """CMB + Synchrotron"""
        likelihood_val_cmb_sync = cf.likelihood_pws(model_matrix, data_matrix_cmb_sync,
                                                    f_sky=f_sky_SAT)
        likelihood_values_cmb_sync.append(likelihood_val_cmb_sync)

        """CMB + Dust + Synchrotron"""
        likelihood_val_cmb_dust_sync = cf.likelihood_pws(model_matrix, data_matrix_cmb_dust_sync,
                                                         f_sky=f_sky_SAT)
        likelihood_values_cmb_dust_sync.append(likelihood_val_cmb_dust_sync)

    likelihood_norm_cmb_CAMB = np.array(
        likelihood_values_cmb_CAMB) - min(likelihood_values_cmb_CAMB)
    likelihood_norm_cmb = np.array(
        likelihood_values_cmb) - min(likelihood_values_cmb)
    likelihood_norm_cmb_dust = np.array(
        likelihood_values_cmb_dust) - min(likelihood_values_cmb_dust)
    likelihood_norm_cmb_sync = np.array(
        likelihood_values_cmb_sync) - min(likelihood_values_cmb_sync)
    likelihood_norm_cmb_dust_sync = np.array(
        likelihood_values_cmb_dust_sync) - min(likelihood_values_cmb_dust_sync)

    plt.plot(angle_grid, np.exp(-likelihood_norm_cmb_CAMB),
             label='CMB CAMB')
    plt.plot(angle_grid, np.exp(-likelihood_norm_cmb),
             label='CMB')
    plt.plot(angle_grid, np.exp(-likelihood_norm_cmb_dust),
             label='CMB + Dust')
    plt.plot(angle_grid, np.exp(-likelihood_norm_cmb_sync),
             label='CMB + Synchrotron')
    plt.plot(angle_grid, np.exp(-likelihood_norm_cmb_dust_sync),
             label='CMB + Dust + Synchrotron')
    plt.title('Likelihood on miscalibration with CMB and foregrounds at {} GHz and CMB only for the model'.format(
        sky_map.frequencies[dust_band]))
    plt.vlines(0, 0, 1, linestyles='--', label='true miscalibration angle')
    plt.legend()
    plt.xlabel('miscalibration angle')
    plt.ylabel('likelihood')
    plt.show()

    IPython.embed()
    exit()


if __name__ == "__main__":
    main()

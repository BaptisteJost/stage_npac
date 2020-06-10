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
from scipy.optimize import minimize
import numdifftools as nd
import mk_noise_map2 as mknm


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
    pysm_model = 'c1s1d1'
    f_sky_SAT = 0.1
    sensitivity = 0
    knee_mode = 0
    norm_hits_map_path = '/home/baptiste/BBPipe/test_mapbased_param/norm_nHits_SA_35FOV_G_nside512.fits'
    no_inh = False
    ny_lf = 1

    # from config.yml
    nside = 512
    lmin = 30
    lmax = 500
    nlb = 10
    custom_bins = True
    aposize = 8.0
    apotype = 'C1'
    purify_b = True

    spectra_cl = cf.power_spectra_operation(r=r, rotation_angle=rotation_angle,
                                            powers_name='total')
    spectra_cl.get_spectra()
    spectra_cl.spectra_rotation()
    spectra_cl.get_frequencies()

    true_spectra = cf.power_spectra_obj(spectra_cl.cl_rot.spectra,
                                        spectra_cl.cl_rot.ell)
    true_spectra.ell = np.arange(lmin, lmax)

    sky_map = l_SO.sky_map(nside=nside, sky_model=pysm_model)
    sky_map.get_pysm_sky()
    sky_map.get_frequency()

    print('initializing Namaster ...')
    start_namaster = time.time()
    wsp = nmt.NmtWorkspace()
    b = binning_definition(nside, lmin=lmin, lmax=lmax,
                           nlb=nlb, custom_bins=custom_bins)
    # wsp.wsp.ncls = 4

    print('building mask ... ')
    # mask = hp.read_map("outputs_22VTMDJN55_00000_00000/binary_mask_cut.fits")
    mask_ = hp.read_map(
        "/home/baptiste/BBPipe/test_mapbased_param/norm_nHits_SA_35FOV_G_nside512.fits")
    mask = hp.ud_grade(mask_, nside)
    del mask_
    mask_apo = nmt.mask_apodization(
        mask, aposize, apotype=apotype)

    cltt, clee, clbb, clte = hp.read_cl(
        "outputs_22VTMDJN55_00000_00000/Cls_Planck2018_lensed_scalar.fits")[:, :4000]
    mp_t_sim, mp_q_sim, mp_u_sim = hp.synfast(
        [cltt, clee, clbb, clte], nside=nside, new=True, verbose=False)
    f2y0 = get_field(mp_q_sim, mp_u_sim, mask_apo)
    wsp.compute_coupling_matrix(f2y0, f2y0, b)
    print('Namaster initialized in {}s'.format(time.time() - start_namaster))

    frequencies2use = [93]
    frequencies_index = []
    for f in frequencies2use:
        frequencies_index.append(sky_map.frequencies.tolist().index(f))

    """____________________________map creation____________________________"""
    print("creatin maps ...")
    nhits, noise_maps_, nlev = mknm.get_noise_sim(sensitivity=sensitivity,
                                                  knee_mode=knee_mode, ny_lf=ny_lf,
                                                  nside_out=nside,
                                                  norm_hits_map=hp.read_map(
                                                      norm_hits_map_path),
                                                  no_inh=no_inh)
    noise_maps = []
    noise_maps_masked = []

    for i in frequencies_index:
        noise_maps.append([noise_maps_[i*3],
                           noise_maps_[i*3+1],
                           noise_maps_[i*3+2]])
        noise_maps_masked.append([noise_maps_[i*3] * mask,
                                  noise_maps_[i*3+1] * mask,
                                  noise_maps_[i*3+2] * mask])
    del noise_maps_
    # noise_maps.append(noise_maps_[i*3+2] * mask)

    print("creating foregrounds maps ...")
    start_map = time.time()
    dust_map_freq = []
    sync_map_freq = []
    Cl_dust_freq = []
    Cl_sync_freq = []

    for f in frequencies2use:
        dust_maps_ = sky_map.sky.dust(f) * \
            convert_units('K_RJ', 'K_CMB', f)
        dust_maps = lib.map_rotation(dust_maps_, rotation_angle)
        dust_map_freq.append(dust_maps)

        sync_maps_ = sky_map.sky.synchrotron(f) * \
            convert_units('K_RJ', 'K_CMB', f)
        sync_maps = lib.map_rotation(sync_maps_, rotation_angle)
        sync_map_freq.append(sync_maps)

        # f0_dust = nmt.NmtField(mask_apo, [mask*dust_maps[0, :]])

        f2_dust = get_field(mask*dust_maps[1, :],
                            mask*dust_maps[2, :], mask_apo,
                            purify_e=purify_e, purify_b=purify_b)

        # f0_sync = nmt.NmtField(mask_apo, [mask*sync_maps[0, :]])

        f2_sync = get_field(mask*sync_maps[1, :],
                            mask*sync_maps[2, :], mask_apo,
                            purify_e=purify_e, purify_b=purify_b)
        Cl_dust_freq.append(compute_master(f2_dust, f2_dust, wsp))
        Cl_sync_freq.append(compute_master(f2_sync, f2_sync, wsp))
    dust_map_freq = np.array(dust_map_freq)
    sync_map_freq = np.array(sync_map_freq)
    Cl_dust = np.array(Cl_dust_freq)
    Cl_sync = np.array(Cl_sync_freq)

    cmb_map_nsim = []
    print("creating CMB map simulations ...")
    print('WARNING binning_definition sets seed, CMB map created before, need proper solution')
    print('WARNING: for now seed will always be the same for debugging purposes')
    for i in range(nsim):
        # np.random.seed(int(time.time()))
        np.random.seed(i)

        cmb_map = hp.synfast(spectra_cl.cl_rot.spectra.T, nside, new=True)
        cmb_map_nsim.append(cmb_map)
    cmb_map_nsim.append(cmb_map + noise_maps[0])
    # cmb_map_nsim = np.array(cmb_map_nsim) + noise_maps[0]
    cmb_map_nsim.append(noise_maps[0])
    cmb_map_nsim.append(noise_maps_masked[0])

    cmb_map_nsim = np.array(cmb_map_nsim)

    nsim += 3
    print("")
    print("WARNING : noise maps added directly to cmb map")
    print("IT WILL CAUSE PB when multiple frequencies are used")
    print("Component maps created successfully in {} s".format(
        time.time() - start_map))

    """____________________________Cl estimation____________________________"""
    Cl_cmb = []
    Cl_cmb_dust = []
    Cl_cmb_sync = []
    Cl_cmb_dust_sync = []
    # IPython.embed()
    start_10sim = time.time()
    for i in range(nsim):
        start_1sim = time.time()
        print('i', i)
        # np.random.seed(int(time.time()))
        # cmb_map = hp.synfast(spectra_cl.cl_rot.spectra.T, nside, new=True)

        # print(cmb_map)
        # f0_cmb = nmt.NmtField(mask_apo, [mask*cmb_map[0, :]])
        f2_cmb = get_field(mask*cmb_map_nsim[i, 1, :],
                           mask*cmb_map_nsim[i, 2, :], mask_apo,
                           purify_e=purify_e, purify_b=purify_b)
        Cl_cmb.append(compute_master(f2_cmb, f2_cmb, wsp))

        Cl_cmb_dust_freq = []
        Cl_cmb_sync_freq = []
        Cl_cmb_dust_sync_freq = []

        for f in range(len(frequencies2use)):  # sky_map.frequencies:

            # dust_maps_ = sky_map.sky.dust(f) * \
            #     convert_units('K_RJ', 'K_CMB', f)
            # dust_maps = lib.map_rotation(dust_maps_, rotation_angle)
            #
            # f0_dust = nmt.NmtField(mask_apo, [mask*dust_maps[0, :]])
            #
            # f2_dust = get_field(mask*dust_maps[1, :],
            #                     mask*dust_maps[2, :], mask_apo,
            #                     purify_e=purify_e, purify_b=purify_b)
            #
            # sync_maps_ = sky_map.sky.synchrotron(f) * \
            #     convert_units('K_RJ', 'K_CMB', f)
            # sync_maps = lib.map_rotation(sync_maps_, rotation_angle)
            # # f0_sync = nmt.NmtField(mask_apo, [mask*sync_maps[0, :]])
            #
            # f2_sync = get_field(mask*sync_maps[1, :],
            #                     mask*sync_maps[2, :], mask_apo,
            #                     purify_e=purify_e, purify_b=purify_b)

            cmb_dust_map = (cmb_map_nsim[i]/6) + dust_map_freq[f]
            f2_cmb_dust = get_field(mask*cmb_dust_map[1, :],
                                    mask*cmb_dust_map[2, :], mask_apo,
                                    purify_e=purify_e, purify_b=purify_b)

            cmb_sync_map = (cmb_map_nsim[i]/6) + sync_map_freq[f]
            f2_cmb_sync = get_field(mask*cmb_sync_map[1, :],
                                    mask*cmb_sync_map[2, :], mask_apo,
                                    purify_e=purify_e, purify_b=purify_b)

            cmb_dust_sync_map = (cmb_map_nsim[i]/6) + dust_map_freq[f] + sync_map_freq[f]
            f2_cmb_dust_sync = get_field(mask*cmb_dust_sync_map[1, :],
                                         mask*cmb_dust_sync_map[2, :], mask_apo,
                                         purify_e=purify_e, purify_b=purify_b)
            # IPython.embed()

            Cl_cmb_dust_freq.append(compute_master(f2_cmb_dust, f2_cmb_dust, wsp))
            Cl_cmb_sync_freq.append(compute_master(f2_cmb_sync, f2_cmb_sync, wsp))
            Cl_cmb_dust_sync_freq.append(compute_master(f2_cmb_dust_sync, f2_cmb_dust_sync, wsp))
            # Cl_dust_freq.append(compute_master(f2_dust, f2_dust, wsp))
            # Cl_sync_freq.append(compute_master(f2_sync, f2_sync, wsp))
        Cl_cmb_dust.append(Cl_cmb_dust_freq)
        Cl_cmb_sync.append(Cl_cmb_sync_freq)
        Cl_cmb_dust_sync.append(Cl_cmb_dust_sync_freq)
        # Cl_dust.append(Cl_dust_freq)
        # Cl_sync.append(Cl_sync_freq)

        # Cl0_dust.append(nmt.compute_full_master(f0_dust, f0_dust, b))
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

    IPython.embed()

    # Cl0_dust = np.array(Cl0_dust)
    """________________________test nsim likelihood________________________"""
    min_angle = -0.005  # rotation.value - 5*(1/np.sqrt(fishaa))
    max_angle = 0.005  # rotation.value + 5*(1/np.sqrt(fishaa))
    nstep_angle = 1000
    angle_grid = np.arange(min_angle, max_angle,
                           (max_angle - min_angle)/nstep_angle)*u.radian
    fit_alpha_cmb = []
    H_cmb = []

    fit_alpha_cmb_dust = []
    H_cmb_dust = []
    fit_alpha_cmb_sync = []
    H_cmb_sync = []
    fit_alpha_cmb_dust_sync = []
    H_cmb_dust_sync = []
    # likelihood_values_cmb_nsim = []
    for i in range(nsim):
        fit_alpha_freq_cmb = []
        H_freq_cmb = []
        fit_alpha_freq_cmb_dust = []
        H_freq_cmb_dust = []
        fit_alpha_freq_cmb_sync = []
        H_freq_cmb_sync = []
        fit_alpha_freq_cmb_dust_sync = []
        H_freq_cmb_dust_sync = []
        for f in range(len(frequencies2use)):
            data_spectra_cmb = Cl_cmb[i].T
            data_spectra_cmb_dust = Cl_cmb_dust[i, f].T
            data_spectra_cmb_sync = Cl_cmb_sync[i, f].T
            data_spectra_cmb_dust_sync = Cl_cmb_dust_sync[i, f].T

            data_matrix_cmb = cf.power_spectra_obj(
                np.array([[data_spectra_cmb[:, 0], data_spectra_cmb[:, 2]],
                          [data_spectra_cmb[:, 1], data_spectra_cmb[:, 3]]]).T,
                b.get_effective_ells())

            data_matrix_cmb_dust = cf.power_spectra_obj(
                np.array([[data_spectra_cmb_dust[:, 0], data_spectra_cmb_dust[:, 2]],
                          [data_spectra_cmb_dust[:, 1], data_spectra_cmb_dust[:, 3]]]).T,
                b.get_effective_ells())
            data_matrix_cmb_sync = cf.power_spectra_obj(
                np.array([[data_spectra_cmb_sync[:, 0], data_spectra_cmb_sync[:, 2]],
                          [data_spectra_cmb_sync[:, 1], data_spectra_cmb_sync[:, 3]]]).T,
                b.get_effective_ells())
            data_matrix_cmb_dust_sync = cf.power_spectra_obj(
                np.array([[data_spectra_cmb_dust_sync[:, 0], data_spectra_cmb_dust_sync[:, 2]],
                          [data_spectra_cmb_dust_sync[:, 1], data_spectra_cmb_dust_sync[:, 3]]]).T,
                b.get_effective_ells())

            likelihood_minimised_cmb = minimize(
                cf.likelihood_for_hessian_a,    0,
                (spectra_cl, data_matrix_cmb, b, nside, f_sky_SAT))
            H_cmb_ = nd.Hessian(cf.likelihood_for_hessian_a)(
                likelihood_minimised_cmb.x,
                spectra_cl, data_matrix_cmb, b,
                nside, f_sky_SAT)
            fit_alpha_freq_cmb.append(likelihood_minimised_cmb.x[0])
            H_freq_cmb.append(H_cmb_[0][0])

            likelihood_minimised_cmb_dust = minimize(
                cf.likelihood_for_hessian_a, 0,
                (spectra_cl, data_matrix_cmb_dust, b, nside, f_sky_SAT))
            H_cmb_dust_ = nd.Hessian(cf.likelihood_for_hessian_a)(
                likelihood_minimised_cmb_dust.x,
                spectra_cl, data_matrix_cmb_dust, b,
                nside, f_sky_SAT)
            fit_alpha_freq_cmb_dust.append(likelihood_minimised_cmb_dust.x[0])
            H_freq_cmb_dust.append(H_cmb_dust_[0][0])

            likelihood_minimised_cmb_sync = minimize(
                cf.likelihood_for_hessian_a, 0,
                (spectra_cl, data_matrix_cmb_sync, b, nside, f_sky_SAT))
            H_cmb_sync_ = nd.Hessian(cf.likelihood_for_hessian_a)(
                likelihood_minimised_cmb_sync.x,
                spectra_cl, data_matrix_cmb_sync, b,
                nside, f_sky_SAT)
            fit_alpha_freq_cmb_sync.append(likelihood_minimised_cmb_sync.x[0])
            H_freq_cmb_sync.append(H_cmb_sync_[0][0])

            likelihood_minimised_cmb_dust_sync = minimize(
                cf.likelihood_for_hessian_a, 0,
                (spectra_cl, data_matrix_cmb_dust_sync, b, nside, f_sky_SAT))
            H_cmb_dust_sync_ = nd.Hessian(cf.likelihood_for_hessian_a)(
                likelihood_minimised_cmb_dust_sync.x,
                spectra_cl, data_matrix_cmb_dust_sync, b,
                nside, f_sky_SAT)
            fit_alpha_freq_cmb_dust_sync.append(likelihood_minimised_cmb_dust_sync.x[0])
            H_freq_cmb_dust_sync.append(H_cmb_dust_sync_[0][0])

        fit_alpha_cmb.append(fit_alpha_freq_cmb)
        H_cmb.append(H_freq_cmb)
        fit_alpha_cmb_dust.append(fit_alpha_freq_cmb_dust)
        H_cmb_dust.append(H_freq_cmb_dust)
        fit_alpha_cmb_sync.append(fit_alpha_freq_cmb_sync)
        H_cmb_sync.append(H_freq_cmb_sync)
        fit_alpha_cmb_dust_sync.append(fit_alpha_freq_cmb_dust_sync)
        H_cmb_dust_sync.append(H_freq_cmb_dust_sync)
    fit_alpha_cmb = np.array(fit_alpha_cmb)
    H_cmb = np.array(H_cmb)
    fit_alpha_cmb_dust = np.array(fit_alpha_cmb_dust)
    H_cmb_dust = np.array(H_cmb_dust)
    fit_alpha_cmb_sync = np.array(fit_alpha_cmb_sync)
    H_cmb_sync = np.array(H_cmb_sync)
    fit_alpha_cmb_dust_sync = np.array(fit_alpha_cmb_dust_sync)
    H_cmb_dust_sync = np.array(H_cmb_dust_sync)

    plt.errorbar(range(nsim), fit_alpha_cmb[:, 0],
                 yerr=1/np.sqrt(H_cmb[:, 0]), fmt='o', label='CMB')
    plt.errorbar(range(nsim, nsim+nsim), fit_alpha_cmb_dust[:, 0],
                 yerr=1/np.sqrt(H_cmb_dust[:, 0]), fmt='o', label='CMB + Dust')
    plt.errorbar(range(2*nsim, 2*nsim+nsim), fit_alpha_cmb_sync[:, 0],
                 yerr=1/np.sqrt(H_cmb_sync[:, 0]), fmt='o', label='CMB + Synchrotron')
    plt.errorbar(range(3*nsim, 3*nsim+nsim), fit_alpha_cmb_dust_sync[:, 0],
                 yerr=1/np.sqrt(H_cmb_dust_sync[:, 0]), fmt='o', label='CMB + Dust + Synchrotron')
    plt.hlines(0, 0, 3*nsim+nsim,  label='input miscalibration angle')
    plt.hlines(np.mean(fit_alpha_cmb[:, 0]), 0, 3*nsim+nsim,
               linestyles='--', label='average CMB fit', color='blue')
    plt.hlines(np.mean(fit_alpha_cmb_dust[:, 0]), 0, 3*nsim+nsim,
               linestyles='--', label='average CMB + Dust fit', color='orange')
    plt.hlines(np.mean(fit_alpha_cmb_sync[:, 0]), 0, 3*nsim+nsim,
               linestyles='--', label='average CMB + synchrotron fit', color='green')
    plt.hlines(np.mean(fit_alpha_cmb_dust_sync[:, 0]), 0, 3*nsim+nsim,
               linestyles='--', label='average CMB + Dust + synchrotron fit', color='red')
    plt.legend()
    plt.grid(linestyle=':')
    plt.xlabel('simulations')
    plt.ylabel('miscalibration angle')
    plt.title('miscalibration angle fit at {} GHz'.format(frequencies2use))
    plt.show()
    IPython.embed()

    # likelihood_values_cmb = []
    # for angle in angle_grid:
    #     spectra_cl.spectra_rotation(angle)
    #     model_cl_bined = b.bin_cell(spectra_cl.cl_rot.spectra[:3*nside].T).T
    #
    #     model_matrix = cf.power_spectra_obj(np.array(
    #         [[model_cl_bined[:, 1], model_cl_bined[:, 4]],
    #          [model_cl_bined[:, 4], model_cl_bined[:, 2]]]).T,
    #         b.get_effective_ells())
    #
    #     """CMB"""
    #     likelihood_val_cmb = cf.likelihood_pws(model_matrix, data_matrix_cmb,
    #                                            f_sky=f_sky_SAT)
    #     likelihood_values_cmb.append(likelihood_val_cmb)
    # likelihood_values_cmb_nsim.append(likelihood_values_cmb)

    # for i in range(nsim):
    #     min_i = min(likelihood_values_cmb_nsim[i])
    #     plt.plot(
    #         angle_grid, np.exp(-(np.array(likelihood_values_cmb_nsim[i])-min_i)),
    #         label='{}'.format(i))

    plt.ylabel('likelihood')
    plt.xlabel('angle')
    plt.vlines(0, 0, 1, linestyles='--', label='true miscalibration angle')
    plt.legend()
    # IPython.embed()

    # Cl0_mean_dust = np.mean(Cl0_dust, axis=0)
    Cl_mean_dust = np.mean(Cl_dust, axis=0)
    true_cl_bined = b.bin_cell(spectra_cl.cl_rot.spectra[:3*nside].T)
    prim_cl_bined = b.bin_cell(spectra_cl.spectra.spectra[:3*nside].T)

    # simple_anafast = hp.anafast(sim_maps*mask)
    ells = [int(i) for i in b.get_effective_ells()]

    lensed_cl = cf.power_spectra_operation(r=r,
                                           rotation_angle=rotation_angle, powers_name='lensed_scalar')
    lensed_cl.get_spectra()

    # lensed_cl_bined = b.bin_cell(lensed_cl.cl_rot.spectra[:3*nside].T)
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

    """______________________likelihood initialisation______________________"""
    min_angle = -0.02  # rotation.value - 5*(1/np.sqrt(fishaa))
    max_angle = 0.02  # rotation.value + 5*(1/np.sqrt(fishaa))
    nstep_angle = 1000
    angle_grid = np.arange(min_angle, max_angle,
                           (max_angle - min_angle)/nstep_angle)*u.radian
    # idx1 = (np.abs(angle_grid.value - rotation.value)).argmin()
    # print('angle_grid check ', angle_grid[idx1])
    """dust bands = [27,39,93,145,225,280]"""
    dust_band = 0
    simu_number = 1

    data_spectra_cmb = Cl_cmb[simu_number].T
    data_matrix_cmb = cf.power_spectra_obj(
        np.array([[data_spectra_cmb[:, 0], data_spectra_cmb[:, 2]],
                  [data_spectra_cmb[:, 1], data_spectra_cmb[:, 3]]]).T,
        b.get_effective_ells())

    # data_spectra_cmb_no_noise = Cl_cmb[0].T
    # data_matrix_cmb_no_noise = cf.power_spectra_obj(
    #     np.array([[data_spectra_cmb_no_noise[:, 0], data_spectra_cmb_no_noise[:, 2]],
    #               [data_spectra_cmb_no_noise[:, 1], data_spectra_cmb_no_noise[:, 3]]]).T,
    #     b.get_effective_ells())

    data_spectra_cmb_dust = Cl_cmb_dust[simu_number][dust_band].T
    data_matrix_cmb_dust = cf.power_spectra_obj(
        np.array([[data_spectra_cmb_dust[:, 0], data_spectra_cmb_dust[:, 2]],
                  [data_spectra_cmb_dust[:, 1], data_spectra_cmb_dust[:, 3]]]).T,
        b.get_effective_ells())

    data_spectra_cmb_sync = Cl_cmb_sync[simu_number][dust_band].T
    data_matrix_cmb_sync = cf.power_spectra_obj(
        np.array([[data_spectra_cmb_sync[:, 0], data_spectra_cmb_sync[:, 2]],
                  [data_spectra_cmb_sync[:, 1], data_spectra_cmb_sync[:, 3]]]).T,
        b.get_effective_ells())

    data_spectra_cmb_dust_sync = Cl_cmb_dust_sync[simu_number][dust_band].T
    data_matrix_cmb_dust_sync = cf.power_spectra_obj(
        np.array([[data_spectra_cmb_dust_sync[:, 0], data_spectra_cmb_dust_sync[:, 2]],
                  [data_spectra_cmb_dust_sync[:, 1], data_spectra_cmb_dust_sync[:, 3]]]).T,
        b.get_effective_ells())

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
    likelihood_minimised = minimize(cf.likelihood_for_hessian_a,
                                    0,
                                    (spectra_cl, data_matrix_cmb, b, nside, f_sky_SAT))

    H = nd.Hessian(cf.likelihood_for_hessian_a)(likelihood_minimised.x,
                                                spectra_cl, data_matrix_cmb, b,
                                                nside, f_sky_SAT)

    """__________________plot cmb + dust + likelihood fit__________________"""
    fig, ax = plt.subplots()
    dust_band = 2
    spectrum_num_namaster = 1

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

    rotation_angle_fit = likelihood_minimised.x[0] * u.rad
    spectra_cl_plot.spectra_rotation(rotation_angle_fit)
    fit_cl_bined = b.bin_cell(spectra_cl_plot.cl_rot.spectra[:3*nside].T)
    ax.plot(b.get_effective_ells(), fit_cl_bined[4]/36,
            '*', label='fit alpha = {}'.format(rotation_angle_fit), color=color)

    rotation_angle_fit_sigmasup = (likelihood_minimised.x[0] + (
        1/np.sqrt(H[0][0]))) * u.rad
    rotation_angle_fit_sigmainf = (likelihood_minimised.x[0] - (
        1/np.sqrt(H[0][0]))) * u.rad

    spectra_cl_plot.spectra_rotation(rotation_angle_fit_sigmasup)
    simgasup_cl_bined = b.bin_cell(spectra_cl_plot.cl_rot.spectra[:3*nside].T)

    spectra_cl_plot.spectra_rotation(rotation_angle_fit_sigmainf)
    simgainf_cl_bined = b.bin_cell(spectra_cl_plot.cl_rot.spectra[:3*nside].T)

    ax.fill_between(b.get_effective_ells(),
                    simgasup_cl_bined[4]/36, simgainf_cl_bined[4]/36,
                    label='{} +/- sigma {}'.format(rotation_angle_fit, 1/np.sqrt(H[0][0])),
                    alpha=0.8)

    ax.set_xscale('log')
    # plt.yscale('log')
    ax.set_xlabel('ell')
    ax.set_ylabel('Cell {}'.format(spectrum_name))
    # ax.set_title('Dust and synchrotron {} spectra recovered from PySM maps'.format(spectrum_name))
    ax.legend(loc='upper left')
    plt.show()

    """_________________________Likelihood gridding_________________________"""

    likelihood_values_cmb_CAMB = []
    likelihood_values_cmb = []
    likelihood_values_cmb_dust = []
    likelihood_values_cmb_sync = []
    likelihood_values_cmb_dust_sync = []
    # likelihood_values_cmb_no_noise = []
    data_array = []
    model_cl_bined_array = []
    det_model_matrix = []
    for angle in angle_grid:
        # model_spectra.spectra_rotation(angle)
        spectra_cl.spectra_rotation(angle)
        model_cl_bined = b.bin_cell(spectra_cl.cl_rot.spectra[:3*nside].T).T
        model_cl_bined_array.append(model_cl_bined)
        # model_cl_bined = spectra_cl.cl_rot.spectra[:3*nside]
        # model_cl_bined = model_cl_bined[ells]

        # model.get_instrument_spectra()
        # model_spectra_rot = model_spectra.cl_rot.spectra

        model_matrix = cf.power_spectra_obj(np.array(
            [[model_cl_bined[:, 1], model_cl_bined[:, 4]],
             [model_cl_bined[:, 4], model_cl_bined[:, 2]]]).T,
            b.get_effective_ells())

        det_model_matrix.append(np.linalg.inv(model_matrix.spectra[20]))
        # """CMB CAMB"""
        # likelihood_val_cmb_CAMB = cf.likelihood_pws(model_matrix, data_matrix_cmb_CAMB,
        #                                             f_sky=f_sky_SAT)
        # likelihood_values_cmb_CAMB.append(likelihood_val_cmb_CAMB)

        """CMB"""
        likelihood_val_cmb, data = cf.likelihood_pws(model_matrix, data_matrix_cmb,
                                                     f_sky=f_sky_SAT)
        likelihood_values_cmb.append(likelihood_val_cmb)
        data_array.append(data)

        # """CMB no noise"""
        # likelihood_val_cmb_no_noise = cf.likelihood_pws(model_matrix, data_matrix_cmb_no_noise,
        #                                                 f_sky=f_sky_SAT)
        # likelihood_values_cmb_no_noise.append(likelihood_val_cmb_no_noise)

        # """CMB + Dust"""
        # likelihood_val_cmb_dust = cf.likelihood_pws(model_matrix, data_matrix_cmb_dust,
        #                                             f_sky=f_sky_SAT)
        # likelihood_values_cmb_dust.append(likelihood_val_cmb_dust)
        #
        # """CMB + Synchrotron"""
        # likelihood_val_cmb_sync = cf.likelihood_pws(model_matrix, data_matrix_cmb_sync,
        #                                             f_sky=f_sky_SAT)
        # likelihood_values_cmb_sync.append(likelihood_val_cmb_sync)
        #
        # """CMB + Dust + Synchrotron"""
        # likelihood_val_cmb_dust_sync = cf.likelihood_pws(model_matrix, data_matrix_cmb_dust_sync,
        #                                                  f_sky=f_sky_SAT)
        # likelihood_values_cmb_dust_sync.append(likelihood_val_cmb_dust_sync)

    # likelihood_norm_cmb_CAMB = np.array(
    #     likelihood_values_cmb_CAMB) - min(likelihood_values_cmb_CAMB)

    # likelihood_norm_cmb_no_noise = np.array(
    #     likelihood_values_cmb_no_noise) - min(likelihood_values_cmb_no_noise)

    likelihood_norm_cmb = np.array(
        likelihood_values_cmb) - min(likelihood_values_cmb)
    # likelihood_norm_cmb_dust = np.array(
    #     likelihood_values_cmb_dust) - min(likelihood_values_cmb_dust)
    # likelihood_norm_cmb_sync = np.array(
    #     likelihood_values_cmb_sync) - min(likelihood_values_cmb_sync)
    # likelihood_norm_cmb_dust_sync = np.array(
    #     likelihood_values_cmb_dust_sync) - min(likelihood_values_cmb_dust_sync)

    # likelihood_fit_norm_ = (
    #     angle_grid.value - likelihood_minimised.x[0])**2 / (likelihood_minimised.hess_inv[0][0])
    # likelihood_fit_norm = np.array(
    #     likelihood_fit_norm_) - min(likelihood_fit_norm_)

    # likelihood_fit_normH_ = (
    #     angle_grid.value - likelihood_minimised.x[0])**2 * (0.5 * H[0][0])
    # likelihood_fit_normH = np.array(
    #     likelihood_fit_normH_) - min(likelihood_fit_normH_)

    # plt.plot(angle_grid, np.exp(- likelihood_fit_norm),
    #          label='fit minimize')
    plt.plot(angle_grid, np.exp(-likelihood_norm_cmb_CAMB),
             label='CMB CAMB')
    plt.plot(angle_grid, np.exp(-likelihood_norm_cmb),
             label='CMB')
    # plt.plot(angle_grid, np.exp(-likelihood_norm_cmb_no_noise),
    #          label='CMB no noise')
    plt.plot(angle_grid, np.exp(-likelihood_norm_cmb_dust),
             label='CMB + Dust')
    # plt.plot(angle_grid, np.exp(- likelihood_fit_normH), '--',
    #          label='fit minimize + numdifftools Hessian')
    plt.plot(angle_grid, np.exp(-likelihood_norm_cmb_sync),
             label='CMB + Synchrotron')
    plt.plot(angle_grid, np.exp(-likelihood_norm_cmb_dust_sync),
             label='CMB + Dust + Synchrotron')
    plt.title('Likelihood on miscalibration with CMB and foregrounds at {} GHz and CMB only for the model'.format(
        sky_map.frequencies[dust_band]))
    plt.vlines(0, 0, 1, linestyles='--', label='true miscalibration angle')
    # plt.vlines(likelihood_minimised.x, 0, 1, linestyles='--', label='angle fit')
    # plt.vlines(likelihood_minimised.x + (1/np.sqrt(H)), 0,
    #            1, linestyles='--', label='angle fit + sigma')
    # plt.vlines(likelihood_minimised.x - (1/np.sqrt(H)), 0,
    #            1, linestyles='--', label='angle fit - sigma')

    plt.legend()
    plt.xlabel('miscalibration angle')
    plt.ylabel('likelihood')
    plt.show()

    IPython.embed()
    exit()


if __name__ == "__main__":
    main()

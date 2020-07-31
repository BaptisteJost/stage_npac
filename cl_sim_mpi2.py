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
import V3calc as V3
from mpi4py import MPI


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


def error_68(x_grid, y_grid, x_max_likelihood=0):

    total_integral = np.trapz(y_grid, x=x_grid)
    if np.isnan(total_integral):
        print('nan in x_grid or y_grid impossible to compute integral')
        return np.nan
    positive_index = np.where(x_grid >= 0 + x_max_likelihood)
    negative_index = np.where(x_grid < 0 + x_max_likelihood)
    partial_integral = 0
    i = 0
    y_array = np.array([])
    x_array = np.array([])
    while partial_integral <= 0.682689*total_integral:
        y_array = np.append(y_grid[negative_index[0][-(i+1)]], y_array)
        y_array = np.append(y_array, y_grid[positive_index[0][i]])
        x_array = np.append(
            x_grid[negative_index[0][-(i+1)]], x_array)
        x_array = np.append(x_array, x_grid[positive_index[0][i]])
        partial_integral = np.trapz(y_array, x_array)

        # partial_integral += np.exp(in_exp[positive_index[0][i]]) +
        #     np.exp(in_exp[negative_index[0][-i]])
        i += 1
    error = x_grid[positive_index[0][i]] - x_max_likelihood
    return error


def L_gridding(spectra_model, data_matrix, alpha_grid,
               bin, nside, f_sky_SAT, spectra_used='all'):

    grid_logL = []
    # grid_model_matrix = []
    # grid_data_matrix = []
    # grid_Cm1D = []
    # likelihood_element_grid = []
    for alpha in alpha_grid:
        a = cf.likelihood_for_hessian_a(alpha, spectra_model, data_matrix,
                                        bin, nside, f_sky_SAT, spectra_used)
        grid_logL.append(a)
        # grid_model_matrix.append(b.spectra)
        # grid_data_matrix.append(c.spectra)
        # grid_Cm1D.append(d)
        # likelihood_element_grid.append(e)
        # grid_logL.append(cf.likelihood_for_hessian_a(
        # alpha, spectra_model, data_matrix,
        # b, nside, f_sky_SAT, spectra_used))

    grid_logL = np.array(grid_logL)
    max_logL = max(-grid_logL[~np.isnan(grid_logL)])

    # grid_LogL_woMax = grid_logL[~np.array(-grid_logL == max(-grid_logL))]
    # in_exp_2 = -grid_LogL_woMax - max(-grid_LogL_woMax)
    # grid_alpha2 = alpha_grid[~np.array(-grid_logL == max(-grid_logL))]
    in_exp = -np.array(grid_logL) - max_logL  # max(-np.array(grid_logL))

    # , np.array(grid_model_matrix), np.array(grid_data_matrix), np.array(grid_Cm1D), np.array(likelihood_element_grid), (grid_alpha2, np.exp(in_exp_2))
    return np.exp(in_exp)


def grid_error(spectra_model, data_matrix, alpha_grid, bin, nside, f_sky_SAT,
               spectra_used='all', x_max_likelihood=0, output_grid=False):
    likelihood_grid = L_gridding(spectra_model, data_matrix, alpha_grid, bin,
                                 nside, f_sky_SAT, spectra_used=spectra_used)
    error = error_68(alpha_grid, likelihood_grid, x_max_likelihood)
    if output_grid:
        return error, likelihood_grid
    return error


def jac_n2logL(angle, spectra_model, data_matrix, bin, nside, f_sky_SAT, spectra_used='all'):
    angle = angle * u.rad
    model = copy.deepcopy(spectra_model)
    model.spectra_rotation(angle)
    model.l_min_instru = 0
    model.l_max_instru = 3*nside
    model.get_noise()
    model.get_instrument_spectra()
    model_spectra = bin.bin_cell(model.instru_spectra.spectra[:3*nside].T).T

    deriv = cf.power_spectra_obj(bin.bin_cell(lib.cl_rotation_derivative(
        spectra_model.spectra.spectra, angle)[:3*nside].T).T, bin.get_effective_ells())

    if spectra_used == 'all':
        model_matrix = cf.power_spectra_obj(np.array(
            [[model_spectra[:, 1], model_spectra[:, 4]],
             [model_spectra[:, 4], model_spectra[:, 2]]]).T, bin.get_effective_ells())
        model_inverse = np.linalg.inv(model_matrix.spectra)
        Cm1D = np.einsum('kij,kjl->kil', model_inverse, data_matrix.spectra)

        deriv_matrix = cf.power_spectra_obj(np.array(
            [[deriv.spectra[:, 1], deriv.spectra[:, 4]],
             [deriv.spectra[:, 4], deriv.spectra[:, 2]]]).T, bin.get_effective_ells())
        Cm1dC = np.einsum('kij,kjl->kil', model_inverse, deriv_matrix.spectra)
        Cm1dCCm1D = np.einsum('kij,kjl->kil', Cm1dC, Cm1D)
        jac_trace = np.trace(-Cm1dCCm1D + Cm1dC, axis1=1, axis2=2)

    elif spectra_used == 'EE':
        model_matrix = cf.power_spectra_obj(np.array(
            model_spectra[:, 1]).T, bin.get_effective_ells())
        model_inverse = 1/model_matrix.spectra
        Cm1D = model_inverse * data_matrix.spectra

        deriv_matrix = cf.power_spectra_obj(np.array(
            deriv.spectra[:, 1]).T, bin.get_effective_ells())
        Cm1dC = model_inverse * deriv_matrix.spectra
        Cm1dCCm1D = Cm1dC * Cm1D
        jac_trace = -Cm1dCCm1D + Cm1dC

    elif spectra_used == 'BB':
        model_matrix = cf.power_spectra_obj(np.array(
            model_spectra[:, 2]).T, bin.get_effective_ells())
        model_inverse = 1/model_matrix.spectra
        Cm1D = model_inverse * data_matrix.spectra

        deriv_matrix = cf.power_spectra_obj(np.array(
            deriv.spectra[:, 2]).T, bin.get_effective_ells())
        Cm1dC = model_inverse * deriv_matrix.spectra
        Cm1dCCm1D = Cm1dC * Cm1D
        jac_trace = -Cm1dCCm1D + Cm1dC

    elif spectra_used == 'EB':
        model_matrix = cf.power_spectra_obj(np.array(
            model_spectra[:, 4]).T, bin.get_effective_ells())
        model_inverse = 1/model_matrix.spectra
        Cm1D = model_inverse * data_matrix.spectra

        deriv_matrix = cf.power_spectra_obj(np.array(
            deriv.spectra[:, 4]).T, bin.get_effective_ells())
        Cm1dC = model_inverse * deriv_matrix.spectra
        Cm1dCCm1D = Cm1dC * Cm1D
        jac_trace = -Cm1dCCm1D + Cm1dC
    jac = 0
    ell_counter = 0
    jac_element = []
    for l in bin.get_effective_ells():
        jac_ell = f_sky_SAT*(2*l + 1)*0.5 * jac_trace[ell_counter]
        jac += jac_ell
        jac_element.append(jac_ell)
        ell_counter += 1
    jac = np.array([jac])
    print(jac)
    return jac


def noise_maps_simulation(sensitivity, knee_mode, ny_lf, nside,
                          norm_hits_map_path, no_inh, frequencies_index):
    nhits, noise_maps_, nlev = mknm.get_noise_sim(
        sensitivity=sensitivity, knee_mode=knee_mode,
        ny_lf=ny_lf, nside_out=nside,
        norm_hits_map=hp.read_map(norm_hits_map_path), no_inh=no_inh)
    noise_maps = []

    # noise_maps_masked = []

    for i in frequencies_index:
        noise_maps.append([noise_maps_[i*3],
                           noise_maps_[i*3+1],
                           noise_maps_[i*3+2]])
        # noise_maps_masked.append([noise_maps_[i*3] * mask,
        #                           noise_maps_[i*3+1] * mask,
        #                           noise_maps_[i*3+2] * mask])
    noise_maps = np.array(noise_maps)
    # noise_maps_masked = np.array(noise_maps_masked)

    del noise_maps_
    return noise_maps, nhits


def get_foreground_maps_and_cl(sky_obj, miscalibration_angle, frequencies2use,
                               mask, mask_apo, wsp, purify_e, purify_b,
                               return_dust=True, return_synchrotron=True,
                               return_maps=False):
    dust_map_freq = []
    sync_map_freq = []
    Cl_dust_freq = []
    Cl_sync_freq = []

    for f in frequencies2use:
        if return_dust:
            dust_maps_ = sky_obj.sky.dust(f) * \
                convert_units('K_RJ', 'K_CMB', f)
            dust_maps = lib.map_rotation(dust_maps_, miscalibration_angle)
            dust_map_freq.append(dust_maps)
            f2_dust = get_field(mask*dust_maps[1, :],
                                mask*dust_maps[2, :], mask_apo,
                                purify_e=purify_e, purify_b=purify_b)
            Cl_dust_freq.append(compute_master(f2_dust, f2_dust, wsp))

        if return_synchrotron:
            sync_maps_ = sky_obj.sky.synchrotron(f) * \
                convert_units('K_RJ', 'K_CMB', f)
            sync_maps = lib.map_rotation(sync_maps_, miscalibration_angle)
            sync_map_freq.append(sync_maps)

            f2_sync = get_field(mask*sync_maps[1, :],
                                mask*sync_maps[2, :], mask_apo,
                                purify_e=purify_e, purify_b=purify_b)
            Cl_sync_freq.append(compute_master(f2_sync, f2_sync, wsp))

    if return_dust:
        dust_map_freq = np.array(dust_map_freq)
        Cl_dust_freq = np.array(Cl_dust_freq)

    if return_synchrotron:
        sync_map_freq = np.array(sync_map_freq)
        Cl_sync_freq = np.array(Cl_sync_freq)
    if return_maps:
        if return_dust and return_synchrotron:
            return Cl_dust_freq, Cl_sync_freq, dust_map_freq, sync_map_freq
        elif return_dust:
            return Cl_dust_freq, dust_map_freq
        elif return_synchrotron:
            return Cl_sync_freq, sync_map_freq
    else:
        return Cl_dust_freq, Cl_sync_freq


def get_nsim_freq_cl(input_map, nsim, frequencies_index, mask, mask_apo,
                     purify_e, purify_b, wsp):

    Cl_nsim_freq = []
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print('SIZE = ', size)
    print('nside =', nsim)
    root = 0
    Cl_freq = []
    for f in range(len(frequencies_index)):
        f2 = get_field(mask*input_map[f, 1, :],
                       mask*input_map[f, 2, :], mask_apo,
                       purify_e=purify_e, purify_b=purify_b)
        Cl_freq.append(compute_master(f2, f2, wsp))
    print('shape Cl_freq=', np.shape(Cl_freq))
    Cl_freq = np.array(Cl_freq)

    # shape_ = np.array(np.shape(Cl_freq))
    # shape = np.append(nsim, shape_).tolist()
    # print('shape shape = ', shape)
    #
    # Cl_nsim_freq = None
    # if rank == 0:
    #     Cl_nsim_freq = np.empty(shape)
    # print('shape Cl_nsim_freq', np.shape(Cl_nsim_freq))
    # comm.Gather(Cl_freq, Cl_nsim_freq, root)
    # del Cl_freq
    #
    # if rank == 0:
    #     Cl_nsim_freq = np.array(Cl_nsim_freq)
    #
    # else:
    #     Cl_nsim_freq = np.empty(shape)
    # comm.Bcast(Cl_nsim_freq, root)
    # Cl_nsim_freq = np.array(Cl_nsim_freq)
    # print('Cl_nsim_freq shape = ', Cl_nsim_freq.shape)

    # Cl_nsim_freq = comm.gather(Cl_freq, root)
    # Cl_nsim_freq.append(Cl_freq)
    # Cl_nsim_freq = np.array(Cl_nsim_freq)
    # Cl_nsim_freq = comm.bcast(Cl_nsim_freq, root)

    return Cl_freq


def min_and_error_nsim_freq(
        model, data_cl, nsim, frequencies_index, bin, nside, f_sky_SAT,
        spectra_used='all', spectra_indexation='NaMaster',
        minimisation_init=0.001,
        compute_error68=False, step_size=1e-4, output_grid=False):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    root = 0
    if spectra_indexation == 'NaMaster':
        indexation = {'EE': 0, 'BB': 3, 'EB': 1}
    elif spectra_indexation == 'CAMB':
        indexation = {'EE': 1, 'BB': 2, 'EB': 4}
    minimisation_nsim_freq = []
    H_nsim_freq = []
    if compute_error68:
        error_nsim_freq = []
        if output_grid:
            grid_nsim_freq = []

    minimisation_freq = []
    H_freq = []
    if compute_error68:
        error_freq = []
        if output_grid:
            grid_freq = []

    for f in range(len(frequencies_index)):
        if spectra_used == 'all':
            data_matrix = cf.power_spectra_obj(np.array(
                [[data_cl[f, indexation['EE']], data_cl[f, indexation['EB']]],
                 [data_cl[f, indexation['EB']], data_cl[f, indexation['BB']]]]).T,
                bin.get_effective_ells())
        else:
            data_matrix = cf.power_spectra_obj(np.array(
                data_cl[f, indexation[spectra_used]]).T,
                bin.get_effective_ells())

        minimisation_likelihood_results = minimize(
            cf.likelihood_for_hessian_a, minimisation_init,
            (model, data_matrix, bin, nside, f_sky_SAT, spectra_used),
            jac=jac_n2logL)
        H = nd.Hessian(cf.likelihood_for_hessian_a)(
            minimisation_likelihood_results.x,
            model, data_matrix, bin,
            nside, f_sky_SAT, spectra_used)
        minimisation_freq.append(minimisation_likelihood_results.x[0])
        H_freq.append(H[0][0])
        if compute_error68:
            min_alpha = minimisation_likelihood_results.x[0] - 6/np.sqrt(H[0][0])
            max_alpha = minimisation_likelihood_results.x[0] + 6/np.sqrt(H[0][0])
            alpha_grid = np.arange(min_alpha, max_alpha, step_size)
            error_ = grid_error(
                model, data_matrix, alpha_grid, bin, nside, f_sky_SAT,
                spectra_used=spectra_used,
                x_max_likelihood=minimisation_likelihood_results.x[0],
                output_grid=output_grid)
            if output_grid:
                error_freq.append(error_[0])
                grid_freq.append(error_[1])
            else:
                error_freq.append(error_)

    # minimisation_nsim_freq = comm.gather(minimisation_freq, root)
    # minimisation_nsim_freq.append(minimisation_freq)
    # H_nsim_freq = comm.gather(H_freq, root)
    # H_nsim_freq.append(H_freq)
    # del minimisation_freq
    # del H_freq
    minimisation_freq = np.array(minimisation_freq)
    H_freq = np.array(H_freq)
    # minimisation_nsim_freq = None
    # H_nsim_freq = None
    #
    # if rank == 0:
    #     minimisation_nsim_freq = np.empty([nsim, minimisation_freq.shape[0]])
    #     H_nsim_freq = np.empty([nsim, H_freq.shape[0]])
    #
    # comm.Gather(minimisation_freq, minimisation_nsim_freq, root)
    # comm.Gather(H_freq, H_nsim_freq, root)
    #
    # if rank == 0:
    #     minimisation_nsim_freq = np.array(minimisation_nsim_freq)
    #     H_nsim_freq = np.array(H_nsim_freq)
    #
    # else:
    #     minimisation_nsim_freq = np.empty([nsim, minimisation_freq.shape[0]])
    #     H_nsim_freq = np.empty([nsim, H_freq.shape[0]])
    #
    # comm.Bcast(minimisation_nsim_freq, root)
    # comm.Bcast(H_nsim_freq, root)
    #
    # del minimisation_freq
    # del H_freq
    # minimisation_nsim_freq = np.array(minimisation_nsim_freq)
    # H_nsim_freq = np.array(H_nsim_freq)

    if compute_error68:
        error_nsim_freq = comm.gather(error_freq, root)
        # error_nsim_freq.append(error_freq)
        del error_freq
        if output_grid:
            grid_nsim_freq = comm.gather(grid_freq, root)
            # grid_nsim_freq.append(grid_freq)
            del grid_freq

    # minimisation_nsim_freq = comm.bcast(minimisation_nsim_freq, root)
    # minimisation_nsim_freq = np.array(minimisation_nsim_freq)

    # H_nsim_freq = comm.bcast(H_nsim_freq, root)
    # H_nsim_freq = np.array(H_nsim_freq)
    if compute_error68:
        error_nsim_freq = comm.bcast(error_nsim_freq, root)
        error_nsim_freq = np.array(error_nsim_freq)

        if output_grid:
            grid_nsim_freq = comm.bcast(grid_nsim_freq, root)
            grid_nsim_freq = np.array(grid_nsim_freq)
    if compute_error68 and output_grid:
        return minimisation_nsim_freq, H_nsim_freq, error_nsim_freq, grid_nsim_freq
    elif compute_error68:
        return minimisation_nsim_freq, H_nsim_freq, error_nsim_freq

    return minimisation_freq, H_freq


def main():
    comm = MPI.COMM_WORLD
    mpi_rank = MPI.COMM_WORLD.Get_rank()
    nsim = comm.Get_size()
    print(mpi_rank, nsim)

    root = 0
    purify_e = False
    rotation_angle = (3 * u.deg).to(u.rad)
    miscalibration_angle = (0 * u.deg).to(u.rad)
    r = 0.0
    pysm_model = 'c1s0d0'
    f_sky_SAT = 0.1
    sensitivity = 0
    knee_mode = 0
    BBPipe_path = '/global/homes/j/jost/BBPipe'
    # BBPipe_path = '/home/baptiste/BBPipe'
    save_path = '/global/homes/j/jost/test_namaster/96simu_93Ghz_alpha3deg/'

    norm_hits_map_path = BBPipe_path + '/test_mapbased_param/norm_nHits_SA_35FOV_G_nside512.fits'
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
    # IPython.embed()
    true_spectra = cf.power_spectra_obj(spectra_cl.cl_rot.spectra,
                                        spectra_cl.cl_rot.ell)
    true_spectra.ell = np.arange(lmin, lmax)

    sky_map = l_SO.sky_map(nside=nside, sky_model=pysm_model)
    sky_map.get_pysm_sky()
    sky_map.get_frequency()
    frequencies2use = [93]
    frequencies_index = []
    for f in frequencies2use:
        frequencies_index.append(sky_map.frequencies.tolist().index(f))

    noise_maps = []
    np.random.seed(mpi_rank)
    noise_maps_sim, nhits = noise_maps_simulation(
        sensitivity, knee_mode, ny_lf, nside,
        norm_hits_map_path, no_inh, frequencies_index)
    print('noise_maps_sim shape = ', noise_maps_sim.shape)

    noise_maps = np.array(noise_maps_sim)
    # if mpi_rank == 0:
    #     noise_maps = np.empty([nsim, noise_maps_sim.shape[0],
    #                            noise_maps_sim.shape[1], noise_maps_sim.shape[2]])
    #
    # comm.Gather(noise_maps_sim, noise_maps, root)
    # if mpi_rank == 0:
    #     noise_maps = np.array(noise_maps)
    #     print('noise_maps shape = ', noise_maps.shape)
    # else:
    #     noise_maps = np.empty([nsim, noise_maps_sim.shape[0],
    #                            noise_maps_sim.shape[1], noise_maps_sim.shape[2]])
    # comm.Bcast(noise_maps, root)
    # print('noise_maps shape = ', noise_maps.shape)
    #
    # # noise_maps = comm.gather(noise_maps_sim, root)
    # # noise_maps = comm.bcast(noise_maps, root)
    #
    # # noise_maps.append(noise_maps_sim)
    del noise_maps_sim
    # noise_maps = np.array(noise_maps)
    # if comm.rank == 0:
    #     np.save('noise_maps.npy', noise_maps)
    print('noise_maps shape', noise_maps.shape)
    print('noise_maps', noise_maps)

    print('noise_maps type', type(noise_maps))
    # exit()
    print('building mask ... ')
    mask_ = hp.read_map(BBPipe_path +
                        "/test_mapbased_param/norm_nHits_SA_35FOV_G_nside512_binary.fits")
    mask = hp.ud_grade(mask_, nside)
    mask[np.where(nhits < 1e-6)[0]] = 0.0

    '''***********************NAMASTER INITIALISATION***********************'''

    print('initializing Namaster ...')
    start_namaster = time.time()
    wsp = nmt.NmtWorkspace()
    b = binning_definition(nside, lmin=lmin, lmax=lmax,
                           nlb=nlb, custom_bins=custom_bins)

    del mask_
    mask_apo = nmt.mask_apodization(
        mask, aposize, apotype=apotype)
    nh = hp.smoothing(nhits, fwhm=1*np.pi/180.0, verbose=False)
    nh /= nh.max()
    mask_apo *= nh

    cltt, clee, clbb, clte = hp.read_cl(BBPipe_path +
                                        "/test_mapbased_param/Cls_Planck2018_lensed_scalar.fits")[:, :4000]
    mp_t_sim, mp_q_sim, mp_u_sim = hp.synfast(
        [cltt, clee, clbb, clte], nside=nside, new=True, verbose=False)
    f2y0 = get_field(mp_q_sim, mp_u_sim, mask_apo)
    wsp.compute_coupling_matrix(f2y0, f2y0, b)

    print('Namaster initialized in {}s'.format(time.time() - start_namaster))

    """____________________________map creation____________________________"""
    map_creation = 1
    if map_creation:
        print("creating maps ...")

        print("creating foregrounds maps ...")
        start_map = time.time()
        Cl_dust, Cl_sync, dust_map_freq, sync_map_freq = get_foreground_maps_and_cl(
            sky_map, miscalibration_angle,
            frequencies2use, mask, mask_apo, wsp,
            purify_e, purify_b, return_dust=True,
            return_synchrotron=True,
            return_maps=True)
        # dust_map_freq = dust_map_freq * 100
        # sync_map_freq = sync_map_freq * 100
        # print('mask shape', mask.shape)
        # print('mask type', type(mask))
        # print('mask*noise_maps shape', (mask*noise_maps).shape)
        # print('mask*noise_maps type', type(mask*noise_maps))
        Cl_noise = get_nsim_freq_cl(mask*noise_maps, nsim, frequencies_index, mask,
                                    mask_apo, purify_e, purify_b, wsp)
        # for f in range(len(frequencies2use)):
        #     noise_maps_for_cl = noise_maps[f]
        #     f2_noise = get_field(mask*noise_maps_for_cl[1, :],
        #                          mask*noise_maps_for_cl[2, :], mask_apo,
        #                          purify_e=purify_e, purify_b=purify_b)
        #     Cl_noise.append(compute_master(f2_noise, f2_noise, wsp))
        # Cl_noise = np.array(Cl_noise)

        # IPython.embed()

        cmb_map_nsim = []
        print("creating CMB map simulations ...")
        print('WARNING binning_definition sets seed, CMB map created before, need proper solution')
        print('WARNING: for now seed will always be the same for debugging purposes')
        # for i in range(nsim):
        # np.random.seed(int(time.time()))
        np.random.seed(mpi_rank)
        cmb_map_freq = []
        for f in range(len(frequencies2use)):
            cmb_map = hp.synfast(spectra_cl.cl_rot.spectra.T, nside, new=True)
            cmb_map_freq.append(cmb_map + noise_maps[f])
        import os
        import psutil
        process = psutil.Process(os.getpid())
        memory = process.memory_info().rss * u.byte
        print('Memory used = ', memory.to(u.Gbyte))
        print('Memory/size * 32 = ', (memory*32/nsim).to(u.Gbyte))

        del noise_maps
        cmb_map_freq = np.array(cmb_map_freq)
        # cmb_map_nsim = comm.gather(cmb_map_freq, root)
        # cmb_map_nsim = comm.bcast(cmb_map_nsim, root)

        cmb_map_nsim = cmb_map_freq
        # if mpi_rank == 0:
        #     cmb_map_nsim = np.empty([nsim, cmb_map_freq.shape[0],
        #                              cmb_map_freq.shape[1], cmb_map_freq.shape[2]])
        # comm.Gather(cmb_map_freq, cmb_map_nsim, root)
        #
        # if mpi_rank == 0:
        #     cmb_map_nsim = np.array(cmb_map_nsim)
        #     print('cmb_map_nsim shape = ', cmb_map_nsim.shape)
        # else:
        #     cmb_map_nsim = np.empty([nsim, cmb_map_freq.shape[0],
        #                              cmb_map_freq.shape[1], cmb_map_freq.shape[2]])
        # comm.Bcast(cmb_map_nsim, root)
        # print('cmb_map_nsim shape = ', cmb_map_nsim.shape)

        # del cmb_map_freq
        # cmb_map_nsim = np.array(cmb_map_nsim)

        # cmb_map_nsim.append(cmb_map_freq)

        # del cmb_map_freq
        # print('shape noise maps=', np.shape(noise_maps))
        # cmb_map_nsim = np.array(cmb_map_nsim)
        if comm.rank == 0:
            np.save(save_path + 'cmb_map_nsim.npy', cmb_map_nsim)
        print("")
        print("WARNING : noise maps added directly to cmb map")
        print("IT WILL CAUSE PB when multiple frequencies are used")
        print("Component maps created successfully in {} s".format(
            time.time() - start_map))

    """____________________________Cl estimation____________________________"""
    Cl_estimation = 1
    if Cl_estimation:
        start_Cl = time.time()
        Cl_cmb = get_nsim_freq_cl(mask*cmb_map_nsim, nsim, frequencies_index, mask,
                                  mask_apo, purify_e, purify_b, wsp)
        cmb_dust_map = np.add(cmb_map_nsim, dust_map_freq)
        Cl_cmb_dust = get_nsim_freq_cl(mask*cmb_dust_map, nsim,
                                       frequencies_index, mask, mask_apo,
                                       purify_e, purify_b, wsp)

        cmb_sync_map = np.add(cmb_map_nsim, sync_map_freq)
        Cl_cmb_sync = get_nsim_freq_cl(mask*cmb_sync_map, nsim,
                                       frequencies_index, mask, mask_apo,
                                       purify_e, purify_b, wsp)

        cmb_dust_sync_map = np.add(cmb_dust_map, sync_map_freq)
        Cl_cmb_dust_sync = get_nsim_freq_cl(mask*cmb_dust_sync_map, nsim,
                                            frequencies_index, mask, mask_apo,
                                            purify_e, purify_b, wsp)
        print('time for Cl estimation of of different component maps = ',
              time.time()-start_Cl)

        noise_nl_ = V3.so_V3_SA_noise(sensitivity, knee_mode, 1, f_sky_SAT, 3*nside)[1]
        noise_nl = np.insert(noise_nl_[frequencies_index], 0, np.zeros([2]))
        noise_nl = b.bin_cell(noise_nl[:3*nside]).T

        if comm.rank == 0:
            np.save(save_path + 'Cl_cmb.npy', Cl_cmb)
            np.save(save_path + 'Cl_cmb_dust.npy', Cl_cmb_dust)
            np.save(save_path + 'Cl_cmb_sync.npy', Cl_cmb_sync)
            np.save(save_path + 'Cl_cmb_dust_sync.npy', Cl_cmb_dust_sync)

        # IPython.embed()

    fl_plt_Cl1sp = 0
    if fl_plt_Cl1sp:
        plt.plot(b.get_effective_ells(), Cl_cmb[0][1], label='CMB')
        # plt.plot(b.get_effective_ells(), Cl_cmb[1][1] + Cl_noise[0][1], label='CMB + Cl noise')
        plt.plot(b.get_effective_ells(), Cl_cmb[1][1], label='CMB + noise map')
        plt.plot(b.get_effective_ells(), Cl_noise[0][1], label='Noise')
        # plt.plot(b.get_effective_ells(), noise_nl, label='V3 calc nl')
        # plt.plot(b.get_effective_ells(), noise_nl.T[1], label='V3 calc nl')
        plt.plot(b.get_effective_ells(), Cl_dust[0][1], label='Dust')
        plt.plot(b.get_effective_ells(), Cl_sync[0][1], label='Synchrotron')

        plt.legend()
        # plt.yscale('log')
        plt.xscale('log')
        plt.ylabel(r'$C_{\ell}^{EE}$')
        plt.xlabel(r'$\ell$')
        plt.show()

    # IPython.embed()
    # Cl0_dust = np.array(Cl0_dust)
    """________________________test nsim likelihood________________________"""
    min_angle = -0.005  # rotation.value - 5*(1/np.sqrt(fishaa))
    max_angle = 0.005  # rotation.value + 5*(1/np.sqrt(fishaa))
    nstep_angle = 1000
    angle_grid = np.arange(min_angle, max_angle,
                           (max_angle - min_angle)/nstep_angle)*u.radian

    # likelihood_values_cmb_nsim = []
    rotation_angle_model_init = rotation_angle  # 0*u.rad
    spectra_cl = cf.power_spectra_operation(r=r,
                                            rotation_angle=rotation_angle_model_init,
                                            powers_name='total')
    spectra_cl.get_spectra()
    spectra_cl.spectra_rotation()
    save_cmb_model = copy.deepcopy(b.bin_cell(
        copy.deepcopy(spectra_cl.cl_rot.spectra[:3*nside].T)).T)
    save_cmb_model_nobin = copy.deepcopy(spectra_cl.cl_rot.spectra[:3*nside].T)
    spectra_cl.l_min_instru = 0
    spectra_cl.l_max_instru = 3*nside
    spectra_cl.get_noise()
    spectra_cl.get_instrument_spectra()

    # data_model = spectra_cl.instru_spectra.spectra
    data_model = b.bin_cell(copy.deepcopy(spectra_cl.instru_spectra.spectra[:3*nside]).T).T
    data_model_nobin = copy.deepcopy(spectra_cl.instru_spectra.spectra[:3*nside].T)
    np.save(save_path + 'data_model_noise.npy', data_model)
    np.save(save_path + 'data_model_nobin_noise.npy', data_model_nobin)

    # data_model = copy.deepcopy(spectra_cl.instru_spectra.spectra)
    # data_model_nobin = copy.deepcopy(spectra_cl.instru_spectra.spectra[:3*nside].T)

    data_model_matrix = cf.power_spectra_obj(np.array(
        [[data_model[:, 1], data_model[:, 4]],
         [data_model[:, 4], data_model[:, 2]]]).T, b.get_effective_ells())
    data_model_matrix_EE = cf.power_spectra_obj(np.array(
        data_model[:, 1]).T, b.get_effective_ells())
    data_model_matrix_BB = cf.power_spectra_obj(np.array(
        data_model[:, 2]).T, b.get_effective_ells())
    data_model_matrix_EB = cf.power_spectra_obj(np.array(
        data_model[:, 4]).T, b.get_effective_ells())

    deriv = cf.power_spectra_obj(b.bin_cell(lib.cl_rotation_derivative(
        spectra_cl.spectra.spectra, rotation_angle)[:3*nside].T).T, b.get_effective_ells())
    deriv_matrix = cf.power_spectra_obj(np.array(
        [[deriv.spectra[:, 1], deriv.spectra[:, 4]],
         [deriv.spectra[:, 4], deriv.spectra[:, 2]]]).T, deriv.ell)
    fisher_modeldata = cf.fisher_pws(data_model_matrix, deriv_matrix, f_sky_SAT)

    deriv_matrix_EE = cf.power_spectra_obj(
        deriv.spectra[:, 1].T, deriv.ell)
    fisher_modeldata_EE = cf.fisher_pws(data_model_matrix_EE, deriv_matrix_EE, f_sky_SAT)
    fisher_sigma_EE = 1/np.sqrt(fisher_modeldata_EE)

    deriv_matrix_BB = cf.power_spectra_obj(
        deriv.spectra[:, 2].T, deriv.ell)
    fisher_modeldata_BB = cf.fisher_pws(data_model_matrix_BB, deriv_matrix_BB, f_sky_SAT)
    fisher_sigma_BB = 1/np.sqrt(fisher_modeldata_BB)

    deriv_matrix_EB = cf.power_spectra_obj(
        deriv.spectra[:, 4].T, deriv.ell)
    fisher_modeldata_EB = cf.fisher_pws(data_model_matrix_EB, deriv_matrix_EB, f_sky_SAT)
    fisher_sigma_EB = 1/np.sqrt(fisher_modeldata_EB)
    EB_spectra = copy.deepcopy(data_model_matrix_EB.spectra)
    # IPython.embed()
    # data_model_matrix = cf.power_spectra_obj(np.array(
    #     [[data_model[:, 1], data_model[:, 4]],
    #      [data_model[:, 4], data_model[:, 2]]]).T, spectra_cl.instru_spectra.ell)
    # data_model_matrix_EE = cf.power_spectra_obj(np.array(
    #     data_model[:, 1]).T, spectra_cl.instru_spectra.ell)
    # data_model_matrix = cf.power_spectra_obj(np.array(
    #     [[data_model[:, 1]]]).T, b.get_effective_ells())
    #
    data_matrix_noise = cf.power_spectra_obj(
        np.array([[Cl_noise[0].T[:, 0], Cl_noise[0].T[:, 2]],
                  [Cl_noise[0].T[:, 1], Cl_noise[0].T[:, 3]]]).T,
        b.get_effective_ells())
    minimisation = 1
    if minimisation:
        # likelihood_minimised_noise = minimize(
        #     cf.likelihood_for_hessian_a,    0,
        #     (spectra_cl, data_matrix_noise, b, nside, f_sky_SAT))
        # H_noise_ = nd.Hessian(cf.likelihood_for_hessian_a)(
        #     likelihood_minimised_noise.x,
        #     spectra_cl, data_matrix_noise, b,
        #     nside, f_sky_SAT)
        # fit_alpha_freq_noise = []
        # H_freq_noise = []
        # fit_alpha_freq_noise.append(likelihood_minimised_noise.x[0])
        # H_freq_noise.append(H_noise_[0][0])

        likelihood_minimised_model = minimize(
            cf.likelihood_for_hessian_a,    0.001,
            (spectra_cl, data_model_matrix, b, nside, f_sky_SAT),
            jac=jac_n2logL)
        H_model_ = nd.Hessian(cf.likelihood_for_hessian_a)(
            likelihood_minimised_model.x,
            spectra_cl, data_model_matrix, b,
            nside, f_sky_SAT)
        fit_alpha_model = []
        H_model = []
        fit_alpha_model.append(likelihood_minimised_model.x[0])
        H_model.append(H_model_[0][0])

        # likelihood_minimised_model_EE = minimize(
        #     cf.likelihood_for_hessian_a,    0.0,
        #     (spectra_cl, data_model_matrix_EE, b, nside, f_sky_SAT, 'EE'),
        #     jac=jac_n2logL)
        # H_model_EE = nd.Hessian(cf.likelihood_for_hessian_a)(
        #     likelihood_minimised_model_EE.x,
        #     spectra_cl, data_model_matrix_EE, b,
        #     nside, f_sky_SAT, 'EE')
        #
        # fit_alpha_model.append(likelihood_minimised_model_EE.x[0])
        # H_model.append(H_model_EE[0][0])
        #
        # likelihood_minimised_model_BB = minimize(
        #     cf.likelihood_for_hessian_a,    0.0,
        #     (spectra_cl, data_model_matrix_BB, b, nside, f_sky_SAT, 'BB'),
        #     jac=jac_n2logL)
        # H_model_BB = nd.Hessian(cf.likelihood_for_hessian_a)(
        #     likelihood_minimised_model_BB.x,
        #     spectra_cl, data_model_matrix_BB, b,
        #     nside, f_sky_SAT, 'BB')
        #
        # fit_alpha_model.append(likelihood_minimised_model_BB.x[0])
        # H_model.append(H_model_BB[0][0])
        #
        # likelihood_minimised_model_EB = minimize(
        #     cf.likelihood_for_hessian_a,    0.0,
        #     (spectra_cl, data_model_matrix_EB, b, nside, f_sky_SAT, 'EB'),
        #     jac=jac_n2logL)
        # H_model_EB = nd.Hessian(cf.likelihood_for_hessian_a)(
        #     likelihood_minimised_model_EB.x,
        #     spectra_cl, data_model_matrix_EB, b,
        #     nside, f_sky_SAT, 'EB')
        #
        # fit_alpha_model.append(likelihood_minimised_model_EB.x[0])
        # H_model.append(H_model_EB[0][0])

        # fit_alpha_cmb = []
        # H_cmb = []
        # fit_alpha_cmb_BB = []
        # H_cmb_BB = []
        # fit_alpha_cmb_EE = []
        # H_cmb_EE = []
        # fit_alpha_cmb_EB = []
        # H_cmb_EB = []

        # fit_alpha_cmb_dust = []
        # H_cmb_dust = []
        # fit_alpha_cmb_sync = []
        # H_cmb_sync = []
        # fit_alpha_cmb_dust_sync = []
        # H_cmb_dust_sync = []

        fit_alpha_cmb, H_cmb = min_and_error_nsim_freq(
            spectra_cl, Cl_cmb, nsim, frequencies_index, b, nside, f_sky_SAT,
            spectra_used='all', spectra_indexation='NaMaster',
            minimisation_init=0.001,
            compute_error68=False, step_size=1e-5, output_grid=False)
        fit_alpha_cmb_dust, H_cmb_dust = min_and_error_nsim_freq(
            spectra_cl, Cl_cmb_dust, nsim, frequencies_index, b, nside, f_sky_SAT,
            spectra_used='all', spectra_indexation='NaMaster',
            minimisation_init=0.001,
            compute_error68=False, step_size=1e-5, output_grid=False)
        fit_alpha_cmb_sync, H_cmb_sync = min_and_error_nsim_freq(
            spectra_cl, Cl_cmb_sync, nsim, frequencies_index, b, nside, f_sky_SAT,
            spectra_used='all', spectra_indexation='NaMaster',
            minimisation_init=0.001,
            compute_error68=False, step_size=1e-5, output_grid=False)
        fit_alpha_cmb_dust_sync, H_cmb_dust_sync = min_and_error_nsim_freq(
            spectra_cl, Cl_cmb_sync, nsim, frequencies_index, b, nside, f_sky_SAT,
            spectra_used='all', spectra_indexation='NaMaster',
            minimisation_init=0.001,
            compute_error68=False, step_size=1e-5, output_grid=False)

        fit_alpha_cmb_nsim = None
        H_cmb_nsim = None
        fit_alpha_cmb_dust_nsim = None
        H_cmb_dust_nsim = None
        fit_alpha_cmb_sync_nsim = None
        H_cmb_sync_nsim = None
        fit_alpha_cmb_dust_sync_nsim = None
        H_cmb_dust_sync_nsim = None

        if comm.rank == 0:
            fit_alpha_cmb_nsim = np.empty([nsim, fit_alpha_cmb.shape[0]])
            H_cmb_nsim = np.empty([nsim, H_cmb.shape[0]])

            fit_alpha_cmb_dust_nsim = np.empty([nsim, fit_alpha_cmb_dust.shape[0]])
            H_cmb_dust_nsim = np.empty([nsim, H_cmb_dust.shape[0]])

            fit_alpha_cmb_sync_nsim = np.empty([nsim, fit_alpha_cmb_sync.shape[0]])
            H_cmb_sync_nsim = np.empty([nsim, H_cmb_sync.shape[0]])

            fit_alpha_cmb_dust_sync_nsim = np.empty([nsim, fit_alpha_cmb_dust_sync.shape[0]])
            H_cmb_dust_sync_nsim = np.empty([nsim, H_cmb_dust_sync.shape[0]])

        comm.Gather(fit_alpha_cmb, fit_alpha_cmb_nsim, root)
        comm.Gather(H_cmb, H_cmb_nsim, root)

        comm.Gather(fit_alpha_cmb_dust, fit_alpha_cmb_dust_nsim, root)
        comm.Gather(H_cmb_dust, H_cmb_dust_nsim, root)

        comm.Gather(fit_alpha_cmb_sync, fit_alpha_cmb_sync_nsim, root)
        comm.Gather(H_cmb_sync, H_cmb_sync_nsim, root)

        comm.Gather(fit_alpha_cmb_dust_sync, fit_alpha_cmb_dust_sync_nsim, root)
        comm.Gather(H_cmb_dust_sync, H_cmb_dust_sync_nsim, root)

        if comm.rank == 0:

            np.save(save_path + 'fit_alpha_cmb.npy', fit_alpha_cmb_nsim)
            np.save(save_path + 'H_cmb.npy', H_cmb_nsim)

            np.save(save_path + 'fit_alpha_cmb_dust.npy', fit_alpha_cmb_dust_nsim)
            np.save(save_path + 'H_cmb_dust.npy', H_cmb_dust_nsim)

            np.save(save_path + 'fit_alpha_cmb_sync.npy', fit_alpha_cmb_sync_nsim)
            np.save(save_path + 'H_cmb_sync.npy', H_cmb_sync_nsim)

            np.save(save_path + 'fit_alpha_cmb_dust_sync.npy', fit_alpha_cmb_dust_sync_nsim)
            np.save(save_path + 'H_cmb_dust_sync.npy', H_cmb_dust_sync_nsim)
        exit()
        # IPython.embed()

        # for i in range(nsim):
        #     fit_alpha_freq_cmb = []
        #     H_freq_cmb = []
        #     fit_alpha_freq_cmb_BB = []
        #     H_freq_cmb_BB = []
        #     fit_alpha_freq_cmb_EE = []
        #     H_freq_cmb_EE = []
        #     fit_alpha_freq_cmb_EB = []
        #     H_freq_cmb_EB = []
        #
        #     fit_alpha_freq_cmb_dust = []
        #     H_freq_cmb_dust = []
        #     fit_alpha_freq_cmb_sync = []
        #     H_freq_cmb_sync = []
        #     fit_alpha_freq_cmb_dust_sync = []
        #     H_freq_cmb_dust_sync = []
        #     for f in range(len(frequencies2use)):
        #         # if i == 0:
        #         #     data_spectra_cmb = Cl_cmb[i].T
        #         # if i == 1:
        #         #     data_spectra_cmb = Cl_cmb[i].T + Cl_noise[0].T
        #         # if i == 2:
        #         #     data_spectra_cmb = Cl_cmb[i].T
        #
        #         data_spectra_cmb = Cl_cmb[i].T
        #
        #         data_spectra_cmb_dust = Cl_cmb_dust[i, f].T
        #         data_spectra_cmb_sync = Cl_cmb_sync[i, f].T
        #         data_spectra_cmb_dust_sync = Cl_cmb_dust_sync[i, f].T
        #
        #         data_matrix_cmb = cf.power_spectra_obj(
        #             np.array([[data_spectra_cmb[:, 0], data_spectra_cmb[:, 2]],
        #                       [data_spectra_cmb[:, 1], data_spectra_cmb[:, 3]]]).T,
        #             b.get_effective_ells())
        #
        #         data_matrix_cmb_EE = cf.power_spectra_obj(
        #             np.array(data_spectra_cmb[:, 0]).T, b.get_effective_ells())
        #
        #         data_matrix_cmb_BB = cf.power_spectra_obj(
        #             np.array(data_spectra_cmb[:, 3]).T, b.get_effective_ells())
        #         data_matrix_cmb_EB = cf.power_spectra_obj(
        #             np.array(data_spectra_cmb[:, 1]).T, b.get_effective_ells())
        # IPython.embed()
        # data_matrix_cmb_dust = cf.power_spectra_obj(
        #     np.array([[data_spectra_cmb_dust[:, 0], data_spectra_cmb_dust[:, 2]],
        #               [data_spectra_cmb_dust[:, 1], data_spectra_cmb_dust[:, 3]]]).T,
        #     b.get_effective_ells())
        # data_matrix_cmb_sync = cf.power_spectra_obj(
        #     np.array([[data_spectra_cmb_sync[:, 0], data_spectra_cmb_sync[:, 2]],
        #               [data_spectra_cmb_sync[:, 1], data_spectra_cmb_sync[:, 3]]]).T,
        #     b.get_effective_ells())
        # data_matrix_cmb_dust_sync = cf.power_spectra_obj(
        #     np.array([[data_spectra_cmb_dust_sync[:, 0], data_spectra_cmb_dust_sync[:, 2]],
        #               [data_spectra_cmb_dust_sync[:, 1], data_spectra_cmb_dust_sync[:, 3]]]).T,
        #     b.get_effective_ells())

        # likelihood_minimised_cmb = minimize(
        #     cf.likelihood_for_hessian_a,    0,
        #     (spectra_cl, data_matrix_cmb, b, nside, f_sky_SAT))
        # H_cmb_ = nd.Hessian(cf.likelihood_for_hessian_a)(
        #     likelihood_minimised_cmb.x,
        #     spectra_cl, data_matrix_cmb, b,
        #     nside, f_sky_SAT)
        # fit_alpha_freq_cmb.append(likelihood_minimised_cmb.x[0])
        # H_freq_cmb.append(H_cmb_[0][0])
        #
        # likelihood_minimised_cmb_EE = minimize(
        #     cf.likelihood_for_hessian_a,    0,
        #     (spectra_cl, data_matrix_cmb_EE, b, nside, f_sky_SAT, 'EE'))
        # H_cmb_EE_ = nd.Hessian(cf.likelihood_for_hessian_a)(
        #     likelihood_minimised_cmb_EE.x,
        #     spectra_cl, data_matrix_cmb_EE, b,
        #     nside, f_sky_SAT, 'EE')
        # fit_alpha_freq_cmb_EE.append(likelihood_minimised_cmb_EE.x[0])
        # H_freq_cmb_EE.append(H_cmb_EE_[0][0])
        #
        # likelihood_minimised_cmb_BB = minimize(
        #     cf.likelihood_for_hessian_a,    0,
        #     (spectra_cl, data_matrix_cmb_BB, b, nside, f_sky_SAT, 'BB'))
        # H_cmb_BB_ = nd.Hessian(cf.likelihood_for_hessian_a)(
        #     likelihood_minimised_cmb_BB.x,
        #     spectra_cl, data_matrix_cmb_BB, b,
        #     nside, f_sky_SAT, 'BB')
        # fit_alpha_freq_cmb_BB.append(likelihood_minimised_cmb_BB.x[0])
        # H_freq_cmb_BB.append(H_cmb_BB_[0][0])
        # # IPython.embed()
        #
        # likelihood_minimised_cmb_EB = minimize(
        #     cf.likelihood_for_hessian_a,    0,
        #     (spectra_cl, data_matrix_cmb_EB, b, nside, f_sky_SAT, 'EB'))
        # H_cmb_EB_ = nd.Hessian(cf.likelihood_for_hessian_a)(
        #     likelihood_minimised_cmb_EB.x,
        #     spectra_cl, data_matrix_cmb_EB, b,
        #     nside, f_sky_SAT, 'EB')
        # fit_alpha_freq_cmb_EB.append(likelihood_minimised_cmb_EB.x[0])
        # H_freq_cmb_EB.append(H_cmb_EB_[0][0])
        # IPython.embed()

        # likelihood_minimised_cmb_dust = minimize(
        #     cf.likelihood_for_hessian_a, 0,
        #     (spectra_cl, data_matrix_cmb_dust, b, nside, f_sky_SAT))
        # H_cmb_dust_ = nd.Hessian(cf.likelihood_for_hessian_a)(
        #     likelihood_minimised_cmb_dust.x,
        #     spectra_cl, data_matrix_cmb_dust, b,
        #     nside, f_sky_SAT)
        # fit_alpha_freq_cmb_dust.append(likelihood_minimised_cmb_dust.x[0])
        # H_freq_cmb_dust.append(H_cmb_dust_[0][0])
        #
        # likelihood_minimised_cmb_sync = minimize(
        #     cf.likelihood_for_hessian_a, 0,
        #     (spectra_cl, data_matrix_cmb_sync, b, nside, f_sky_SAT))
        # H_cmb_sync_ = nd.Hessian(cf.likelihood_for_hessian_a)(
        #     likelihood_minimised_cmb_sync.x,
        #     spectra_cl, data_matrix_cmb_sync, b,
        #     nside, f_sky_SAT)
        # fit_alpha_freq_cmb_sync.append(likelihood_minimised_cmb_sync.x[0])
        # H_freq_cmb_sync.append(H_cmb_sync_[0][0])

        # likelihood_minimised_cmb_dust_sync = minimize(
        #     cf.likelihood_for_hessian_a, 0,
        #     (spectra_cl, data_matrix_cmb_dust_sync, b, nside, f_sky_SAT))
        # H_cmb_dust_sync_ = nd.Hessian(cf.likelihood_for_hessian_a)(
        #     likelihood_minimised_cmb_dust_sync.x,
        #     spectra_cl, data_matrix_cmb_dust_sync, b,
        #     nside, f_sky_SAT)
        # fit_alpha_freq_cmb_dust_sync.append(likelihood_minimised_cmb_dust_sync.x[0])
        # H_freq_cmb_dust_sync.append(H_cmb_dust_sync_[0][0])

        #     fit_alpha_cmb.append(fit_alpha_freq_cmb)
        #     H_cmb.append(H_freq_cmb)
        #
        #     fit_alpha_cmb_EE.append(fit_alpha_freq_cmb_EE)
        #     H_cmb_EE.append(H_freq_cmb_EE)
        #     fit_alpha_cmb_BB.append(fit_alpha_freq_cmb_BB)
        #     H_cmb_BB.append(H_freq_cmb_BB)
        #     fit_alpha_cmb_EB.append(fit_alpha_freq_cmb_EB)
        #     H_cmb_EB.append(H_freq_cmb_EB)
        #
        #     fit_alpha_cmb_dust.append(fit_alpha_freq_cmb_dust)
        #     H_cmb_dust.append(H_freq_cmb_dust)
        #     fit_alpha_cmb_sync.append(fit_alpha_freq_cmb_sync)
        #     H_cmb_sync.append(H_freq_cmb_sync)
        #     fit_alpha_cmb_dust_sync.append(fit_alpha_freq_cmb_dust_sync)
        #     H_cmb_dust_sync.append(H_freq_cmb_dust_sync)
        # fit_alpha_cmb = np.array(fit_alpha_cmb)
        # H_cmb = np.array(H_cmb)
        #
        # fit_alpha_cmb_EE = np.array(fit_alpha_cmb_EE)
        # H_cmb_EE = np.array(H_cmb_EE)
        # fit_alpha_cmb_BB = np.array(fit_alpha_cmb_BB)
        # H_cmb_BB = np.array(H_cmb_BB)
        # fit_alpha_cmb_EB = np.array(fit_alpha_cmb_EB)
        # H_cmb_EB = np.array(H_cmb_EB)
        #
        # fit_alpha_cmb_dust = np.array(fit_alpha_cmb_dust)
        # H_cmb_dust = np.array(H_cmb_dust)
        # fit_alpha_cmb_sync = np.array(fit_alpha_cmb_sync)
        # H_cmb_sync = np.array(H_cmb_sync)
        # fit_alpha_cmb_dust_sync = np.array(fit_alpha_cmb_dust_sync)
        # H_cmb_dust_sync = np.array(H_cmb_dust_sync)
        #
        # print('H_cmb_EE = ', H_cmb_EE)
        # print('H_cmb_BB = ', H_cmb_BB)
        # print('H_cmb_EB = ', H_cmb_EB)
        # print('H data model EE = ', H_model[1])

    # IPython.embed()

    fisher_sigma = 1/np.sqrt(fisher_modeldata)
    min_angle = rotation_angle.value-fisher_sigma*6
    max_angle = rotation_angle.value + fisher_sigma*6
    nstep_angle = 1000
    alpha_grid_fisherEEBBEB = np.arange(min_angle, max_angle,
                                        (max_angle - min_angle)/nstep_angle)
    error68_EEBBEB = grid_error(spectra_cl, data_model_matrix,
                                alpha_grid_fisherEEBBEB, b, nside, f_sky_SAT,
                                spectra_used='all', x_max_likelihood=rotation_angle.value,
                                output_grid=False)

    min_angle = rotation_angle.value-0.5
    max_angle = rotation_angle.value + 0.5
    nstep_angle = 500
    alpha_grid_fisherEE = np.arange(min_angle, max_angle,
                                    (max_angle - min_angle)/nstep_angle)

    error68_EE = grid_error(spectra_cl, data_model_matrix_EE,
                            alpha_grid_fisherEE, b, nside, f_sky_SAT,
                            spectra_used='EE', x_max_likelihood=rotation_angle.value,
                            output_grid=False)

    min_angle = rotation_angle.value - 0.05
    max_angle = rotation_angle.value + 0.05
    nstep_angle = 500
    alpha_grid_fisherBB = np.arange(min_angle, max_angle,
                                    (max_angle - min_angle)/nstep_angle)
    error68_BB = grid_error(spectra_cl, data_model_matrix_BB,
                            alpha_grid_fisherBB, b, nside, f_sky_SAT,
                            spectra_used='BB', x_max_likelihood=rotation_angle.value,
                            output_grid=False)

    min_angle = rotation_angle.value-(4e-3)
    max_angle = rotation_angle.value + 4e-3
    nstep_angle = 500
    alpha_grid_fisherEB = np.arange(min_angle, max_angle,
                                    (max_angle - min_angle)/nstep_angle)
    error68_EB = grid_error(spectra_cl, data_model_matrix_EB,
                            alpha_grid_fisherEB, b, nside, f_sky_SAT,
                            spectra_used='EB', x_max_likelihood=rotation_angle.value,
                            output_grid=False)

    plot_minimisation_results = 0
    if plot_minimisation_results:
        plt.errorbar(-1, fit_alpha_model[0],
                     yerr=1/np.sqrt(H_model[0]), fmt='o', label='data = model')

        plt.errorbar(range(nsim), fit_alpha_cmb[:, 0],
                     yerr=1/np.sqrt(H_cmb[:, 0]), fmt='o', label='CMB + noise')
        # plt.errorbar(0, fit_alpha_cmb[0, 0],
        #              yerr=1/np.sqrt(H_cmb[0, 0]), fmt='o', label='CMB')

        # plt.errorbar(1, fit_alpha_cmb[1, 0],
        #              yerr=1/np.sqrt(H_cmb[1, 0]), fmt='o', label='CMB + noise map')
        # plt.errorbar(3, fit_alpha_freq_noise,
        #              yerr=1/np.sqrt(H_freq_noise), fmt='o', label='Noise')

        # plt.errorbar(2, fit_alpha_model[1],
        #              yerr=1/np.sqrt(H_model[1]), fmt='o', label='data = model EE')
        # plt.errorbar(range((1+nsim), (1+nsim)+nsim), fit_alpha_cmb_EE[:, 0],
        #              yerr=1/np.sqrt(H_cmb_EE[:, 0]), fmt='o', label='CMB EE only')
        # plt.errorbar(range(2*nsim + 1, 2*nsim+nsim+1), fit_alpha_cmb_BB[:, 0],
        #              yerr=1/np.sqrt(H_cmb_BB[:, 0]), fmt='o', label='CMB BB only')
        # plt.errorbar(range(3*nsim+1, 3*nsim+nsim+1), fit_alpha_cmb_EB[:, 0],
        #              yerr=1/np.sqrt(H_cmb_EB[:, 0]), fmt='o', label='CMB EB only')
        plt.hlines(0, -1, 8,  label='input miscalibration angle')

        # plt.errorbar(range(nsim), fit_alpha_cmb[:, 0],
        #              yerr=1/np.sqrt(H_cmb[:, 0]), fmt='o', label='CMB')
        # plt.errorbar(range(nsim, nsim+nsim), fit_alpha_cmb_dust[:, 0],
        #              yerr=1/np.sqrt(H_cmb_dust[:, 0]), fmt='o', label='CMB + Dust')
        # plt.errorbar(range(2*nsim, 2*nsim+nsim), fit_alpha_cmb_sync[:, 0],
        #              yerr=1/np.sqrt(H_cmb_sync[:, 0]), fmt='o', label='CMB + Synchrotron')
        # plt.errorbar(range(3*nsim, 3*nsim+nsim), fit_alpha_cmb_dust_sync[:, 0],
        #              yerr=1/np.sqrt(H_cmb_dust_sync[:, 0]), fmt='o', label='CMB + Dust + Synchrotron')
        # plt.errorbar(3*nsim+nsim, fit_alpha_freq_noise,
        #              yerr=1/np.sqrt(H_freq_noise), fmt='o', label='Noise')
        # plt.hlines(0, 0, 3*nsim+nsim,  label='input miscalibration angle')
        # plt.hlines(np.mean(fit_alpha_cmb[:, 0]), 0, 3*nsim+nsim,
        #            linestyles='--', label='average CMB fit', color='blue')
        # plt.hlines(np.mean(fit_alpha_cmb_dust[:, 0]), 0, 3*nsim+nsim,
        #            linestyles='--', label='average CMB + Dust fit', color='orange')
        # plt.hlines(np.mean(fit_alpha_cmb_sync[:, 0]), 0, 3*nsim+nsim,
        #            linestyles='--', label='average CMB + synchrotron fit', color='green')
        # plt.hlines(np.mean(fit_alpha_cmb_dust_sync[:, 0]), 0, 3*nsim+nsim,
        #            linestyles='--', label='average CMB + Dust + synchrotron fit', color='red')
        plt.legend()
        plt.grid(linestyle=':')
        plt.xlabel('simulations')
        plt.ylabel('miscalibration angle')
        plt.title('miscalibration angle fit at {} GHz'.format(frequencies2use))
        # plt.show()
        plt.savefig('fit_{}_sim_CMB+noise_test.png'.format(nsim))
        # exit()
        # IPython.embed()
        fit_spectra = cf.power_spectra_operation(r=r, rotation_angle=fit_alpha_cmb[-1][0] * u.rad,
                                                 powers_name='total')
        fit_spectra.get_spectra()
        fit_spectra.spectra_rotation()
        fit_spectra_binned = b.bin_cell(fit_spectra.cl_rot.spectra[:3*nside].T).T

        fit_spectra.spectra_rotation((fit_alpha_cmb[-1][0] - 1/np.sqrt(H_cmb)[-1][0])*u.rad)
        fit_spectra_binned_minus = b.bin_cell(fit_spectra.cl_rot.spectra[:3*nside].T).T

        fit_spectra.spectra_rotation((fit_alpha_cmb[-1][0] + 1/np.sqrt(H_cmb)[-1][0])*u.rad)
        fit_spectra_binned_plus = b.bin_cell(fit_spectra.cl_rot.spectra[:3*nside].T).T
        # IPython.embed()

        fig, ax = plt.subplots(3, 2, figsize=(19.20, 10.80))
        for k in range(3):
            if k == 0:
                i = 0
                j = 0
                spectrum_index = 0
                camb_index = 1
            elif k == 1:
                i = 0
                j = 1
                spectrum_index = 3
                camb_index = 2

            elif k == 2:
                i = 1
                j = 0
                spectrum_index = 1
                camb_index = 4

            spectra_list = ['EE', 'BB', 'EB']

            ax[i, j].plot(b.get_effective_ells(), Cl_cmb[0][spectrum_index], label='CMB+noise')
            # plt.plot(b.get_effective_ells(), Cl_cmb[1][1] + Cl_noise[0][1], label='CMB + Cl noise')
            # ax[i, j].plot(b.get_effective_ells(), Cl_cmb[1][spectrum_index], label='CMB + noise map')
            ax[i, j].plot(b.get_effective_ells(), Cl_noise[0][spectrum_index], label='Noise')
            # plt.plot(b.get_effective_ells(), noise_nl, label='V3 calc nl')
            # plt.plot(b.get_effective_ells(), noise_nl.T[1], label='V3 calc nl')
            ax[i, j].plot(b.get_effective_ells(), Cl_dust[0][spectrum_index], label='Dust')
            ax[i, j].plot(b.get_effective_ells(), Cl_sync[0][spectrum_index], label='Synchrotron')
            ax[i, j].plot(b.get_effective_ells(),
                          fit_spectra_binned.T[camb_index],
                          label='CAMB binned * best fit angle rotation', color='red')
            ax[i, j].fill_between(b.get_effective_ells(), fit_spectra_binned_minus.T[camb_index],
                                  fit_spectra_binned_plus.T[camb_index], color='red', alpha=0.2)
            ax[i, j].plot(b.get_effective_ells(), data_model.T[camb_index],
                          label='Model in likelihood (CMB + noise)')
            ax[i, j].plot(b.get_effective_ells(), save_cmb_model.T[camb_index],
                          label='Model in likelihood (CMB)')
            ax[i, j].legend()
            if k != 2:
                ax[i, j].set_yscale('log')
            ax[i, j].set_xscale('log')
            ylabel = r'$C_{\ell}^{'+spectra_list[k] + '}$'
            # ax[i, j].set_ylabel(r'$C_{\ell}^{}$'.format(spectra_list[k]))
            ax[i, j].set_ylabel(ylabel)

            ax[i, j].set_xlabel(r'$\ell$')

        ax[1, 1].errorbar(-1, fit_alpha_model[0],
                          yerr=1/np.sqrt(H_model[0]), fmt='o', label='data = model')

        # ax[1, 1].errorbar(0, fit_alpha_cmb[0, 0],
        #                   yerr=1/np.sqrt(H_cmb[0, 0]), fmt='o', label='CMB')
        # ax[1, 1].errorbar(1, fit_alpha_cmb[1, 0],
        #                   yerr=1/np.sqrt(H_cmb[1, 0]), fmt='o', label='CMB + noise map')

        ax[1, 1].errorbar(0, fit_alpha_cmb[0, 0],
                          yerr=1/np.sqrt(H_cmb[0, 0]), fmt='o', label='CMB + noise map')
        ax[1, 1].errorbar(1, fit_alpha_model[1],
                          yerr=1/np.sqrt(H_model[1]), fmt='o', label='data = model EE')
        ax[1, 1].errorbar(range(nsim+1, 1+nsim+nsim), fit_alpha_cmb_EE[:, 0],
                          yerr=1/np.sqrt(H_cmb_EE[:, 0]), fmt='o', label='CMB EE only')
        ax[1, 1].errorbar(range(2*nsim+1, 2*nsim+nsim+1), fit_alpha_cmb_BB[:, 0],
                          yerr=1/np.sqrt(H_cmb_BB[:, 0]), fmt='o', label='CMB BB only')
        ax[1, 1].errorbar(range(3*nsim+1, 3*nsim+nsim+1), fit_alpha_cmb_EB[:, 0],
                          yerr=1/np.sqrt(H_cmb_EB[:, 0]), fmt='o', label='CMB EB only')
        ax[1, 1].hlines(0, -1, 8,  label='input miscalibration angle')

        ax[1, 1].legend()
        ax[1, 1].grid(linestyle=':')
        ax[1, 1].set_xlabel('simulations')
        ax[1, 1].set_ylabel('miscalibration angle')
        ax[1, 1].set_title('miscalibration angle fit at {} GHz'.format(frequencies2use))
        ax[1, 1].set_yscale('symlog')
        column = (r'$\alpha$ fit', r'$\sigma (\alpha)$', r'$\sigma 68$')
        # row = ('data = model', 'CMB', 'CMB + noise map', 'data = model EE',
        #        'CMB EE only', 'CMB + noise EE only', 'CMB BB only', 'CMB + noise BB only',
        #        'CMB EB only', 'CMB + noise EB only')
        row = ('data = model', 'data = model EE', 'data = model BB',
               'data = model EB', 'CMB + noise map',
               'CMB + noise EE only', 'CMB + noise BB only',
               'CMB + noise EB only')
        data_table = [[fit_alpha_model[0], fit_alpha_model[1],
                       fit_alpha_model[2], fit_alpha_model[3],
                       fit_alpha_cmb[0][0], fit_alpha_cmb_EE[0][0],
                       fit_alpha_cmb_BB[0][0],
                       fit_alpha_cmb_EB[0][0]],
                      [1/np.sqrt(H_model[0]), 1/np.sqrt(H_model[1]),
                       1/np.sqrt(H_model[2]), 1/np.sqrt(H_model[3]),
                       1/np.sqrt(H_cmb[0, 0]),
                       1/np.sqrt(H_cmb_EE[0, 0]),
                       1/np.sqrt(H_cmb_BB[0, 0]),
                       1/np.sqrt(H_cmb_EB[0, 0])],
                      [error68_EEBBEB, error68_EE, error68_BB, error68_EB,
                       'not computed', 'not computed', 'not computed', 'not computed']]
        data_strg = []
        for irow in range(len(row)):
            data_strg_row = []
            for jcolumn in range(len(column)):
                if type(data_table[jcolumn][irow]) == str:
                    data_strg_row.append(data_table[jcolumn][irow])
                else:
                    data_strg_row.append('{:.2e}'.format(data_table[jcolumn][irow]))
            data_strg.append(data_strg_row)
        ytable = ax[2, 1].table(cellText=data_strg, rowLabels=row, colLabels=column,
                                loc='center right', colWidths=[.40]*len(column))
        cellDict = ytable.get_celld()
        for key in cellDict:
            cellDict[key].set_height(.08)
        ax[2, 1].axis('tight')
        ax[2, 1].axis('off')

        IPython.embed()
        # plt.subplots_adjust(left=0.2, top=0.8)
        plt.savefig('test_abs_a3deg.png', dpi=100)
        # plt.show()
        plt.close()

    '''####################################################################'''
    '''________________EEBBEB______________'''
    grid_EEBBEB = 0
    if grid_EEBBEB:
        fisher_sigma = 1/np.sqrt(fisher_modeldata)
        min_angle = rotation_angle.value-fisher_sigma*6
        max_angle = rotation_angle.value + fisher_sigma*6
        nstep_angle = 1000
        # min_angle = -fisher_sigma*6  # rotation.value - 5*(1/np.sqrt(fishaa))
        # max_angle = fisher_sigma*6  # rotation.value + 5*(1/np.sqrt(fishaa))
        # nstep_angle = 1000
        alpha_grid_fisherEEBBEB = np.arange(min_angle, max_angle,
                                            (max_angle - min_angle)/nstep_angle)
        L_EEBBEB = L_gridding(spectra_cl, data_model_matrix, alpha_grid_fisherEEBBEB,
                              b, nside, f_sky_SAT)[0]
        sigmafisherEEBBEB = error_68(alpha_grid_fisherEEBBEB, L_EEBBEB, rotation_angle.value)

        plt.close()
        plt.plot(alpha_grid_fisherEEBBEB, L_EEBBEB, label='gridding')
        in_exp_sigma68 = -np.array((alpha_grid_fisherEEBBEB-rotation_angle.value)**2 /
                                   (2 * (sigmafisherEEBBEB**2)))  # - max(-np.array(alpha_grid_fisherEEBBEB**2 * fisher_sigma/2))
        in_exp_fishersigma = -np.array((alpha_grid_fisherEEBBEB-rotation_angle.value)**2 /
                                       (2 * (fisher_sigma**2)))
        plt.plot(alpha_grid_fisherEEBBEB, np.exp(in_exp_sigma68), ':',
                 label='predicted likelihood from 68% estimation sigma ={}'.format(sigmafisherEEBBEB))
        plt.plot(alpha_grid_fisherEEBBEB, np.exp(in_exp_fishersigma), '-.',
                 label='predicted likelihood from fisher sigma ={}'.format(fisher_sigma))

        # plt.yscale('symlog')
        plt.title('Likelihood EE,BB,EB, model = data')
        plt.xlabel('angle in rad')
        plt.ylabel('L')
        plt.legend()
        plt.savefig('likelihood_EEBBEBdatamodel_nonoise_{}steps.png'.format(nstep_angle))
        plt.close()

    '''________________EE______________'''
    grid_EE = 0
    if grid_EE:
        min_angle = rotation_angle.value-0.5
        max_angle = rotation_angle.value + 0.5
        nstep_angle = 100
        alpha_grid_fisherEE = np.arange(min_angle, max_angle,
                                        (max_angle - min_angle)/nstep_angle)
        L_EE = L_gridding(spectra_cl, data_model_matrix_EE, alpha_grid_fisherEE,
                          b, nside, f_sky_SAT, 'EE')[0]

        sigma68EE = error_68(alpha_grid_fisherEE, L_EE, rotation_angle.value)

        plt.close()
        plt.plot(alpha_grid_fisherEE, L_EE)
        in_exp_sigma68_EE = -np.array((alpha_grid_fisherEE)**2 /
                                      (2 * (sigma68EE**2)))
        plt.plot(alpha_grid_fisherEE, np.exp(in_exp_sigma68_EE), ':',
                 label='predicted likelihood from 68% estimation sigma ={}'.format(sigma68EE))
        plt.title('Likelihood EE, model = data')
        plt.xlabel('angle in rad')
        plt.ylabel('L')
        plt.legend()
        plt.savefig('likelihood_EEdatamodel_nonoise_{}steps.png'.format(nstep_angle))
        plt.close()
        # sigmafisherEE = error_68(alpha_grid_fisherEE, L_EE)

        plt.close()
    '''________________BB______________'''
    grid_BB = 0
    if grid_BB:
        min_angle = rotation_angle.value - 0.05
        max_angle = rotation_angle.value + 0.05
        nstep_angle = 100
        alpha_grid_fisherBB = np.arange(min_angle, max_angle,
                                        (max_angle - min_angle)/nstep_angle)
        L_BB = L_gridding(spectra_cl, data_model_matrix_BB, alpha_grid_fisherBB,
                          b, nside, f_sky_SAT, 'BB')[0]
        sigma68BB = error_68(alpha_grid_fisherBB, L_BB, rotation_angle.value)

        plt.plot(alpha_grid_fisherBB, L_BB)
        in_exp_sigma68_BB = -np.array((alpha_grid_fisherBB-rotation_angle.value)**2 /
                                      (2 * (sigma68BB**2)))
        plt.plot(alpha_grid_fisherBB, np.exp(in_exp_sigma68_BB), ':',
                 label='predicted likelihood from 68% estimation sigma ={}'.format(sigma68BB))

        fisher_sigma_BB = 1/np.sqrt(fisher_modeldata_BB)
        in_exp_fishersigma = -np.array((alpha_grid_fisherBB-rotation_angle.value)**2 /
                                       (2 * (fisher_sigma_BB**2)))
        plt.plot(alpha_grid_fisherBB, np.exp(in_exp_fishersigma), '-.',
                 label='predicted likelihood from fisher sigma ={}'.format(fisher_sigma_BB))
        plt.title('Likelihood BB, model = data')
        plt.xlabel('angle in rad')
        plt.ylabel('L')
        plt.legend()
        plt.savefig('likelihood_BBdatamodel_nonoise_{}steps.png'.format(nstep_angle))
        plt.close()

    '''________________EB______________'''
    grid_EB = 1
    if grid_EB:
        min_angle = rotation_angle.value-(4e-5)
        max_angle = rotation_angle.value + 4e-5
        nstep_angle = 100
        alpha_grid_fisherEB = np.arange(min_angle, max_angle,
                                        (max_angle - min_angle)/nstep_angle)

        IPython.embed()
        start_grid = time.time()
        L_EB, model_grid, data_grid, Cm1D_grid, likelihood_element_grid, max2_list = L_gridding(spectra_cl, data_model_matrix_EB, alpha_grid_fisherEB,
                                                                                                b, nside, f_sky_SAT, 'EB')
        print('time grid = ', time.time()-start_grid)
        sigma68EB = error_68(alpha_grid_fisherEB, L_EB, rotation_angle.value)

        plt.plot(alpha_grid_fisherEB, L_EB)
        in_exp_sigma68_EB = -np.array((alpha_grid_fisherEB-rotation_angle.value)**2 /
                                      (2 * (sigma68EB**2)))
        plt.plot(alpha_grid_fisherEB, np.exp(in_exp_sigma68_EB), ':',
                 label='predicted likelihood from 68% estimation sigma = {}'.format(sigma68EB))

        in_exp_fisherEB = -np.array(alpha_grid_fisherEB**2 /
                                    (2 * (fisher_sigma_EB**2)))
        plt.plot(alpha_grid_fisherEB, np.exp(in_exp_fisherEB), '-.',
                 label='predicted likelihood from fisher sigma = {}'.format(fisher_sigma_EB))
        plt.title('Likelihood EB, model = data')
        plt.xlabel('angle in rad')
        plt.ylabel('L')
        plt.legend()
        plt.savefig('likelihood_EBdatamodel_noise_goodspectrum_{}steps_a={}.png'.format(
            nstep_angle, rotation_angle.value))
        plt.close()

    IPython.embed()

    '''#####################################################################'''

    print('tset')
    index = 1
    plt.plot(data_model_nobin[index] - save_cmb_model_nobin[index], label='no bin')
    plt.plot(b.get_effective_ells(), data_model.T[index] - save_cmb_model.T[index], label='binned')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.show()

    # Cl0_mean_dust = np.mean(Cl0_dust, axis=0)
    Cl_mean_dust = np.mean(Cl_dust, axis=0)
    true_cl_bined = b.bin_cell(spectra_cl.cl_rot.spectra[: 3*nside].T)
    prim_cl_bined = b.bin_cell(spectra_cl.spectra.spectra[: 3*nside].T)

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

'''================================purgatory================================'''
# gridding1 = 0
# if gridding1:
#     start = time.time()
#     min_angle = -np.pi/2  # rotation.value - 5*(1/np.sqrt(fishaa))
#     max_angle = np.pi/2  # rotation.value + 5*(1/np.sqrt(fishaa))
#     nstep_angle = 5000
#     alpha_grid = np.arange(min_angle, max_angle,
#                            (max_angle - min_angle)/nstep_angle)
#     grid_EE_modeldata4 = []
#     grid_EB = []
#     grid_EEBBEBmodeldata = []
#     for alpha in alpha_grid:
#         # grid_EE_modeldata.append(
#         #     cf.likelihood_for_hessian_a(alpha, spectra_cl, data_model_matrix_EE,
#         #                                 b, nside, f_sky_SAT, 'EE'))
#         # grid_EB.append(
#         #     cf.likelihood_for_hessian_a(alpha, spectra_cl, data_matrix_cmb_EB,
#         #                                 b, nside, f_sky_SAT, 'EB'))
#         grid_EEBBEBmodeldata.append(
#             cf.likelihood_for_hessian_a(alpha, spectra_cl, data_model_matrix,
#                                         b, nside, f_sky_SAT))
#     print('time = ', time.time() - start)
#     min_angle = -np.pi/4  # rotation.value - 5*(1/np.sqrt(fishaa))
#     max_angle = np.pi/4  # rotation.value + 5*(1/np.sqrt(fishaa))
#     nstep_angle = 5000/4
#     alpha_grid4 = np.arange(min_angle, max_angle,
#                             (max_angle - min_angle)/nstep_angle)
#
#     grid_EEBBEBmodeldata4 = []
#     for alpha in alpha_grid4:
#         grid_EEBBEBmodeldata4.append(
#             cf.likelihood_for_hessian_a(alpha, spectra_cl, data_model_matrix,
#                                         b, nside, f_sky_SAT))
#         grid_EE_modeldata4.append(
#             cf.likelihood_for_hessian_a(alpha, spectra_cl, data_model_matrix_EE,
#                                         b, nside, f_sky_SAT, 'EE'))
#     min_angle = -np.pi/100  # rotation.value - 5*(1/np.sqrt(fishaa))
#     max_angle = np.pi/100  # rotation.value + 5*(1/np.sqrt(fishaa))
#     nstep_angle = 5000/100
#     alpha_grid100 = np.arange(min_angle, max_angle,
#                               (max_angle - min_angle)/nstep_angle)
#
#     grid_EEBBEBmodeldata100 = []
#     for alpha in alpha_grid100:
#         grid_EEBBEBmodeldata100.append(
#             cf.likelihood_for_hessian_a(alpha, spectra_cl, data_model_matrix,
#                                         b, nside, f_sky_SAT))
#     # min_angle = -np.pi/4  # rotation.value - 5*(1/np.sqrt(fishaa))
#     # max_angle = np.pi/4  # rotation.value + 5*(1/np.sqrt(fishaa))
#     # nstep_angle = 1000/4
#     # alpha_grid_250 = np.arange(min_angle, max_angle,
#     #                            (max_angle - min_angle)/nstep_angle)
#     # grid_EEBBEBmodeldata250 = []
#     # for alpha in alpha_grid_250:
#     #     grid_EEBBEBmodeldata250.append(
#     #         cf.likelihood_for_hessian_a(alpha, spectra_cl, data_model_matrix,
#     #                                     b, nside, f_sky_SAT))
# plt.close()
# plot_likelihood = 0
# if plot_likelihood:
#     plt.plot(alpha_grid, -np.array(grid_EE_modeldata))
#     plt.title('2log(L) EE with model == data')
#     plt.xlabel('angle in rad')
#     plt.ylabel('2log(L)')
#     plt.savefig('grid_EEmodeldata.png')
#     plt.close()
#
#     plt.plot(alpha_grid, np.array(grid_EE_modeldata))
#     plt.title('-2log(L) EE with model == data')
#     plt.xlabel('angle in rad')
#     plt.ylabel('-2log(L)')
#     plt.savefig('minusgrid_EEmodeldata.png')
#     plt.close()
#
#     plt.plot(alpha_grid, np.array(grid_EB))
#     plt.yscale('symlog')
#     plt.title('-2log(L) EB, data = CMB + noise')
#     plt.xlabel('angle in rad')
#     plt.ylabel('-2log(L)')
#     plt.savefig('minusgrid_EB.png')
#     plt.close()
#
#     plt.plot(alpha_grid, -np.array(grid_EB))
#     plt.yscale('symlog')
#     plt.title('2log(L) EB, data = CMB + noise')
#     plt.xlabel('angle in rad')
#     plt.ylabel('2log(L)')
#     plt.savefig('grid_EB.png')
#     plt.close()
#
#     plt.plot(alpha_grid, np.array(grid_EEBBEBmodeldata))
#     plt.yscale('symlog')
#     plt.title('-2log(L) EE,BB,EB, model = data')
#     plt.xlabel('angle in rad')
#     plt.ylabel('-2log(L)')
#     plt.savefig('minsugrid_EEBBEBdatamodel.png')
#     plt.close()

# IPython.embed()
# gridding2 = 0
# if gridding2:
#     positive_index = np.where(alpha_grid >= 0)
#     negative_index = np.where(alpha_grid < 0)
#
#     positive_index4 = np.where(alpha_grid4 >= 0)
#     negative_index4 = np.where(alpha_grid4 < 0)
#
#     positive_index100 = np.where(alpha_grid100 >= 0)
#     negative_index100 = np.where(alpha_grid100 < 0)
#
#     in_exp = -np.array(grid_EEBBEBmodeldata) - max(-np.array(grid_EEBBEBmodeldata))
#     total_integral = np.trapz(np.exp(in_exp), x=alpha_grid)
#
#     in_exp4 = -np.array(grid_EEBBEBmodeldata4) - max(-np.array(grid_EEBBEBmodeldata4))
#     total_integral4 = np.trapz(np.exp(in_exp4), x=alpha_grid4)
#
#     in_exp100 = -np.array(grid_EEBBEBmodeldata100) - max(-np.array(grid_EEBBEBmodeldata100))
#     total_integral100 = np.trapz(np.exp(in_exp100), x=alpha_grid100)
#
#     # partial_integral = 0
#     # i = 0
#     # y_array = np.array([])
#     # x_array = np.array([])
#     # while partial_integral <= 0.68*total_integral:
#     #     y_array = np.append(np.exp(in_exp[negative_index[0][-(i+1)]]), y_array)
#     #     y_array = np.append(y_array, np.exp(in_exp[positive_index[0][i]]))
#     #     x_array = np.append(alpha_grid[negative_index[0][-(i+1)]], x_array)
#     #     x_array = np.append(x_array, alpha_grid[positive_index[0][i]])
#     #     partial_integral = np.trapz(y_array, x_array)
#     #     # partial_integral += np.exp(in_exp[positive_index[0][i]]) +
#     #     #     np.exp(in_exp[negative_index[0][-i]])
#     #     i += 1
#     # sigma = alpha_grid[positive_index[0][i]]
#
#     partial_integral = 0
#     i = 0
#     y_array = np.array([])
#     x_array = np.array([])
#     while partial_integral <= 0.682*total_integral4:
#         y_array = np.append(np.exp(in_exp4[negative_index4[0][-(i+1)]]), y_array)
#         y_array = np.append(y_array, np.exp(in_exp4[positive_index4[0][i]]))
#         x_array = np.append(alpha_grid4[negative_index4[0][-(i+1)]], x_array)
#         x_array = np.append(x_array, alpha_grid4[positive_index4[0][i]])
#         partial_integral = np.trapz(y_array, x_array)
#         # partial_integral += np.exp(in_exp[positive_index[0][i]]) +
#         #     np.exp(in_exp[negative_index[0][-i]])
#         i += 1
#     sigma4 = alpha_grid4[positive_index4[0][i]]
#
#     partial_integral = 0
#     i = 0
#     y_array = np.array([])
#     x_array = np.array([])
#     while partial_integral <= 0.682*total_integral100:
#         y_array = np.append(np.exp(in_exp100[negative_index100[0][-(i+1)]]), y_array)
#         y_array = np.append(y_array, np.exp(in_exp100[positive_index100[0][i]]))
#         x_array = np.append(alpha_grid100[negative_index100[0][-(i+1)]], x_array)
#         x_array = np.append(x_array, alpha_grid100[positive_index100[0][i]])
#         partial_integral = np.trapz(y_array, x_array)
#         # partial_integral += np.exp(in_exp[positive_index[0][i]]) +
#         #     np.exp(in_exp[negative_index[0][-i]])
#         i += 1
#     sigma100 = alpha_grid100[positive_index100[0][i]]
#     print('sigma4 = ', sigma4)
#     print('fisher model = ', fisher_modeldata)
#     fisher_sigma = 1/np.sqrt(fisher_modeldata)
#     print('fisher sigma = ', fisher_sigma)
#     print('fisher sigma - sigma4 = ', fisher_sigma - sigma4)
#     print('(fisher sigma - sigma)/fisher sigma = ',
#           (fisher_sigma - sigma4)/fisher_sigma)

# grid_EEmodeldata_fisher = []
# start = time.time()
# for alpha in alpha_grid_fisherEE:
#     grid_EEmodeldata_fisher.append(
#         cf.likelihood_for_hessian_a(alpha, spectra_cl, data_model_matrix_EE,
#                                     b, nside, f_sky_SAT, 'EE'))
# print('time grid {} steps = '.format(nstep_angle), time.time() - start,
#       'seconds')
# in_expEE = -np.array(grid_EEmodeldata_fisher) - max(-np.array(grid_EEmodeldata_fisher))

# total_integral_fisherEE = np.trapz(np.exp(in_expEE),
#                                    x=alpha_grid_fisherEE)
# positive_index_fisher_EE = np.where(alpha_grid_fisherEE >= 0)
# negative_index_fisher_EE = np.where(alpha_grid_fisherEE < 0)
# partial_integral = 0
# i = 0
# y_array = np.array([])
# x_array = np.array([])
# while partial_integral <= 0.68*total_integral_fisherEE:
#     y_array = np.append(np.exp(in_expEE[negative_index_fisher_EE[0][-(i+1)]]), y_array)
#     y_array = np.append(y_array, np.exp(in_expEE[positive_index_fisher_EE[0][i]]))
#     x_array = np.append(
#         alpha_grid_fisherEE[negative_index_fisher_EE[0][-(i+1)]], x_array)
#     x_array = np.append(x_array, alpha_grid_fisherEE[positive_index_fisher_EE[0][i]])
#     partial_integral = np.trapz(y_array, x_array)
#     # partial_integral += np.exp(in_exp[positive_index[0][i]]) +
#     #     np.exp(in_exp[negative_index[0][-i]])
#     i += 1
# sigmafisherEE = alpha_grid_fisherEE[positive_index_fisher_EE[0][i]]

# grid_EEBBEBmodeldata_fisher = []
# start_5k = time.time()
# for alpha in alpha_grid_fisherEEBBEB:
#     grid_EEBBEBmodeldata_fisher.append(
#         cf.likelihood_for_hessian_a(alpha, spectra_cl, data_model_matrix,
#                                     b, nside, f_sky_SAT))
# print('time 5k steps = ', time.time() - start_5k)
# in_expEEBBEB = -np.array(grid_EEBBEBmodeldata_fisher) - max(
#     -np.array(grid_EEBBEBmodeldata_fisher))

# total_integral_fisherEBE = np.trapz(np.exp(in_expEEBBEB),
#                                     x=alpha_grid_fisherEEBBEB)
# positive_index_fisher_EBE = np.where(alpha_grid_fisherEEBBEB >= 0)
# negative_index_fisher_EBE = np.where(alpha_grid_fisherEEBBEB < 0)
# partial_integral = 0
# i = 0
# y_array = np.array([])
# x_array = np.array([])
# while partial_integral <= 0.682689*total_integral_fisherEBE:
#     y_array = np.append(np.exp(in_expEEBBEB[negative_index_fisher_EBE[0][-(i+1)]]), y_array)
#     y_array = np.append(y_array, np.exp(in_expEEBBEB[positive_index_fisher_EBE[0][i]]))
#     x_array = np.append(alpha_grid_fisherEEBBEB[negative_index_fisher_EBE[0][-(i+1)]], x_array)
#     x_array = np.append(x_array, alpha_grid_fisherEEBBEB[positive_index_fisher_EBE[0][i]])
#     partial_integral = np.trapz(y_array, x_array)
#     # partial_integral += np.exp(in_exp[positive_index[0][i]]) +
#     #     np.exp(in_exp[negative_index[0][-i]])
#     i += 1
# sigmafisherEEBBEB = alpha_grid_fisherEEBBEB[positive_index_fisher_EBE[0][i]]

# wsp.wsp.ncls = 4
# nhits, noise_maps_, nlev = mknm.get_noise_sim(sensitivity=sensitivity,
#                                               knee_mode=knee_mode, ny_lf=ny_lf,
#                                               nside_out=nside,
#                                               norm_hits_map=hp.read_map(
#                                                   norm_hits_map_path),
#                                               no_inh=no_inh)

# noise_maps = []
# # noise_maps_masked = []
#
# for i in frequencies_index:
#     noise_maps.append([noise_maps_[i*3],
#                        noise_maps_[i*3+1],
#                        noise_maps_[i*3+2]])
# noise_maps_masked.append([noise_maps_[i*3] * mask,
#                           noise_maps_[i*3+1] * mask,
#                           noise_maps_[i*3+2] * mask])
# noise_maps = np.array(noise_maps)
# noise_maps_masked = np.array(noise_maps_masked)

# del noise_maps_
# noise_maps.append(noise_maps_[i*3+2] * mask)

# Cl_cmb = []
# Cl_cmb_dust = []
# Cl_cmb_sync = []
# Cl_cmb_dust_sync = []
# IPython.embed()
# start_10sim = time.time()
# for i in range(nsim):
# start_1sim = time.time()
# print('i', i)
# np.random.seed(int(time.time()))
# cmb_map = hp.synfast(spectra_cl.cl_rot.spectra.T, nside, new=True)

# print(cmb_map)
# f0_cmb = nmt.NmtField(mask_apo, [mask*cmb_map[0, :]])
# f2_cmb = get_field(mask*cmb_map_nsim[i, 0, 1, :],
#                    mask*cmb_map_nsim[i, 0, 2, :], mask_apo,
#                    purify_e=purify_e, purify_b=purify_b)
# Cl_cmb.append(compute_master(f2_cmb, f2_cmb, wsp))
# IPython.embed()
# Cl_cmb_dust_freq = []
# Cl_cmb_sync_freq = []
# Cl_cmb_dust_sync_freq = []

# for f in range(len(frequencies2use)):  # sky_map.frequencies:

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

# cmb_dust_map = (cmb_map_nsim[i]) + dust_map_freq[f]
# f2_cmb_dust = get_field(mask*cmb_dust_map[1, :],
#                         mask*cmb_dust_map[2, :], mask_apo,
#                         purify_e=purify_e, purify_b=purify_b)

# cmb_sync_map = (cmb_map_nsim[i]) + sync_map_freq[f]
# f2_cmb_sync = get_field(mask*cmb_sync_map[1, :],
#                         mask*cmb_sync_map[2, :], mask_apo,
#                         purify_e=purify_e, purify_b=purify_b)

# cmb_dust_sync_map = (cmb_map_nsim[i]) + dust_map_freq[f] + sync_map_freq[f]
# f2_cmb_dust_sync = get_field(mask*cmb_dust_sync_map[1, :],
#                              mask*cmb_dust_sync_map[2, :], mask_apo,
#                              purify_e=purify_e, purify_b=purify_b)

# IPython.embed()
# Cl_cmb_dust_freq.append(compute_master(f2_cmb_dust, f2_cmb_dust, wsp))
# Cl_cmb_sync_freq.append(compute_master(f2_cmb_sync, f2_cmb_sync, wsp))
# Cl_cmb_dust_sync_freq.append(compute_master(
#     f2_cmb_dust_sync, f2_cmb_dust_sync, wsp))
# Cl_dust_freq.append(compute_master(f2_dust, f2_dust, wsp))
# Cl_sync_freq.append(compute_master(f2_sync, f2_sync, wsp))
# Cl_cmb_dust.append(Cl_cmb_dust_freq)
# Cl_cmb_sync.append(Cl_cmb_sync_freq)
# Cl_cmb_dust_sync.append(Cl_cmb_dust_sync_freq)
# Cl_noise.append(Cl_noise)

# Cl_dust.append(Cl_dust_freq)
# Cl_sync.append(Cl_sync_freq)

# Cl0_dust.append(nmt.compute_full_master(f0_dust, f0_dust, b))
# print('time 1 sim = ', time.time() - start_1sim)
# time10 = time.time() - start_10sim
# print('time 10 sim = ', time10)
# print('average time over {} sim = '.format(nsim), time10/nsim)

# Cl_cmb = np.array(Cl_cmb)
# Cl_dust = np.array(Cl_dust)
# Cl_sync = np.array(Cl_sync)
# Cl_cmb_dust = np.array(Cl_cmb_dust)
# Cl_cmb_sync = np.array(Cl_cmb_sync)
# Cl_cmb_dust_sync = np.array(Cl_cmb_dust_sync)

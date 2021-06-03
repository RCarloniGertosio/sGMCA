import numpy as np
import copy as cp
from utils import mad
from starlet2d import wt_trans, wt_rec, get_wt_filters
import IAE_JAX


def sgmca(X, n, **kwargs):
    """sGMCA semi-blind source separation algorithm.

    The algorithm is comprised of three steps:
      - step #1: GMCA initialization (blind step)
      - step #2: application of the model-based constraint on the spectra (semiblind step)
      - step #3: finale estimation of S (with K = 1)

    Parameters
    ----------
    X: np.ndarray
        (m,p) float array, input data, each row corresponds to a channel
    n: int
        number of sources to be estimated
    AInit: np.ndarray
        (m,n) float array, initial value for the mixing matrix. If None, GMCA-based initialization.
    ARef: np.ndarray
        (m,n_ref) or (m,) float array, reference spectra of the mix. matrix, they are fixed during step #1 (0<n_ref<n)
    nbItMin1: int
        minimum number of iterations for step #1
    nnegA: bool
        non-negativity constraint on the spectra of A which are not modeled
    nnegS: bool
        non-negativity constraint on S
    nneg: bool
        non-negativity constraint on A and S. Overrides nnegA and nnegS if not None.
    nStd: float
        noise standard deviation corrupting the data. If not provided, the Median Absolute Deviation (MAD) is used.
    nscales: int
        number of starlet detail scales
    k: float
        parameter of the k-std thresholding
    K_max: float
        maximal L0 norm of the sources. Being a percentage, it should be between 0 and 1
    L1: bool
        if False, L0 rather than L1 penalization
    doSemiblind: bool
        do step #2
    models: dict or str
        models of the spectra of A. Either is a dict of str: int (str is the model filename and int the nb of
        components following the model) or a str (same model applied to all components)
    nbItMax2: int
        maximum number of iterations for step #2
    optimProj: int
        descent algorithm of the model constraint (0: Adam, 1: Momentum, 2: RMSProp, 3:AdaGrad, 4:Nesterov, 5:SGD)
    nbItProj: int
        maximum number of iterations of the descent algorithm of the model constraint
    stepSizeProj: float
        step size of the descent algorithm of the model constraint
    thrEnd: bool
        threshold the sources during step #3
    eps: np.ndarray
        (3,) float array, stopping criteria, resp. for step #1, step #2 and model constraint
    verb: int
        verbosity level, from 0 (mute) to 5 (most talkative)

    Returns
    -------
    (np.ndarray,np.ndarray)
        estimated mixing matrix ((m,n) float array),
        estimated sources ((n,p) float array)
    """

    # Initialize given parameters
    AInit = kwargs.get('AInit', None)
    ARef = kwargs.get('ARef', None)
    nbItMin1 = kwargs.get('nbItMin1', 100)
    nneg = kwargs.get('nneg', None)
    if nneg is not None:
        nnegA = nneg
        nnegS = nneg
    else:
        nnegA = kwargs.get('nnegA', True)
        nnegS = kwargs.get('nnegS', False)
    nStd = kwargs.get('nStd', None)
    nscales = kwargs.get('nscales', 2)
    k = kwargs.get('k', 3)
    K_max = kwargs.get('K_max', .5)
    L1 = kwargs.get('L1', True)
    doSemiBlind = kwargs.get('doSemiBlind', True)
    models = kwargs.get('models', None)
    if models is None and doSemiBlind:
        raise ValueError('models must be provided if doSemiBlind')
    nbItMax2 = kwargs.get('nbItMax2', 50)
    nbItProj = kwargs.get('nbItProj', 1000)
    optimProj = kwargs.get('optimProj', 3)  # AdaGrad algorithm
    stepSizeProj = kwargs.get('stepSizeProj', 0.1)
    thrEnd = kwargs.get('thrEnd', True)
    eps = kwargs.get('eps', np.array([1e-2, 1e-6, 1e-6]))
    verb = kwargs.get('verb', 0)

    # Get size of the data
    m = np.shape(X)[0]
    p = np.shape(X)[1]

    # Project data in starlet domain
    Xwt = wt_trans(X, nscales=nscales)
    Xwt = np.reshape(Xwt, (m, p * (nscales + 1)), order="F")

    stds = None  # to remove warnings...
    if nStd is not None:
        std_dir2wt = np.sqrt((np.sum(get_wt_filters(p, nscales) ** 2, axis=0) / p))
    else:
        std_dir2wt = None  # to remove warnings...

    maxNbAnchorPoints = 2
    if doSemiBlind:
        if type(models) is str:
            models = {models: n}
        IAEModels = []  # contains the IAE objects
        IAEModelsInfo = []  # contains some needed extra info for sGMCA (same order than IAEModels)
        for fname in models:
            IAEModels.append(IAE_JAX.IAE(Model=IAE_JAX.load_model(fname), optim_proj=optimProj, niter=nbItProj,
                                         step_size=stepSizeProj, eps_cvg=eps[2]))
            IAEModelsInfo.append({'nb_sources': models[fname],
                                  'nb_AnchorPoints': np.shape(IAEModels[-1].AnchorPoints)[0]})
            if IAEModelsInfo[-1]['nb_AnchorPoints'] > maxNbAnchorPoints:
                maxNbAnchorPoints = IAEModelsInfo[-1]['nb_AnchorPoints']
    else:
        IAEModels = None  # to remove warnings...
        IAEModelsInfo = None

    # Mixing matrix initialization
    if AInit is None:
        R = np.dot(Xwt[:, :p * nscales], Xwt[:, :p * nscales].T)  # only take into account detail scales
        D, V = np.linalg.eig(R)
        A = V[:, :n].real
        step = 1
        it = -1
        if ARef is not None:
            if len(np.shape(ARef)) == 1:
                n_ref = 1
                A[:, 0] = ARef.copy()
            else:
                n_ref = np.shape(ARef)[1]
                A[:, :n_ref] = ARef.copy()
        else:
            n_ref = 0
    else:
        A = AInit.copy()
        step = 2  # directly start with semi-blind step if A provided
        it = nbItMin1
        end_step1 = it
        n_ref = 0
    A /= np.maximum(np.linalg.norm(A, axis=0), 1e-9)

    A_old = A.copy()
    S_old = np.zeros((n, p * (nscales + 1)))

    while True:

        it += 1

        if verb >= 2:
            print("Iteration #", it + 1)

        # --- Estimate the sources

        # Least-squares

        Ra = np.dot(A.T, A)
        Ua, Sa, Va = np.linalg.svd(Ra)
        Sa[Sa < np.max(Sa) * 1e-9] = np.max(Sa) * 1e-9
        iRa = np.dot(Va.T, np.dot(np.diag(1. / Sa), Ua.T))
        piA = np.dot(iRa, A.T)
        S = piA @ Xwt
        if nStd is not None:
            stds = np.sqrt(np.diag(piA @ piA.T)) * nStd

        if step == 3 and not thrEnd:
            S = piA @ X
            return A, S

        # Thresholding

        K = np.minimum(K_max / (nbItMin1 / 2 - 1) * (it + 1), K_max)
        if step == 3:
            K = 1
        if verb >= 3:
            print("Maximal L0 norm of the sources: %.1f %%" % (K * 100))

        for i in range(n):
            for j in range(nscales):
                Swtij = S[i, p * j:p * (j + 1)]
                Swt_rwij = S_old[i, p * j:p * (j + 1)]
                if nStd is None:
                    std = mad(Swtij)
                else:
                    std = stds[i] * std_dir2wt[j]
                thrd = k * std

                # Support based threshold
                if K != 1:
                    npix = np.sum(abs(Swtij) - thrd > 0)
                    Kval = np.maximum(np.int(K * npix), 5)
                    thrd = np.partition(abs(Swtij), p - Kval)[p - Kval]

                if verb >= 4:
                    print("Threshold of source %i at scale %i: %.5e" % (i + 1, j + 1, thrd))

                # Adapt the threshold if reweighting demanded
                if L1 and (K == K_max or step == 3):  # apply L1 reweighting once thresholds stabilized
                    thrd = thrd / (np.abs(Swt_rwij) / (k * std) + 1)
                else:
                    thrd = thrd * np.ones(p)

                # Apply the threshold
                Swtij[(abs(Swtij) < thrd)] = 0
                if L1:
                    indNZ = np.where(abs(Swtij) > thrd)[0]
                    Swtij[indNZ] = Swtij[indNZ] - thrd[indNZ] * np.sign(Swtij[indNZ])

                S[i, p * j:p * (j + 1)] = Swtij

        if nnegS and K >= K_max:
            nneg_p = wt_rec(np.reshape(S, (n, p, nscales + 1), order='F')) >= 0  # locate neg samples in direct dom
            S *= np.tile(nneg_p, nscales + 1)

        if step == 3:
            S = wt_rec(np.reshape(S, (n, p, nscales + 1), order='F'))  # reconstruct sources in direct domain
            return A, S

        # --- Update the mixing matrix

        # Least-squares

        Rs = S[n_ref:, :p * nscales] @ S[n_ref:, :p * nscales].T  # only take into account detail scales
        Us, Sigs, Vs = np.linalg.svd(Rs)
        Sigs[Sigs < np.max(Sigs) * 1e-9] = np.max(Sigs) * 1e-9
        iRs = Vs.T @ np.diag(1 / Sigs) @ Us.T
        piS = np.dot(S[n_ref:, :p * nscales].T, iRs)
        A[:, n_ref:] = np.dot(Xwt[:, :p * nscales], piS)

        # Constraints

        if nnegA:
            sign = np.sign(np.sum(A, axis=0))
            sign[sign == 0] = 1
            A *= sign
            A = np.maximum(A, 0)

        A /= np.maximum(np.linalg.norm(A, axis=0), 1e-9)

        if step > 1:

            # --- Model-to-source mapping and starting point initialization
            counter = np.array([IAEModelInfo['nb_sources'] for IAEModelInfo in
                                IAEModelsInfo])  # nb of sources which can be mapped to each model
            not_mapped_sources = np.arange(n)

            n_alpha = 11  # nb of points along one dimension of the mesh grid
            alpha_grid = np.linspace(0, 1, 11)

            M = np.zeros((np.sum(counter), m))  # identified spectra so far

            for IAEModelInfo in IAEModelsInfo:
                IAEModelInfo['sources'] = np.array([], dtype=int)  # model-to-source mapping
                IAEModelInfo['Lambda0'] = np.empty((0, IAEModelInfo['nb_AnchorPoints']))  # Lambda starting points
                IAEModelInfo['Amplitude0'] = np.array([])  # Amplitude starting points

            for it_map in range(np.minimum(n, np.sum(counter))):
                # Generate the grid of dimension it_map
                grid = [alpha_grid] * it_map
                meshgrid = np.reshape(np.meshgrid(*grid), (it_map, n_alpha ** it_map))
                # Delete samples where interference would dominate
                meshgrid = np.delete(meshgrid, np.where(np.sum(meshgrid, axis=0) >= 1)[0], axis=1)

                # Calculate the projection error of each spectra yet to associate with each model and each combination
                # of identified spectra so far. The calculations are parallelized per model for the sake of speed.
                Spectra = A[:, not_mapped_sources].T
                Spectra = Spectra[np.newaxis, :, :] - (meshgrid.T @ M[:it_map, :])[:, np.newaxis, :]
                Spectra = np.reshape(Spectra, (np.shape(meshgrid)[1] * len(not_mapped_sources), m))
                proj_errors = np.ones((np.shape(meshgrid)[1] * len(not_mapped_sources), len(IAEModels))) * np.inf
                for l, IAEModel in enumerate(IAEModels):
                    if counter[l] != 0:  # check if a new source can me mapped to the current model
                        output = IAEModel.fast_interpolation(Spectra)
                        proj_errors[:, l] = np.linalg.norm(Spectra - output['XRec'], axis=1) / \
                                            np.linalg.norm(Spectra, axis=1)
                if nnegA:
                    proj_errors[np.sum(Spectra, axis=1) <= 0.5, :] = np.inf  # remove cases with too many neg. coeff.
                proj_errors = np.reshape(proj_errors, (np.shape(meshgrid)[1], len(not_mapped_sources), len(IAEModels)))

                # Identify the couple source-model with the lowest projection error
                i, j, l = np.unravel_index(np.argmin(proj_errors), np.shape(proj_errors))
                alpha = meshgrid[:,
                        np.unravel_index(np.argmin(proj_errors[:, j, :]), (np.shape(meshgrid)[1], len(IAEModels)))[0]]
                spectrum = A[:, not_mapped_sources[j]] - alpha @ M[:it_map, :]  # spectrum free of interference
                output = IAEModels[l].fast_interpolation(spectrum[np.newaxis, :])

                # Save the identified spectrum, the source-to-model map and the starting point
                M[it_map, :] = np.squeeze(output['XRec'])
                IAEModelsInfo[l]['sources'] = np.append(IAEModelsInfo[l]['sources'], np.int(not_mapped_sources[j]))
                counter[l] -= 1
                IAEModelsInfo[l]['Lambda0'] = np.vstack([IAEModelsInfo[l]['Lambda0'], output['Lambda']])
                IAEModelsInfo[l]['Amplitude0'] = np.append(IAEModelsInfo[l]['Amplitude0'], output['Amplitude'])
                not_mapped_sources = np.delete(not_mapped_sources, j)

            # --- Application of the model-based constraint
            for l, IAEModel in enumerate(IAEModels):
                output = IAEModel.barycentric_span_projection(A[:, IAEModelsInfo[l]['sources']].T,
                                                              Lambda0=IAEModelsInfo[l]['Lambda0'],
                                                              Amplitude0=IAEModelsInfo[l]['Amplitude0'], niter=nbItProj)
                A[:, IAEModelsInfo[l]['sources']] = cp.copy(cp.copy(output['XRec'].T))

        # --- Post processing

        delta_S = np.sqrt(np.sum((S - S_old) ** 2) / np.sum(S ** 2))
        S_old = S.copy()

        delta_A = np.max(abs(1. - abs(np.sum(A * A_old, axis=0))))  # angular variations
        cond_A = np.linalg.cond(A)  # condition number
        A_old = A.copy()

        if verb >= 2:
            print("delta_S = %.2e - delta_A = %.2e - cond(A) = %.2f" % (delta_S, delta_A, cond_A))

        if step == 1 and it >= nbItMin1 and (delta_S <= eps[0] or it >= nbItMin1 * 2):
            if verb:
                print('End of step 1')
            end_step1 = it
            n_ref = 0
            if doSemiBlind:
                step = 2
                if verb:
                    for IAEModel in IAEModels:
                        IAEModel.verb = True
            else:
                step = 3

        elif step == 2 and it > end_step1 + 1 and (delta_S <= eps[1] or it == end_step1 + nbItMax2):
            if verb:
                print('End of step 2')
            step = 3

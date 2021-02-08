import numpy as np


# Miscellaneous

def mad(X=0, M=None):
    """Compute median absolute estimator.

    Parameters
    ----------
    X: np.ndarray
        data
    M: np.ndarray
        mask with the same size of x, optional

    Returns
    -------
    float
        mad estimate
        """
    if M is None:
        return np.median(abs(X - np.median(X))) / 0.6735
    xm = X[M == 1]
    return np.median(abs(xm - np.median(xm))) / 0.6735


# Metrics

def evaluate(A0, S0, A, S, N=None, each_source=False):
    """Computes the source performance measurements and the CA.

    Parameters
    ----------
    A0: np.ndarray
        (m,n) float array, ground truth mixing matrix
    S0: np.ndarray
        (n,p) float array, ground truth sources
    A: np.ndarray
        (m,n) float array, estimated mixing matrix
    S: np.ndarray
        (n,p) float array, estimated sources
    N: np.ndarray
        (m,p) float array, observation noise (optional)
    each_source: bool
        return the metrics for each source as well

    Returns
    -------
    dict
    """

    A, S = corr_perm(A0, A, S, norm_data=False)

    # Renormalize all data
    A0 = A0/np.maximum(1e-9, np.linalg.norm(A0, axis=0))
    S0 = S0/np.maximum(1e-9, np.linalg.norm(S0, axis=1))[:, np.newaxis]
    A = A/np.maximum(1e-9, np.linalg.norm(A, axis=0))
    S = S/np.maximum(1e-9, np.linalg.norm(S, axis=1))[:, np.newaxis]

    SDR, SIR, SNR, SAR = source_eval(S0, S, N=N, each_source=each_source)
    # SNMSE = -10*np.log10(np.sum((S-S0)**2)/np.sum(S0**2))
    # if each_source:
    #     SNMSE = np.hstack((SNMSE, -10*np.log10(np.sum((S-S0)**2, axis=1)/np.sum(S0**2, axis=1))))

    # CA = ca(A0, A)
    SAD = sad(A0, A)
    # ANMSE = -10*np.log10(np.sum((A-A0)**2)/np.sum(A0**2))

    res = {'SDR': SDR,
           'SIR': SIR,
           'SNR': SNR,
           'SAR': SAR,
           # 'SNMSE': SNMSE,
           # 'CA': CA,
           'SAD': SAD,
           # 'ANMSE': ANMSE
           }

    return res


def corr_perm(A0, A, S, inplace=False, optInd=False, norm_data=True):
    """Correct the permutation of the solution.

    Parameters
    ----------
    A0: np.ndarray
        (m,n) float array, ground truth mixing matrix
    A: np.ndarray
        (m,n) float array, estimated mixing matrix
    S: np.ndarray
        (n,p) float array, estimated sources
    inplace: bool
        in-place update of A and S
    optInd: bool
        return permutation
    norm_data: bool
        normalize the lines of S and the columns of A according to the columns of A0

    Returns
    -------
    None or np.ndarray or (np.ndarray,np.ndarray) or (np.ndarray,np.ndarray,np.ndarray)
        A (if not inplace),
        S (if not inplace),
        ind (if optInd)
    """

    n = np.shape(A0)[1]

    if not inplace:
        A = A.copy()
        S = S.copy()

    # Normalize data for comparison
    A_norm = A/np.maximum(1e-9, np.linalg.norm(A, axis=0))
    A0_norm = A0/np.maximum(1e-9, np.linalg.norm(A0, axis=0))

    try:
        diff = abs(np.dot(np.linalg.inv(np.dot(A0_norm.T, A0_norm)), np.dot(A0_norm.T, A_norm)))
    except np.linalg.LinAlgError:
        diff = abs(np.dot(np.linalg.pinv(A0_norm), A_norm))
        print('Warning! Pseudo-inverse used.')

    ind = np.argmax(diff, axis=1)

    if len(np.unique(ind)) != n:  # if there are duplicates in ind, we proceed differently
        ind = np.ones(n)*-1
        args = np.flip(np.unravel_index(np.argsort(diff, axis=None), (n, n)), axis=1)
        for i in range(n**2):
            if ind[args[0, i]] == -1 and args[1, i] not in ind:
                ind[args[0, i]] = args[1, i]

    A[:] = A[:, ind.astype(int)]
    S[:] = S[ind.astype(int), :]

    for i in range(0, n):
        p = np.sum(A[:, i] * A0[:, i])
        if p < 0:
            S[i, :] = -S[i, :]
            A[:, i] = -A[:, i]

    # Norm data
    if norm_data:
        factor = np.maximum(1e-9, np.linalg.norm(A0, axis=0)/np.linalg.norm(A, axis=0))
        S /= factor[:, np.newaxis]
        A *= factor

    if inplace and not optInd:
        return None
    elif inplace and optInd:
        return ind
    elif not optInd:
        return A, S
    else:
        return A, S, ind


def ca(A0, A):
    """Compute the criterion on A (CA) in dB.

    Parameters
    ----------
    A0: np.ndarray
        (m,n) float array, ground truth mixing matrix
    A: np.ndarray
        (m,n) float array, estimated mixing matrix

    Returns
    -------
    float
        CA (dB)
    """

    return -10 * np.log10(np.mean(np.abs(np.dot(np.linalg.pinv(A), A0) - np.eye(np.shape(A0)[1]))))


def sad(A0, A):
    """Compute the spectral angle distance in dB.

    Parameters
    ----------
    A0: np.ndarray
        (m,n) float array, ground truth mixing matrix
    A: np.ndarray
        (m,n) float array, estimated mixing matrix

    Returns
    -------
    float
        CA (dB)
    """

    return -10*np.log10(np.mean(np.arccos(
        np.clip(np.sum(A0*A, axis=0)/np.sqrt(np.sum(A0**2, axis=0)*np.sum(A**2, axis=0)), -1, 1))))


def source_eval(S0, S, N=None, each_source=False):
    """Compute source performance measurements

    Parameters
    ----------
    S0: np.ndarray
        (n,p) float array, ground truth sources
    S: np.ndarray
        (n,p) float array, estimated sources
    N: np.ndarray
        (m,p) float array, observation noise (optional)
    each_source: bool
        return the metrics for each source as well

    Returns
    -------
    (float, float, float, float) or (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        SDR or [overall SDR, SDR of each source] (dB),
        SIR or [overall SIR, SIR of each source] (dB),
        SNR or [overall SNR, SNR of each source] (dB),
        SAR or [overall SAR, SAR of each source] (dB)
    """

    S_target = (np.sum(S*S0, axis=1)/np.sum(S0**2, axis=1))[:, np.newaxis]*S0
    Ps = np.linalg.solve(S0@S0.T, S0@S.T@S0)
    E_interf = Ps - S_target

    if N is None:
        E_noise = 0
    else:
        E_noise = (S@N.T/np.sum(N**2, axis=1))@N

    E_artif = S - (Ps + E_noise)

    SDR = 10*np.log10(np.sum(S_target**2) / np.sum((E_interf+E_noise+E_artif)**2))
    SIR = 10*np.log10(np.sum(S_target**2) / np.sum(E_interf**2))
    if N is None:
        SNR = np.inf
    else:
        SNR = 10*np.log10(np.sum((S_target+E_interf)**2) / np.sum(E_noise**2))
    SAR = 10*np.log10(np.sum((S_target+E_interf+E_noise)**2) / np.sum(E_artif**2))

    if each_source:
        SDR = np.hstack((SDR, 10*np.log10(np.sum(S_target**2, axis=1) / np.sum((E_interf+E_noise+E_artif)**2, axis=1))))
        SIR = np.hstack((SIR, 10*np.log10(np.sum(S_target**2, axis=1) / np.sum(E_interf**2, axis=1))))
        if N is None:
            SNR = SIR*np.inf
        else:
            SNR = np.hstack((SNR,
                            10*np.log10(np.sum((S_target+E_interf)**2, axis=1) / np.sum(E_noise**2, axis=1))))
        SAR = np.hstack((SAR, 10*np.log10(np.sum((S_target+E_interf+E_noise)**2, axis=1) / np.sum(E_artif**2, axis=1))))

    return SDR, SIR, SNR, SAR

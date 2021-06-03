# sGMCA

The sGMCA algorithm (Semi-blind Generalized Morphological Component Analysis) aims at solving Semi-Blind Source Separation (sBSS) problems, in which the spectra of the sought-after sources are constrained to belong to an unknown, physics-based, manifold. To that end, the  [Interpolatory AutoEncoder (IAE)](https://github.com/jbobin/IAE) framework is employed.

## Contents
1. [Introduction](#intro)
1. [Procedure](#procedure)
1. [Dependencies](#dep)
1. [Parameters](#param)
1. [Example](#example)
1. [Authors](#authors)
1. [Reference](#ref)
1. [License](#license)

<a name="intro"></a>
## Introduction

Let us consider the forward model:
> ![equation](https://latex.codecogs.com/svg.latex?\mathbf{X}%20=%20%20\mathbf{A}%20\mathbf{S}%20+%20\mathbf{N}),

where:
- ![equation](https://latex.codecogs.com/svg.latex?\mathbf{X}%20\in%20\mathbb{R}^{m%20\times%20p}) are the ![equation](https://latex.codecogs.com/svg.latex?m) multiwavelength data of size ![equation](https://latex.codecogs.com/svg.latex?p), stacked in a matrix,
- ![equation](https://latex.codecogs.com/svg.latex?\mathbf{A}%20\in%20\mathbb{R}^{m%20\times%20n}) is the mixing matrix,
- ![equation](https://latex.codecogs.com/svg.latex?\mathbf{S}%20\in%20\mathbb{R}^{n%20\times%20p}) are the ![equation](https://latex.codecogs.com/svg.latex?n) sources, stacked in a matrix,
- ![equation](https://latex.codecogs.com/svg.latex?\mathbf{N}%20\in%20\mathbb{R}^{m%20\times%20p}) is a Gaussian, independent and identically distributed noise.

The sources are assumed to be sparse in the starlet representation ![equation](https://latex.codecogs.com/svg.latex?\mathbf{W}).
Moreover, in a *semi-blind* approach, we suppose that among the ![equation](https://latex.codecogs.com/svg.latex?n) sources, ![equation](https://latex.codecogs.com/svg.latex?m) have a spectrum modeled by a IAE and ![equation](https://latex.codecogs.com/svg.latex?n-m) are fully unknown. Let ![equation](https://latex.codecogs.com/svg.latex?\mathcal{M}%20\subset%20[1%20.%20.%20n]) be the indices of the modeled components.
The sGMCA aims at minimizing the following objective function with respect to ![equation](https://latex.codecogs.com/svg.latex?\mathbf{A}) and ![equation](https://latex.codecogs.com/svg.latex?\mathbf{S}):
> ![equation](https://latex.codecogs.com/svg.latex?\min\limits_{\mathbf{A},%20\mathbf{S}}%20~\frac{1}{2}%20\left\lVert\mathbf{X}%20-%20\mathbf{AS}\right\rVert^2_2%20+\left\lVert\mathbf{\Lambda}%20\odot%20\left(\mathbf{S}\mathbf{W}^\top\right)\right\rVert_1%20+\sum_{i%20\in%20\mathcal{M}}%20\iota_{\mathcal{B}_{m_i}}\left(\mathbf{A}^i\right)%20+%20\sum_{i%20\notin%20\mathcal{M}}%20\iota_{\mathcal{O}}\left(\mathbf{A}^i\right))
>
where ![equation](https://latex.codecogs.com/svg.latex?%5Codot) denotes the element-wise product,
![equation](https://latex.codecogs.com/svg.latex?\mathbf{\Lambda}) are the sparsity regularization parameters,
![equation](https://latex.codecogs.com/svg.latex?\mathcal{O}) is the *m*-dimensional Euclidean unit sphere,
and ![equation](https://latex.codecogs.com/svg.latex?\mathcal{B}_{m_i}) is the manifold associated to the corresponding IAE model ![equation](https://latex.codecogs.com/svg.latex?m_i).

<a name="procedure"></a>
## Procedure

The sGMCA algorithm is based on GMCA, which is a BSS procedure built upon a projected alternate least-squares (pALS) minimization scheme.

The sources and the mixing matrix are initialized with GMCA; at this point, the mixing matrix and the sources are likely to be contaminated by remnants of other components.
![equation](https://latex.codecogs.com/svg.latex?\mathbf{A}) and ![equation](https://latex.codecogs.com/svg.latex?\mathbf{S}) are then updated alternatively and iteratively until converge is reached. Each update comprises a least-squares estimate, so as to minimize the data-fidelity term, followed by the application of the proximal operator of the corresponding regularization term.

<a name="dep"></a>
## Dependencies

### Required packages
- Python (last tested with v3.7.6)
- NumPy (last tested with v1.19.2)
- JAX (last tested with v0.2.5)

### Optional packages

- matplotlib (last tested with v3.3.3)
- tqdm (last tested with v4.52.0)

<a name="param"></a>
## Parameters

Below are the three parameters of the `sgmca` function which must always be provided.

| Parameter | Type                            | Information                                                                                | Default value            |
|-----------|---------------------------------|--------------------------------------------------------------------------------------------|--------------------------|
| `X`       | (m,p) float numpy.ndarray       | input data, each row corresponds to a channel                                              | N/A                      |
| `n`       | int                             | number of sources to be estimated                                                          | N/A                      |
| `models`  | dict or str                     | IAE models of the spectra. Is either a dict of str: int (str is the model filename and int the nb of components following the model) or a str (same model applied to all components) | N/A|

Below are the essential parameters of the `sgmca` function. They may be assigned their default value.

| Parameter | Type                            | Information                                                                                | Default value            |
|-----------|---------------------------------|--------------------------------------------------------------------------------------------|--------------------------|
| `nnegA`   | bool                            | non-negativity constraint on the spectra of ![equation](https://latex.codecogs.com/svg.latex?\mathbf{A}) which are not modeled   | True                    |
| `nnegS`   | bool                            | non-negativity constraint on ![equation](https://latex.codecogs.com/svg.latex?\mathbf{S})  | False                    |
| `nneg`    | bool                            | non-negativity constraint on ![equation](https://latex.codecogs.com/svg.latex?\mathbf{A}) and ![equation](https://latex.codecogs.com/svg.latex?\mathbf{S}), overrides nnegA and nnegS if not None   | None                     |
| `nStd`    | float                           | noise standard deviation contaminating ![equation](https://latex.codecogs.com/svg.latex?\mathbf{X}). If None, MAD is used to calculate the thresholds ![equation](https://latex.codecogs.com/svg.latex?\mathbf{\Lambda}). | None   |
| `nscales` | int                             | number of starlet detail scales                                                            | 2                      |
| `k`       | float                           | parameter of the k-std thresholding                                                        | 3                        |
| `K_max`   | float                           | maximal L0 norm of the sources. Being a percentage, it should be between 0 and 1.          | 0.5                      |
| `stepSizeProj`| float                       | step size of the descent algorithm of the model constraint                                         | 0.1      |
| `thrEnd`  | bool                            | perform thresholding during the finale estimation of the sources                           | True                     |
| `eps`     | (3,) float numpy.ndarray        | stopping criteria of (1) the GMCA initialization, (2) sGMCA and (3) the descent algorithm of the model constraint| [1e-2, 1e-6, 1e-6]       |
| `verb`    | int                             | verbosity level, from 0 (mute) to 5 (most talkative)                                       | 0                        |

Below are other parameters of the `sgmca` function, which can reasonably be assigned their default value.

| Parameter | Type                            | Information                                                                                | Default value            |
|-----------|---------------------------------|--------------------------------------------------------------------------------------------|--------------------------|
| `AInit`   | (m,n) float numpy.ndarray       | initial value for the mixing matrix. If None, GMCA-based initialization.                    | None                     |
| `ARef`   | (m,n_ref) or (m,) float numpy.ndarray  | reference spectra of the mixing matrix, they are fixed during step #1 (0<n_ref<n) | None                     |
| `nbItMin1`| int                             | minimum number of iterations for GMCA initialization                                               | 100                      |
| `L1`      | bool                            | if False, L0 rather than L1 penalization                                                   | True                     |
| `doSemiblind` | bool                        | perform semi-blind estimation                                                              | True                     |
| `nbItMax2`| int                             | maximum number of sGMCA iterations                                                         | 50                       |
| `optimProj`| int                            | descent algorithm of the model constraint (0: Adam, 1: Momentum, 2: RMSProp, 3: AdaGrad, 4: Nesterov, 5: SGD) | 3      |
| `nbItProj`| int                            | maximum number of iterations of the descent algorithm of the model constraint                 | 1000                     |

Below are the values returned by the `sgmca` function.

| Output    | Type                            | Information                                                                                |
|-----------|---------------------------------|--------------------------------------------------------------------------------------------|
| `A`       | (m,n) float numpy.ndarray       | Estimated mixing matrix ![equation](https://latex.codecogs.com/svg.latex?\mathbf{A})       |
| `S`       | (n,p) float numpy.ndarray       | Estimated sources ![equation](https://latex.codecogs.com/svg.latex?\mathbf{S})       |

<a name="example"></a>
## Example

Perform a sBSS on the data `X` with four sources, three of which having a spectrum modeled with a IAE.

```python
models = {"model_1": 2,  # IAE model for component of type 1, two components are expected in A
          "model_2": 1}  # IAE model for component of type 2, one component is expected in A
A, S = sgmca(X=X, n=4, nStd=1e-7, models=models)
```

<a name="authors"></a>
## Authors

- Rémi Carloni Gertosio
- Jérôme Bobin
- Fabio Acero

<a name="ref"></a>
## Reference

*TODO: add ref*

<a name="license"></a>
## License

This project is licensed under the LGPL-3.0 License.

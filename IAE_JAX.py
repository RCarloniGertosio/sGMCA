"""
Metric Learning
"""

import pickle
from jax import grad, jit
import jax.numpy as np
from jax.experimental.optimizers import adam, momentum, sgd, nesterov, adagrad, rmsprop
import numpy as onp
from tqdm import trange


###################################################
# Elementary functions
###################################################

def load_model(fname):
    dataf = open(fname + '.pkl', 'rb')
    model = pickle.load(dataf)
    dataf.close()
    return model


############################################################
# Main code
############################################################

class IAE(object):
    """
    Model - input IAE model, overrides other parameters if provided (except the number of layers)
    fname - filename for the IAE model
    AnchorPoints - anchor points
    NSize - network structure (e.g. [8, 8, 8, 8] for a 3-layer neural network of size 8)
    active_forward - activation function in the encoder
    active_backward - activation function in the decoder
    res_factor - residual injection factor in the ResNet-like architecture
    reg_parameter - weighting constant to balance between the sample and transformed domains
    cost_weight - weighting constant to balance between the sample and transformed domains in the learning stage
    reg_inv - regularization term in the barycenter computation
    simplex - simplex constraint onto the barycentric coefficients
    nneg_weights - non-negative constraint onto the barycentric coefficients
    nneg_output - non-negative constraint onto the output
    noise_level - noise level in the learning stage as in the denoising autoencoder
    cost_type - cost function (not used)
    optim_learn - optimization algorithm in the learning stage
        (0: Adam, 1: Momentum, 2: RMSprop, 3: AdaGrad, 4: Nesterov, 5: SGD)
    optim_proj - optimization algorithm in the barycentric span projection
    step_size - step size of the optimization algorithms
    niter - number of iterations of the optimization algorithms
    eps_cvg - convergence tolerance
    verb - verbose mode
    """

    def __init__(self, Model=None, fname='IAE_model', AnchorPoints=None, NSize=None, active_forward='lRelu',
                 active_backward='lRelu', res_factor=0.1, reg_parameter=1., cost_weight=None, reg_inv=1e-8,
                 simplex=True, nneg_weights=False, nneg_output=False, noise_level=None, cost_type=0, optim_learn=0,
                 optim_proj=3, step_size=1e-2, niter=5000, eps_cvg=1e-9, verb=False,
                 code_version="version_Feb_15th_2021"):
        """
        Initialization
        """

        self.Model = Model
        self.fname = fname
        self.AnchorPoints = AnchorPoints
        self.num_anchor_points = None
        self.Params = {}
        self.PhiE = None
        self.NSize = NSize
        self.nlayers = None
        self.active_forward = active_forward
        self.active_backward = active_backward
        self.res_factor = res_factor
        self.ResParams = None
        self.reg_parameter = reg_parameter
        self.cost_weight = cost_weight
        self.reg_inv = reg_inv
        self.simplex = simplex
        self.nneg_weights = nneg_weights
        self.nneg_output = nneg_output
        self.noise_level = noise_level
        self.cost_type = cost_type
        self.optim_learn = optim_learn
        self.optim_proj = optim_proj
        self.step_size = step_size
        self.niter = niter
        self.eps_cvg = eps_cvg
        self.verb = verb
        self.code_version = code_version

        self.init_parameters()

    def init_parameters(self):

        if self.NSize is None:
            if self.Model is None:
                print("Hey, there's a problem, provide either NSize or Model !")
            else:
                self.NSize = self.Model["NSize"]
        self.nlayers = len(self.NSize) - 1

        if self.AnchorPoints is None:
            if self.Model is None:
                print("Hey, there's a problem, provide either AnchorPoints or Model !")
            else:
                self.AnchorPoints = self.Model["AnchorPoints"]

        for j in range(self.nlayers):
            W0 = onp.random.randn(self.NSize[j], self.NSize[j + 1])
            self.Params["Wt" + str(j)] = W0 / onp.linalg.norm(W0)
            self.Params["bt" + str(j)] = onp.zeros(self.NSize[j + 1])

        for j in range(self.nlayers):
            W0 = onp.random.randn(self.NSize[-j - 1], self.NSize[-j - 2])
            self.Params["Wp" + str(j)] = W0 / onp.linalg.norm(W0)
            self.Params["bp" + str(j)] = onp.zeros(self.NSize[-j - 2], )

        if self.Model is not None:

            if self.verb > 2:
                print("IAE model is given")

            if self.Model["code_version"] != self.code_version:
                print('Compatibility warning!')

            dL = self.nlayers - self.Model["nlayers"]
            for j in range(self.Model["nlayers"]):
                self.Params["Wt" + str(j)] = self.Model["Params"]["Wt" + str(j)]
                self.Params["bt" + str(j)] = self.Model["Params"]["bt" + str(j)]
            for j in range(self.Model["nlayers"]):
                self.Params["Wp" + str(j + dL)] = self.Model["Params"]["Wp" + str(j)]
                self.Params["bp" + str(j + dL)] = self.Model["Params"]["bp" + str(j)]

            # self.fname = self.Model["fname"]
            self.AnchorPoints = self.Model["AnchorPoints"]
            self.active_forward = self.Model["active_forward"]
            self.active_backward = self.Model["active_backward"]
            self.res_factor = self.Model["res_factor"]
            self.reg_parameter = self.Model["reg_parameter"]
            self.cost_weight = self.Model["cost_weight"]
            self.simplex = self.Model["simplex"]
            self.nneg_output = self.Model["nneg_output"]
            self.nneg_weights = self.Model["nneg_weights"]
            self.noise_level = self.Model["noise_level"]
            self.reg_inv = self.Model["reg_inv"]
            self.cost_type = self.Model["cost_type"]
            self.optim_learn = self.Model["optim_learn"]
            self.step_size = self.Model["step_size"]
            self.niter = self.Model["niter"]
            self.eps_cvg = self.Model["eps_cvg"]
            self.verb = self.Model["verb"]
            self.code_version = self.Model["code_version"]

        self.ResParams = self.res_factor * (2 ** (onp.arange(self.nlayers) / (self.nlayers - 1)) - 1)
        self.num_anchor_points = onp.shape(self.AnchorPoints)[0]
        self.encode_anchor_points()

    def update_parameters(self, Params):
        """
        Update the parameters from learnt params
        """

        for j in range(self.nlayers):
            self.Params["Wt" + str(j)] = Params["Wt" + str(j)]
            self.Params["bt" + str(j)] = Params["bt" + str(j)]
            self.Params["Wp" + str(j)] = Params["Wp" + str(j)]
            self.Params["bp" + str(j)] = Params["bp" + str(j)]

    def learnt_params_init(self):
        """
        Update the parameters from learnt params
        """

        Params = {}

        for j in range(self.nlayers):
            Params["Wt" + str(j)] = self.Params["Wt" + str(j)]
            Params["bt" + str(j)] = self.Params["bt" + str(j)]
            Params["Wp" + str(j)] = self.Params["Wp" + str(j)]
            Params["bp" + str(j)] = self.Params["bp" + str(j)]

        return Params

    def save_model(self):

        Model = {"fname": self.fname,
                 "AnchorPoints": self.AnchorPoints,
                 "Params": self.Params,
                 "NSize": self.NSize,
                 "nlayers": self.nlayers,
                 "active_forward": self.active_forward,
                 "active_backward": self.active_backward,
                 "res_factor": self.res_factor,
                 "reg_parameter": self.reg_parameter,
                 "cost_weight": self.cost_weight,
                 "simplex": self.simplex,
                 "nneg_output": self.nneg_output,
                 "nneg_weights": self.nneg_weights,
                 "noise_level": self.noise_level,
                 "reg_inv": self.reg_inv,
                 "cost_type": self.cost_type,
                 "optim_learn": self.optim_learn,
                 "step_size": self.step_size,
                 "niter": self.niter,
                 "eps_cvg": self.eps_cvg,
                 "verb": self.verb,
                 "code_version": self.code_version}
        outfile = open(self.fname + '.pkl', 'wb')
        pickle.dump(Model, outfile)
        outfile.close()

    def get_optimizer(self, optim=None, stage='learn', step_size=None):

        if optim is None:
            if stage == 'learn':
                optim = self.optim_learn
            else:
                optim = self.optim_proj
        if step_size is None:
            step_size = self.step_size

        if optim == 1:
            if self.verb > 2:
                print("With momentum optimizer")
            opt_init, opt_update, get_params = momentum(step_size=step_size, mass=0.95)
        elif optim == 2:
            if self.verb > 2:
                print("With rmsprop optimizer")
            opt_init, opt_update, get_params = rmsprop(step_size, gamma=0.9, eps=1e-8)
        elif optim == 3:
            if self.verb > 2:
                print("With adagrad optimizer")
            opt_init, opt_update, get_params = adagrad(step_size, momentum=0.9)
        elif optim == 4:
            if self.verb > 2:
                print("With Nesterov optimizer")
            opt_init, opt_update, get_params = nesterov(step_size, 0.9)
        elif optim == 5:
            if self.verb > 2:
                print("With SGD optimizer")
            opt_init, opt_update, get_params = sgd(step_size)
        else:
            if self.verb > 2:
                print("With adam optimizer")
            opt_init, opt_update, get_params = adam(step_size)

        return opt_init, opt_update, get_params

    def encoder(self, X, W=None):

        if W is None:
            W = self.Params

        PhiX = X
        PhiE = self.AnchorPoints
        ResidualX = X
        ResidualE = self.AnchorPoints

        for l in range(self.nlayers):
            PhiX = self.activation_function(np.dot(PhiX, W["Wt" + str(l)]) + W["bt" + str(l)], direction='forward')
            PhiX += self.ResParams[l] * ResidualX

            PhiE = self.activation_function(np.dot(PhiE, W["Wt" + str(l)]) + W["bt" + str(l)], direction='forward')
            PhiE += self.ResParams[l] * ResidualE

            ResidualX = PhiX
            ResidualE = PhiE

        return PhiX, PhiE

    def encode_anchor_points(self):

        X0 = onp.ones((1, onp.shape(self.AnchorPoints)[1]))  # arbitrary X, but necessary to use encoder method
        _, self.PhiE = self.encoder(X0)

    def decoder(self, B, W=None):

        if W is None:
            W = self.Params

        XRec = B
        ResidualR = B

        for l in range(self.nlayers):
            XRec = self.activation_function(np.dot(XRec, W["Wp" + str(l)]) + W["bp" + str(l)], direction='backward')

            XRec += self.ResParams[-(l + 1)] * ResidualR

            ResidualR = XRec

        if self.nneg_output:
            XRec = XRec * (XRec > 0)

        return XRec

    def activation_function(self, X, direction='forward'):

        if direction == 'forward':
            active = self.active_forward
        else:
            active = self.active_backward

        if active == 'linear':
            Y = X
        elif active == 'Relu':
            Y = X * (X > 0)
        elif active == 'lRelu':
            Y1 = ((X > 0) * X)
            Y2 = ((X <= 0) * X * 0.01)  # with epsilon = 0.01
            Y = Y1 + Y2
        else:
            Y = np.tanh(X)

        return Y

    def interpolator(self, PhiX, PhiE):

        Lambda = np.dot(PhiX, np.dot(PhiE.T, np.linalg.inv(
            np.dot(PhiE, PhiE.T) + self.reg_inv * onp.eye(self.num_anchor_points))))
        if self.simplex:
            Lambda = Lambda / (np.sum(Lambda, axis=1)[:, np.newaxis] + 1e-3)  # not really a projection on the simplex

        B = Lambda @ PhiE

        return B, Lambda

    def learning_stage(self, X, XValidation=None, batch_size=None):
        """
        Learning the parameters
        """

        # Learning objective
        def learning_objective(W, XBatch):

            if self.noise_level is not None:
                XBatch += self.noise_level * onp.random.randn(onp.shape(XBatch)[0], onp.shape(XBatch)[1])

            # Encode data and anchor points
            PhiX, PhiE = self.encoder(XBatch, W=W)

            # Define the barycenter
            B, Lambda = self.interpolator(PhiX, PhiE)

            # Decode the barycenter
            XRec = self.decoder(B, W=W)

            # Define the cost function - We could also consider others

            if self.cost_weight is None:
                cost1 = np.linalg.norm(PhiX - B)
                cost2 = self.reg_parameter * np.linalg.norm(XRec - XBatch)
                cost = cost1 + cost2
            else:
                cost1 = np.linalg.norm(PhiX - B)
                cost2 = self.reg_parameter * np.linalg.norm((XRec - XBatch) / self.cost_weight)
                cost = cost1 + cost2

            return cost, cost1, cost2

        # Learning stage

        opt_init, opt_update, get_params = self.get_optimizer(stage='learn')

        def cost_objective(params, XBatch):
            cost, _, _ = learning_objective(params, XBatch)
            return cost

        @jit
        def update(it, XBatch, optstate):  # We could also use random batches as well
            params = get_params(optstate)
            return opt_update(it, grad(cost_objective)(params, XBatch), optstate)

        # Initializing the parameters
        initP = self.learnt_params_init()

        opt_state = opt_init(initP)

        out_val = []
        out_val1 = []
        out_val2 = []
        rel_acc = 0

        if batch_size is not None:
            batch_size = onp.minimum(batch_size, X.shape[0])
            num_batches = onp.floor(X.shape[0] / batch_size).astype('int')
        else:
            num_batches = 1
            batch_size = X.shape[0]

        t = trange(self.niter, desc='Learning stage - loss = %g, loss rel. var. = %g - ' % (0., 0.),
                   disable=not self.verb)  # may be some interference with prints

        for epoch in t:
            # We should use vmap ...
            UPerm = onp.random.permutation(X.shape[0])  # For batch-based optimization
            for b in range(num_batches):
                batch = X[UPerm[b * batch_size:(b + 1) * batch_size], :]
                opt_state = update(epoch, batch, opt_state)
                Params = get_params(opt_state)
            if XValidation is not None:
                train_acc, train_acc1, train_acc2 = learning_objective(Params, XValidation)
            else:
                train_acc, train_acc1, train_acc2 = learning_objective(Params, X)
            out_val.append(train_acc)
            out_val1.append(train_acc1)
            out_val2.append(train_acc2)

            if epoch > 50:
                average_epoch = onp.mean(out_val[len(out_val) - 50::])
                rel_acc = (abs(average_epoch - train_acc) / (average_epoch + 1e-16))

            if onp.mod(epoch, 100) == 0:
                t.set_description('Learning stage - loss = %g, loss rel. var. = %g ' % (train_acc, rel_acc))

        self.update_parameters(Params)
        if self.fname is not None:
            if self.verb > 2:
                print('Saving model...')
            self.save_model()
        self.encode_anchor_points()

        out_curves = {"total_cost": out_val, "trans_cost": out_val1, "samp_cost": out_val2}

        return self.Params, out_curves

    def fast_interpolation(self, X, Amplitude=None):

        """
        Quick forward-interpolation-backward estimation
        """

        if Amplitude is None:
            Amplitude = onp.sum(onp.abs(X), axis=1) / onp.mean(onp.sum(onp.abs(self.AnchorPoints), axis=1))
            estimate_amplitude = True
        else:
            estimate_amplitude = False
            if not hasattr(Amplitude, "__len__"):
                Amplitude = onp.ones(len(X)) * Amplitude

        # Encode data
        PhiX, _ = self.encoder(X / Amplitude[:, onp.newaxis])

        # Define the barycenter
        B, Lambda = self.interpolator(PhiX, self.PhiE)

        # Decode the barycenter
        XRec = self.decoder(B)

        if estimate_amplitude:
            Amplitude = onp.sum(XRec * X, axis=1) / onp.maximum(onp.sum(XRec ** 2, axis=1), 1e-3)

        XRec = XRec * Amplitude[:, onp.newaxis]

        Output = {"PhiX": PhiX, "PhiE": self.PhiE, "Barycenter": B, "Lambda": Lambda, "XRec": XRec,
                  "Amplitude": Amplitude}

        return Output

    ##############
    # Projection onto the barycentric span
    ##############

    def barycentric_span_projection(self, X, Amplitude=None, Lambda0=None, Amplitude0=None, niter=None, optim=None,
                                    step_size=None):

        """
        Project on the barycentric span.
        """

        if Lambda0 is None or (Amplitude0 is None and Amplitude is None):
            output = self.fast_interpolation(X=X, Amplitude=Amplitude)
            if Lambda0 is None:
                Lambda0 = output["Lambda"]
            if Amplitude0 is None and Amplitude is None:
                Amplitude0 = output["Amplitude"]
            if not hasattr(Amplitude0, "__len__") and Amplitude is None:
                Amplitude0 = onp.ones(len(X)) * Amplitude0
        if Amplitude is not None and not hasattr(Amplitude, "__len__"):
            Amplitude = onp.ones(len(X)) * Amplitude
        if niter is None:
            niter = self.niter
        if step_size is None:
            step_size = self.step_size

        Params = {}
        if not self.simplex:
            Params["Lambda"] = Lambda0
        else:  # if simplex constraint, optimization is performed on first dimensions of barycentric weights
            Params["Lambda"] = Lambda0[:, :-1]
        if Amplitude is None:
            Params["Amplitude"] = Amplitude0.copy()

        def get_cost(params):

            # Define the barycenter

            if not self.simplex:
                B = params["Lambda"] @ self.PhiE
            else:
                B = np.hstack((params["Lambda"], 1 - np.sum(params["Lambda"], axis=1)[:, np.newaxis])) @ self.PhiE

            XRec = self.decoder(B)

            if Amplitude is None:
                XRec = params["Amplitude"][:, np.newaxis] * XRec
            else:
                XRec = Amplitude[:, np.newaxis] * XRec

            return np.linalg.norm(XRec - X) ** 2

        opt_init, opt_update, get_params = self.get_optimizer(stage="project", step_size=step_size)

        @jit
        def update(i, OptState):
            params = get_params(OptState)
            return opt_update(i, grad(get_cost)(params), OptState)

        opt_state = opt_init(Params)
        train_acc_old = 1e32

        t = trange(niter, desc='Projection - loss = %g, loss rel. var. = %g - ' % (0., 0.), disable=not self.verb)

        for epoch in t:

            opt_state = update(epoch, opt_state)
            Params = get_params(opt_state)
            train_acc = get_cost(Params)
            rel_var = abs(train_acc_old - train_acc) / (train_acc_old + 1e-16)
            if rel_var < self.eps_cvg:
                break
            train_acc_old = train_acc

            if onp.mod(epoch, 100) == 0:
                t.set_description('Projection - loss = %g, loss rel. var. = %g' % (train_acc, rel_var))

        if self.verb:
            print("Finished in %i it. - loss = %g, loss rel. var. = %g " % (epoch, train_acc, rel_var))

        if self.simplex:
            Params['Lambda'] = onp.hstack((Params["Lambda"], 1 - onp.sum(Params["Lambda"], axis=1)[:, onp.newaxis]))
        if Amplitude is not None:
            Params['Amplitude'] = Amplitude
        Params['XRec'] = self.get_barycenter(Params['Lambda'], Params['Amplitude'])

        return Params

        ####

    def get_barycenter(self, Lambda, Amplitude=None):

        """
        Get barycenter for a fixed Lambda
        """

        # Get barycenter
        B = Lambda @ self.PhiE

        # Decode barycenter
        XRec = self.decoder(B)

        if Amplitude is not None:
            XRec = Amplitude[:, onp.newaxis] * XRec

        return XRec

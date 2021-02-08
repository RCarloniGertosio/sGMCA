import pickle
import jax.numpy as np
import numpy as onp
from tqdm import trange


###################################################
#  Elementary functions
###################################################

def proxSimplex(x):
    return x / (np.sum(x, axis=1)[:, np.newaxis] + 1e-3)  # Pas une vraiment projection


def load_model(fname):
    dataf = open(fname + '.pkl', 'rb')
    model = pickle.load(dataf)
    dataf.close()
    return model


############################################################
# Main code
############################################################

class IAEProjection(object):
    """
    AnchorPoints - Anchor points
    Optim - optimize
    fname - filename for the model
    NSize=[8,8,8] - network structure
    Params=None - input parameters
    reg_parameter - weighting constant to balance between the sample and transformed domains
    step_size - step size
    niter - number of iterations in the learning stage
    cost_type= - cost funtion
    CostWeight - weighting constant to balance between the sample and transformed domains
    ActiveForward - activation function in the encoder
    ActiveBackward - activation function in the decoder
    reg_inv - regularization term in the barycenter computation
    eps_cvg - convergence tolerance
    verb - verbose mode
    simplex - constraint onto the barycentric coefficients
    niter_simplex - number of iterations in the simplex calculation
    noise_level - add noise in the learning stage as in the denoising autoencoder
    ResFactor - residual injection factor in the ResNet-like archetecture
    """

    def __init__(self, AnchorPoints=None, Optim=0, fname='test', NSize=None, Params=None, reg_parameter=1,
                 step_size=1e-2, niter=5000, cost_type=0, ResFactor=0.1, CostWeight=None, PositiveWeights=False,
                 PositiveOutput=False, ActiveForward='tanh', ActiveBackward='tanh', reg_inv=1e-3, eps_cvg=1e-9,
                 verb=False, simplex=0, noise_level=None, niter_simplex=100, code_version="version_Sept_8th_2020"):
        """
        Initialization
        """

        self.NLayers = None
        self.NSize = NSize
        self.Params = None
        self.ParamsInit = Params
        self.AnchorPoints = AnchorPoints
        self.nb_AnchorPoints = 0
        self.step_size = step_size
        self.reg_parameter = reg_parameter
        self.niter = niter
        self.reg_inv = reg_inv
        self.ActiveBackward = ActiveBackward
        self.ActiveForward = ActiveForward
        self.PositiveOutput = PositiveOutput
        self.CostWeight = CostWeight
        self.cost_type = cost_type
        self.Optim = Optim
        self.verb = verb
        self.fname = fname
        self.eps_cvg = eps_cvg
        self.simplex = simplex
        self.niter_simplex = niter_simplex
        self.code_version = code_version
        self.PositiveWeights = PositiveWeights
        self.noise_level = noise_level
        self.ResFactor = ResFactor

        # For sGMCA
        self.nb_sources = None
        self.sources = None
        self.Lambda0 = None
        self.Amplitude0 = None

        self.Init_Params()

    def Init_Params(self):

        init_params = {}

        # NSize and NLayers

        if self.NSize is None:
            if self.ParamsInit is None:
                print("Hey, there's a problem, provide either NSize or Params !")
            else:
                self.NSize = self.ParamsInit["NSize"]

        init_params["NSize"] = self.NSize
        self.NLayers = len(self.NSize) - 1

        # AnchorPoints

        if self.AnchorPoints is None:
            if self.ParamsInit is None:
                print("Hey, there's a problem, provide either AnchorPoints or Params !")
            else:
                self.AnchorPoints = self.ParamsInit["AnchorPoints"]
        self.nb_AnchorPoints = np.shape(self.AnchorPoints)[0]

        init_params["reg_parameter"] = self.reg_parameter
        init_params["niter"] = self.niter
        init_params["reg_inv"] = self.reg_inv
        init_params["ActiveBackward"] = self.ActiveBackward
        init_params["ActiveForward"] = self.ActiveForward
        init_params["PositiveOutput"] = self.PositiveOutput
        init_params["CostWeight"] = self.CostWeight
        init_params["cost_type"] = self.cost_type
        init_params["Optim"] = self.Optim
        init_params["verb"] = self.verb
        init_params["fname"] = self.fname
        init_params["eps_cvg"] = self.eps_cvg
        init_params["simplex"] = self.simplex
        init_params["niter_simplex"] = self.niter_simplex
        init_params["code_version"] = self.code_version
        init_params["AnchorPoints"] = self.AnchorPoints
        init_params["NLayers"] = self.NLayers
        init_params["PositiveWeights"] = self.PositiveWeights
        init_params["noise_level"] = self.noise_level
        init_params["ResFactor"] = self.ResFactor

        for j in range(self.NLayers):
            W0 = onp.random.randn(self.NSize[j], self.NSize[j + 1])
            W0 = W0 / onp.linalg.norm(W0)
            b0 = onp.zeros(self.NSize[j + 1], )

            init_params["Wt" + str(j)] = W0
            init_params["bt" + str(j)] = b0

        for j in range(self.NLayers):
            W0 = onp.random.randn(self.NSize[-j - 1], self.NSize[-j - 2])
            W0 = W0 / onp.linalg.norm(W0)
            b0 = onp.zeros(self.NSize[-j - 2], )

            init_params["Wp" + str(j)] = W0
            init_params["bp" + str(j)] = b0

        if self.ParamsInit is not None:  # if asking for more layers

            dL = self.NLayers - (len(self.ParamsInit["NSize"]) - 1)

            for j in range(self.ParamsInit["NLayers"]):
                init_params["Wt" + str(j)] = self.ParamsInit["Wt" + str(j)]
                init_params["bt" + str(j)] = self.ParamsInit["bt" + str(j)]

            for j in range(self.ParamsInit["NLayers"]):
                init_params["Wp" + str(j + dL)] = self.ParamsInit["Wp" + str(j)]
                init_params["bp" + str(j + dL)] = self.ParamsInit["bp" + str(j)]

            init_params["reg_parameter"] = self.ParamsInit["reg_parameter"]
            init_params["niter"] = self.ParamsInit["niter"]
            init_params["reg_inv"] = self.ParamsInit["reg_inv"]
            init_params["ActiveBackward"] = self.ParamsInit["ActiveBackward"]
            init_params["ActiveForward"] = self.ParamsInit["ActiveForward"]
            init_params["PositiveOutput"] = self.ParamsInit["PositiveOutput"]
            init_params["CostWeight"] = self.ParamsInit["CostWeight"]
            init_params["cost_type"] = self.ParamsInit["cost_type"]
            init_params["Optim"] = self.ParamsInit["Optim"]
            init_params["verb"] = self.ParamsInit["verb"]
            init_params["fname"] = self.ParamsInit["fname"]
            init_params["eps_cvg"] = self.ParamsInit["eps_cvg"]
            init_params["simplex"] = self.ParamsInit["simplex"]
            init_params["niter_simplex"] = self.ParamsInit["niter_simplex"]
            init_params["code_version"] = self.ParamsInit["code_version"]
            init_params["AnchorPoints"] = self.ParamsInit["AnchorPoints"]
            init_params["noise_level"] = self.ParamsInit["noise_level"]
            init_params["ResFactor"] = self.ParamsInit["ResFactor"]

        self.Params = init_params

    def Fast_Interpolation(self, Xsamples=None, Amplitude=None, Simplex=None):

        """
        Quick forward-interpolation-backward estimation
        """

        ActiveBackward = self.Params["ActiveBackward"]
        ActiveForward = self.Params["ActiveForward"]
        LastSize = np.shape(self.Params["AnchorPoints"])[0]
        if Simplex is None:
            Simplex = self.Params["simplex"]

        if len(np.shape(Xsamples)) == 1:
            Xsamples = Xsamples[np.newaxis, :]
        if Amplitude is None:
            Amplitude = np.sum(Xsamples, axis=1)
            estimate_amplitude = True
        else:
            estimate_amplitude = False
            if not hasattr(Amplitude, "__len__"):
                Amplitude = np.ones(len(Xsamples)) * Amplitude

        l = 0

        # Define phi X:

        y = np.dot(Xsamples / Amplitude[:, np.newaxis], self.Params["Wt" + str(l)]) + self.Params["bt" + str(l)]

        if ActiveForward == 'tanh':
            phiX = np.tanh(y)
        if ActiveForward == 'linear':
            phiX = y
        if ActiveForward == 'Relu':
            phiX = y * (y > 0)
        if ActiveForward == 'lRelu':
            y1 = ((y > 0) * y)
            y2 = ((y <= 0) * y * 0.01)  # with epsilon = 0.01
            phiX = y1 + y2

        y = np.dot(self.Params["AnchorPoints"], self.Params["Wt" + str(l)]) + self.Params["bt" + str(l)]
        if ActiveForward == 'tanh':
            phiE = np.tanh(y)
        if ActiveForward == 'linear':
            phiE = y
        if ActiveForward == 'Relu':
            phiE = y * (y > 0)
        if ActiveForward == 'lRelu':
            y1 = ((y > 0) * y)
            y2 = ((y <= 0) * y * 0.01)  # with epsilon = 0.01
            phiE = y1 + y2

        residualE = phiE
        residualX = phiX

        for l in range(1, self.Params["NLayers"]):

            rho = self.Params["ResFactor"] * (np.exp((l - 1.) / (self.Params["NLayers"] - 2. + 1e-9) * np.log(2)) - 1)

            y = np.dot(phiX, self.Params["Wt" + str(l)]) + self.Params["bt" + str(l)]
            if ActiveForward == 'tanh':
                phiX = np.tanh(y)
            if ActiveForward == 'linear':
                phiX = y
            if ActiveForward == 'Relu':
                phiX = y * (y > 0)
            if ActiveForward == 'lRelu':
                y1 = ((y > 0) * y)
                y2 = ((y <= 0) * y * 0.01)  # with epsilon = 0.01
                phiX = y1 + y2

            y = np.dot(phiE, self.Params["Wt" + str(l)]) + self.Params["bt" + str(l)]
            if ActiveForward == 'tanh':
                phiE = np.tanh(y)
            if ActiveForward == 'linear':
                phiE = y
            if ActiveForward == 'Relu':
                phiE = y * (y > 0)
            if ActiveForward == 'lRelu':
                y1 = ((y > 0) * y)
                y2 = ((y <= 0) * y * 0.01)  # with epsilon = 0.01
                phiE = y1 + y2

            phiE = phiE + rho * residualE
            phiX = phiX + rho * residualX
            residualE = phiE
            residualX = phiX

        # Define the barycenter weights
        if Simplex == 0:
            Lambda = phiX @ phiE.T @ np.linalg.inv(
                phiE @ phiE.T + self.Params["reg_inv"] * np.eye(LastSize))  # Could be done differently
        elif Simplex == 1:
            Lambda = proxSimplex(
                phiX @ phiE.T @ np.linalg.inv(phiE @ phiE.T + self.Params["reg_inv"] * np.eye(LastSize)))
        else:
            L = 1e-2
            Lambda = phiX @ phiE.T @ np.linalg.inv(
                phiE @ phiE.T + self.Params["reg_inv"] * np.eye(LastSize))  # Could be done differently
            for i in range(self.Params["niter_simplex"]):
                Lambda = proxSimplex(Lambda - L * (Lambda @ phiE - phiX) @ phiE.T)

        if self.Params["PositiveWeights"]:
            Lambda = Lambda * (Lambda > 0)

        # Define the barycenter
        B = Lambda @ phiE

        # Define the reconstruction
        l = 0
        y = np.dot(B, self.Params["Wp" + str(l)]) + self.Params["bp" + str(l)]

        if ActiveBackward == 'tanh':
            Xrec = np.tanh(y)
        if ActiveBackward == 'linear':
            Xrec = y
        if ActiveBackward == 'Relu':
            Xrec = y * (y > 0)
        if ActiveBackward == 'lRelu':
            y1 = ((y > 0) * y)
            y2 = ((y <= 0) * y * 0.01)  # with epsilon = 0.01
            Xrec = y1 + y2

        residualR = Xrec

        for l in range(1, self.Params["NLayers"]):

            rho = self.Params["ResFactor"] * (np.exp(
                (self.Params["NLayers"] - 1. - l) / (self.Params["NLayers"] - 2. + 1e-9) * np.log(2.)) - 1)

            y = np.dot(Xrec, self.Params["Wp" + str(l)]) + self.Params["bp" + str(l)]

            if ActiveBackward == 'tanh':
                Xrec = np.tanh(y)
            if ActiveBackward == 'linear':
                Xrec = y
            if ActiveBackward == 'Relu':
                Xrec = y * (y > 0)
            if ActiveBackward == 'lRelu':
                y1 = ((y > 0) * y)
                y2 = ((y <= 0) * y * 0.01)  # with epsilon = 0.01
                Xrec = y1 + y2

            Xrec = Xrec + rho * residualR
            residualR = Xrec

        if self.PositiveOutput:
            Xrec = Xrec * (Xrec > 0)

        if estimate_amplitude:
            Amplitude = np.sum(Xrec * Xsamples, axis=1) / np.maximum(np.sum(Xrec ** 2, axis=1), 1e-9)

        Xrec = Xrec * Amplitude[:, np.newaxis]

        Output = {}
        Output["phiX"] = phiX
        Output["phiE"] = phiE
        Output["Barycenter"] = B
        Output["Weight"] = Lambda
        Output["Xrec"] = Xrec
        Output["Amplitude"] = Amplitude

        return Output

    ##############
    # Projection onto the barycentric span
    ##############

    def BarycentricSpan_Projection(self, Xb, Amplitude=None, Simplex=None, Lambda0=None, Amplitude0=None):

        """
        Project on the barycentric span.
        """

        from jax import grad, jit
        from jax.experimental.optimizers import adam, momentum, sgd, nesterov, adagrad, rmsprop

        AnchorPoints = self.Params["AnchorPoints"]
        NLayers = self.Params["NLayers"]
        ActiveBackward = self.Params["ActiveBackward"]
        ActiveForward = self.Params["ActiveForward"]
        ResFactor = self.Params["ResFactor"]
        if Simplex is None:
            Simplex = self.Params["simplex"]

        if Lambda0 is None:
            output = self.Fast_Interpolation(Xsamples=Xb, Amplitude=Amplitude, Simplex=Simplex)
            Lambda0 = output["Weight"]

        if Amplitude0 is None:
            Xrec0 = self.Get_Barycenter(Lambda0)
            Amplitude0 = np.sum(Xrec0 * Xb, axis=1) / np.maximum(np.sum(Xrec0 ** 2, axis=1), 1e-9)

        if Amplitude is not None and not hasattr(Amplitude, "__len__"):
            Amplitude = np.ones(len(Xb)) * Amplitude

        Params = {}
        if not Simplex:
            Params["LambdaCore"] = Lambda0
        else:
            Params["LambdaCore"] = Lambda0[:, :-1] / np.maximum(np.sum(Lambda0, axis=1)[:, np.newaxis], 1e-9)
        if Amplitude is None:
            Params["Amplitude"] = Amplitude0.copy()

        def GetCost(Params):

            l = 0

            # Define phi X:

            y = np.dot(AnchorPoints, self.Params["Wt" + str(l)]) + self.Params["bt" + str(l)]
            if ActiveForward == 'tanh':
                phiE = np.tanh(y)
            if ActiveForward == 'linear':
                phiE = y
            if ActiveForward == 'Relu':
                phiE = y * (y > 0)
            if ActiveForward == 'lRelu':
                y1 = ((y > 0) * y)
                y2 = ((y <= 0) * y * 0.01)  # with epsilon = 0.01
                phiE = y1 + y2

            residualE = phiE

            for l in range(1, NLayers):

                rho = ResFactor * (np.exp((l - 1) / (NLayers - 2 + 1e-9) * np.log(2)) - 1)

                y = np.dot(phiE, self.Params["Wt" + str(l)]) + self.Params["bt" + str(l)]
                if ActiveForward == 'tanh':
                    phiE = np.tanh(y)
                if ActiveForward == 'linear':
                    phiE = y
                if ActiveForward == 'Relu':
                    phiE = y * (y > 0)
                if ActiveForward == 'lRelu':
                    y1 = ((y > 0) * y)
                    y2 = ((y <= 0) * y * 0.01)  # with epsilon = 0.01
                    phiE = y1 + y2

                phiE = phiE + rho * residualE
                residualE = phiE

            # Define the barycenter

            if not Simplex:
                B = Params["LambdaCore"] @ phiE
            else:
                B = np.hstack((Params["LambdaCore"], 1 - np.sum(Params["LambdaCore"], axis=1)[:, np.newaxis])) @ phiE

            # Define the reconstruction
            l = 0

            y = np.dot(B, self.Params["Wp" + str(l)]) + self.Params["bp" + str(l)]

            if ActiveBackward == 'tanh':
                Xrec = np.tanh(y)
            if ActiveBackward == 'linear':
                Xrec = y
            if ActiveBackward == 'Relu':
                Xrec = y * (y > 0)
            if ActiveBackward == 'lRelu':
                y1 = ((y > 0) * y)
                y2 = ((y <= 0) * y * 0.01)  # with epsilon = 0.01
                Xrec = y1 + y2

            residualR = Xrec

            for l in range(1, NLayers):

                rho = ResFactor * (np.exp((NLayers - 1 - l) / (NLayers - 2 + 1e-9) * np.log(2)) - 1)

                y = np.dot(Xrec, self.Params["Wp" + str(l)]) + self.Params["bp" + str(l)]

                if ActiveBackward == 'tanh':
                    Xrec = np.tanh(y)
                if ActiveBackward == 'linear':
                    Xrec = y
                if ActiveBackward == 'Relu':
                    Xrec = y * (y > 0)
                if ActiveBackward == 'lRelu':
                    y1 = ((y > 0) * y)
                    y2 = ((y <= 0) * y * 0.01)  # with epsilon = 0.01
                    Xrec = y1 + y2

                Xrec = Xrec + rho * residualR
                residualR = Xrec

            if self.PositiveOutput:
                Xrec = Xrec * (Xrec > 0)

            if Amplitude is None:
                Xrec = Params["Amplitude"][:, np.newaxis] * Xrec
            else:
                Xrec = Amplitude[:, np.newaxis] * Xrec

            return np.linalg.norm(Xrec - Xb) ** 2.

        if self.Optim == 0:
            opt_init, opt_update, get_params = adam(self.step_size)
        if self.Optim == 1:
            opt_init, opt_update, get_params = momentum(step_size=self.step_size, mass=0.95)
        if self.Optim == 2:
            opt_init, opt_update, get_params = rmsprop(self.step_size, gamma=0.9, eps=1e-8)
        if self.Optim == 3:
            opt_init, opt_update, get_params = adagrad(self.step_size, momentum=0.9)
        if self.Optim == 4:
            opt_init, opt_update, get_params = nesterov(self.step_size, 0.9)
        if self.Optim == 5:
            opt_init, opt_update, get_params = sgd(self.step_size)

        opt_state = opt_init(Params)
        relvar = 1.
        train_acc_old = 1e32

        @jit
        def update(i, opt_state):
            params = get_params(opt_state)
            return opt_update(i, grad(GetCost)(params), opt_state)

        if self.verb:
            t = trange(self.niter, desc='Minimize - ')
        else:
            t = range(self.niter)

        for epoch in t:

            opt_state = update(epoch, opt_state)
            Params = get_params(opt_state)
            train_acc = GetCost(Params)
            relvar = abs(train_acc_old - train_acc) / (train_acc_old + 1e-16)
            if relvar < self.eps_cvg:
                break
            train_acc_old = train_acc

            if self.verb and onp.mod(epoch, 100) == 0:
                t.set_description('ML (loss=%g)' % train_acc + '(loss rel. var.=%g)' % relvar)

        if self.verb:
            print("Finished in ", epoch, " it. (loss=%g)" % train_acc + '(loss rel. var.=%g)' % relvar)

        if not Simplex:
            Params['Lambda'] = Params['LambdaCore']
        else:
            Params['Lambda'] = np.hstack(
                (Params["LambdaCore"], 1 - np.sum(Params["LambdaCore"], axis=1)[:, np.newaxis]))
        if Amplitude is not None:
            Params['Amplitude'] = Amplitude
        Params["Rec"] = self.Get_Barycenter(Params['Lambda'], Params["Amplitude"])

        return Params

        ####

    def Get_Barycenter(self, Lambda, Amplitude=None):

        """
        Get barycenter for a fixed Lambda
        """

        import numpy as np

        AnchorPoints = self.Params["AnchorPoints"]
        NLayers = self.Params["NLayers"]
        ActiveBackward = self.Params["ActiveBackward"]
        ActiveForward = self.Params["ActiveForward"]

        l = 0

        # Define phi X:

        y = np.dot(AnchorPoints, self.Params["Wt" + str(l)]) + self.Params["bt" + str(l)]
        if ActiveForward == 'tanh':
            phiE = np.tanh(y)
        if ActiveForward == 'linear':
            phiE = y
        if ActiveForward == 'Relu':
            phiE = y * (y > 0)
        if ActiveForward == 'lRelu':
            y1 = ((y > 0) * y)
            y2 = ((y <= 0) * y * 0.01)  # with epsilon = 0.01
            phiE = y1 + y2

        residualE = phiE

        for l in range(1, NLayers):

            rho = self.Params["ResFactor"] * (np.exp((l - 1) / (NLayers - 2 + 1e-9) * np.log(2)) - 1)

            y = np.dot(phiE, self.Params["Wt" + str(l)]) + self.Params["bt" + str(l)]
            if ActiveForward == 'tanh':
                phiE = np.tanh(y)
            if ActiveForward == 'linear':
                phiE = y
            if ActiveForward == 'Relu':
                phiE = y * (y > 0)
            if ActiveForward == 'lRelu':
                y1 = ((y > 0) * y)
                y2 = ((y <= 0) * y * 0.01)  # with epsilon = 0.01
                phiE = y1 + y2

            phiE = phiE + rho * residualE
            residualE = phiE

        # Define the barycenter
        B = Lambda @ phiE

        # Define the reconstruction
        l = 0

        y = np.dot(B, self.Params["Wp" + str(l)]) + self.Params["bp" + str(l)]

        if ActiveBackward == 'tanh':
            Xrec = np.tanh(y)
        if ActiveBackward == 'linear':
            Xrec = y
        if ActiveBackward == 'Relu':
            Xrec = y * (y > 0)
        if ActiveBackward == 'lRelu':
            y1 = ((y > 0) * y)
            y2 = ((y <= 0) * y * 0.01)  # with epsilon = 0.01
            Xrec = y1 + y2

        residualR = Xrec

        for l in range(1, NLayers):

            rho = self.Params["ResFactor"] * (np.exp((NLayers - 1 - l) / (NLayers - 2 + 1e-9) * np.log(2)) - 1)

            y = np.dot(Xrec, self.Params["Wp" + str(l)]) + self.Params["bp" + str(l)]

            if ActiveBackward == 'tanh':
                Xrec = np.tanh(y)
            if ActiveBackward == 'linear':
                Xrec = y
            if ActiveBackward == 'Relu':
                Xrec = y * (y > 0)
            if ActiveBackward == 'lRelu':
                y1 = ((y > 0) * y)
                y2 = ((y <= 0) * y * 0.01)  # with epsilon = 0.01
                Xrec = y1 + y2

            Xrec = Xrec + rho * residualR
            residualR = Xrec

        if self.PositiveOutput:
            Xrec = Xrec * (Xrec > 0)

        if Amplitude is not None:
            Xrec = Amplitude[:, np.newaxis] * Xrec

        return Xrec

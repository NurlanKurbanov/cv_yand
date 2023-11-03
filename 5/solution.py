import numpy as np

from interface import *


# ================================= 1.4.1 SGD ================================
class SGD(Optimizer):
    def __init__(self, lr):
        self.lr = lr

    def get_parameter_updater(self, parameter_shape):
        """
            :param parameter_shape: tuple, the shape of the associated parameter

            :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
                :param parameter: np.array, current parameter values
                :param parameter_grad: np.array, current gradient, dLoss/dParam

                :return: np.array, new parameter values
            """
            # your code here \/
            return parameter - self.lr * parameter_grad
            # your code here /\

        return updater


# ============================= 1.4.2 SGDMomentum ============================
class SGDMomentum(Optimizer):
    def __init__(self, lr, momentum=0.0):
        self.lr = lr
        self.momentum = momentum

    def get_parameter_updater(self, parameter_shape):
        """
            :param parameter_shape: tuple, the shape of the associated parameter

            :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
                :param parameter: np.array, current parameter values
                :param parameter_grad: np.array, current gradient, dLoss/dParam

                :return: np.array, new parameter values
            """
            # your code here \/
            updater.inertia = self.momentum * updater.inertia + self.lr * parameter_grad
            return parameter - updater.inertia
            # your code here /\

        updater.inertia = np.zeros(parameter_shape)
        return updater


# ================================ 2.1.1 ReLU ================================
class ReLU(Layer):
    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, ...)), input values

            :return: np.array((n, ...)), output values

                n - batch size
                ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        inp = np.copy(inputs)
        inp[inp < 0] = 0
        return inp
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, ...)), dLoss/dOutputs

            :return: np.array((n, ...)), dLoss/dInputs

                n - batch size
                ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        f_grad = np.copy(self.forward_inputs)
        mask = f_grad < 0
        f_grad[mask] = 0
        f_grad[~mask] = 1
        return grad_outputs * f_grad
        # your code here /\


# =============================== 2.1.2 Softmax ==============================
class Softmax(Layer):
    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d)), input values

            :return: np.array((n, d)), output values

                n - batch size
                d - number of units
        """
        # your code here \/
        stabilizing = np.max(inputs, axis=1)
        inp_stabilized = inputs - stabilizing[:, None]
        return np.exp(inp_stabilized) / np.sum(np.exp(inp_stabilized), axis=1)[:, None]
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, d)), dLoss/dOutputs

            :return: np.array((n, d)), dLoss/dInputs

                n - batch size
                d - number of units
        """
        # your code here \/
        outs = self.forward_outputs # softmax applied
        return grad_outputs * outs - np.sum(grad_outputs * outs, axis=1)[:,None] * outs
        # your code here /\


# ================================ 2.1.3 Dense ===============================
class Dense(Layer):
    def __init__(self, units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_units = units

        self.weights, self.weights_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_units, = self.input_shape
        output_units = self.output_units

        # Register weights and biases as trainable parameters
        # Note, that the parameters and gradients *must* be stored in
        # self.<p> and self.<p>_grad, where <p> is the name specified in
        # self.add_parameter

        self.weights, self.weights_grad = self.add_parameter(
            name='weights',
            shape=(input_units, output_units),
            initializer=he_initializer(input_units)
        )

        self.biases, self.biases_grad = self.add_parameter(
            name='biases',
            shape=(output_units,),
            initializer=np.zeros
        )

        self.output_shape = (output_units,)

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d)), input values

            :return: np.array((n, c)), output values

                n - batch size
                d - number of input units
                c - number of output units
        """
        # your code here \/
        return np.dot(inputs, self.weights) + self.biases
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, c)), dLoss/dOutputs

            :return: np.array((n, d)), dLoss/dInputs

                n - batch size
                d - number of input units
                c - number of output units
        """
        # your code here \/
        self.biases_grad = np.sum(grad_outputs, axis=0)
        x = np.copy(self.forward_inputs)
        self.weights_grad = np.dot(x.T, grad_outputs)

        return np.dot(grad_outputs, self.weights.T)
        # your code here /\


# ============================ 2.2.1 Crossentropy ============================
class CategoricalCrossentropy(Loss):
    def value_impl(self, y_gt, y_pred):
        """
            :param y_gt: np.array((n, d)), ground truth (correct) labels
            :param y_pred: np.array((n, d)), estimated target values

            :return: np.array((1,)), mean Loss scalar for batch

                n - batch size
                d - number of units
        """
        # your code here \/
        loss = -np.sum(y_gt * np.log(y_pred), axis=1)
        return np.ravel(np.array(np.mean(loss)))
        # your code here /\

    def gradient_impl(self, y_gt, y_pred):
        """
            :param y_gt: np.array((n, d)), ground truth (correct) labels
            :param y_pred: np.array((n, d)), estimated target values

            :return: np.array((n, d)), dLoss/dY_pred

                n - batch size
                d - number of units
        """
        # your code here \/
        denom = np.where(y_pred <= eps, eps, y_pred)
        return - (y_gt / denom) / y_gt.shape[0]
        # your code here /\


# ======================== 2.3 Train and Test on MNIST =======================
def train_mnist_model(x_train, y_train, x_valid, y_valid):
    # your code here \/
    # 1) Create a Model
    model = Model(loss=CategoricalCrossentropy(),
                  optimizer=SGDMomentum(lr=2e-3, momentum=0.7))

    # 2) Add layers to the model
    #   (don't forget to specify the input shape for the first layer)
    model.add(Dense(input_shape=(x_train.shape[1],), units=400))
    model.add(ReLU())
    model.add(Dense(units=10))
    model.add(Softmax())

    print(model)

    # 3) Train and validate the model using the provided data
    model.fit(x_train,
              y_train,
              batch_size=128,
              epochs=3,
              x_valid=x_valid,
              y_valid=y_valid)

    # your code here /\
    return model


# ============================== 3.3.2 convolve ==============================
def convolve(inputs, kernels, padding=0):
    """
        :param inputs: np.array((n, d, ih, iw)), input values
        :param kernels: np.array((c, d, kh, kw)), convolution kernels
        :param padding: int >= 0, the size of padding, 0 means 'valid'

        :return: np.array((n, c, oh, ow)), output values

            n - batch size
            d - number of input channels
            c - number of output channels
            (ih, iw) - input image shape
            (oh, ow) - output image shape
    """
    # !!! Don't change this function, it's here for your reference only !!!
    assert isinstance(padding, int) and padding >= 0
    assert inputs.ndim == 4 and kernels.ndim == 4
    assert inputs.shape[1] == kernels.shape[1]

    if os.environ.get('USE_FAST_CONVOLVE', False):
        return convolve_pytorch(inputs, kernels, padding)
    else:
        return convolve_numpy(inputs, kernels, padding)


def convolve_numpy(inputs, kernels, padding):
    """
        :param inputs: np.array((n, d, ih, iw)), input values
        :param kernels: np.array((c, d, kh, kw)), convolution kernels
        :param padding: int >= 0, the size of padding, 0 means 'valid'

        :return: np.array((n, c, oh, ow)), output values

            n - batch size
            d - number of input channels
            c - number of output channels
            (ih, iw) - input image shape
            (oh, ow) - output image shape
    """
    # your code here \/
    res_h = inputs.shape[2] + 2 * padding - kernels.shape[2] + 1
    res_w = inputs.shape[3] + 2 * padding - kernels.shape[3] + 1
    res = np.zeros((inputs.shape[0], kernels.shape[0], res_h, res_w))

    kernels = np.flip(kernels, axis=(2, 3))

    pad_inp = np.zeros((inputs.shape[0],
                        inputs.shape[1],
                        inputs.shape[2] + 2*padding,
                        inputs.shape[3] + 2*padding))
    pad_inp[:, :, padding:inputs.shape[2] + padding, padding:inputs.shape[3] + padding] = inputs

    for i in range(res_h):
        for j in range(res_w):
            # n_:_d_h_w *c_d_h_w
            pre_cnv = pad_inp[:, :, i:i + kernels.shape[2], j:j + kernels.shape[3]][:, None, :, :, :] * kernels
            cnv = np.sum(pre_cnv, axis=(-3, -2, -1))
            res[:, :, i, j] = cnv

    return res
    # your code here /\


# =============================== 4.1.1 Conv2D ===============================
class Conv2D(Layer):
    def __init__(self, output_channels, kernel_size=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert kernel_size % 2, "Kernel size should be odd"

        self.output_channels = output_channels
        self.kernel_size = kernel_size

        self.kernels, self.kernels_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_channels, input_h, input_w = self.input_shape
        output_channels = self.output_channels
        kernel_size = self.kernel_size

        self.kernels, self.kernels_grad = self.add_parameter(
            name='kernels',
            shape=(output_channels, input_channels, kernel_size, kernel_size),
            initializer=he_initializer(input_h * input_w * input_channels)
        )

        self.biases, self.biases_grad = self.add_parameter(
            name='biases',
            shape=(output_channels,),
            initializer=np.zeros
        )

        self.output_shape = (output_channels,) + self.input_shape[1:]

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, h, w)), input values

            :return: np.array((n, c, h, w)), output values

                n - batch size
                d - number of input channels
                c - number of output channels
                (h, w) - image shape
        """
        # your code here \/
        return convolve(inputs, self.kernels, padding=self.kernel_size // 2) + self.biases[None, :, None, None]
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, c, h, w)), dLoss/dOutputs

            :return: np.array((n, d, h, w)), dLoss/dInputs

                n - batch size
                d - number of input channels
                c - number of output channels
                (h, w) - image shape
        """
        # your code here \/
        def tr(x):
            return np.transpose(x, axes=(1, 0, 2, 3))

        self.biases_grad = np.sum(np.sum(grad_outputs, axis=(-2, -1)), axis=0)

        x_flip = np.flip(self.forward_inputs, axis=(-2, -1))
        self.kernels_grad = tr(convolve(tr(x_flip), tr(grad_outputs), self.kernel_size // 2))

        k_flip = np.flip(self.kernels, axis=(-2, -1))
        return convolve(grad_outputs, tr(k_flip), self.kernel_size // 2)
        # your code here /\


# ============================== 4.1.2 Pooling2D =============================
class Pooling2D(Layer):
    def __init__(self, pool_size=2, pool_mode='max', *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert pool_mode in {'avg', 'max'}

        self.pool_size = pool_size
        self.pool_mode = pool_mode
        self.forward_idxs = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        channels, input_h, input_w = self.input_shape
        output_h, rem_h = divmod(input_h, self.pool_size)
        output_w, rem_w = divmod(input_w, self.pool_size)
        assert not rem_h, "Input height should be divisible by the pool size"
        assert not rem_w, "Input width should be divisible by the pool size"

        self.output_shape = (channels, output_h, output_w)

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, ih, iw)), input values

            :return: np.array((n, d, oh, ow)), output values

                n - batch size
                d - number of channels
                (ih, iw) - input image shape
                (oh, ow) - output image shape
        """
        # your code here \/
        windows_overlapped = np.lib.stride_tricks.sliding_window_view(inputs,
                                                                      window_shape=(self.pool_size, self.pool_size),
                                                                      axis=(-2, -1),
                                                                      writeable=True)
        windows = windows_overlapped[:, :, ::self.pool_size, ::self.pool_size]

        if self.pool_mode == 'max':
            windows_ = np.copy(windows)
            windows_for_2d_argmax = np.reshape(windows_, (inputs.shape[0], inputs.shape[1], -1, self.pool_size**2))
            max_idx = np.argmax(windows_for_2d_argmax, axis=-1)

            mask = np.zeros_like(windows_for_2d_argmax)
            np.put_along_axis(mask, max_idx[:, :, :, None], 1, axis=-1)

            mask = np.reshape(mask, newshape=windows_.shape)
            mask = np.reshape(mask.swapaxes(-3, -2), newshape=inputs.shape)
            self.forward_idxs = mask.astype('bool')

            return np.max(windows, axis=(-2, -1))
        else:
            return np.mean(windows, axis=(-2, -1))
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, d, oh, ow)), dLoss/dOutputs

            :return: np.array((n, d, ih, iw)), dLoss/dInputs

                n - batch size
                d - number of channels
                (ih, iw) - input image shape
                (oh, ow) - output image shape
        """
        # your code here \/
        def make_windows(m):
            return np.kron(m, np.ones((self.pool_size, self.pool_size)))

        return make_windows(grad_outputs) * self.forward_idxs if self.pool_mode == 'max' else make_windows(grad_outputs) / self.pool_size**2
        # your code here /\


# ============================== 4.1.3 BatchNorm =============================
class BatchNorm(Layer):
    def __init__(self, momentum=0.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum = momentum

        self.running_mean = None
        self.running_var = None

        self.beta, self.beta_grad = None, None
        self.gamma, self.gamma_grad = None, None

        self.forward_inverse_std = None
        self.forward_centered_inputs = None
        self.forward_normalized_inputs = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_channels, input_h, input_w = self.input_shape
        self.running_mean = np.zeros((input_channels,))
        self.running_var = np.ones((input_channels,))

        self.beta, self.beta_grad = self.add_parameter(
            name='beta',
            shape=(input_channels,),
            initializer=np.zeros
        )

        self.gamma, self.gamma_grad = self.add_parameter(
            name='gamma',
            shape=(input_channels,),
            initializer=np.ones
        )

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, h, w)), input values

            :return: np.array((n, d, h, w)), output values

                n - batch size
                d - number of channels
                (h, w) - image shape
        """
        # your code here \/
        if self.is_training:
            m = np.mean(inputs, axis=(0, -2, -1), keepdims=True)
            v = np.var(inputs, axis=(0, -2, -1), keepdims=True)

            self.forward_centered_inputs = inputs - m
            self.forward_inverse_std = 1 / np.sqrt(eps + np.ravel(v))
            self.forward_normalized_inputs = self.forward_centered_inputs * self.forward_inverse_std[:, None, None]

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * np.ravel(m)
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * np.ravel(v)
        else:
            self.forward_normalized_inputs = (inputs - self.running_mean[:, None, None]) / np.sqrt(eps + self.running_var[:, None, None])

        return self.gamma[:, None, None] * self.forward_normalized_inputs + self.beta[:, None, None]
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, d, h, w)), dLoss/dOutputs

            :return: np.array((n, d, h, w)), dLoss/dInputs

                n - batch size
                d - number of channels
                (h, w) - image shape
        """
        # your code here \/
        self.beta_grad = np.sum(grad_outputs, axis=(0, 2, 3))
        self.gamma_grad = np.sum(self.forward_normalized_inputs * grad_outputs, axis=(0, 2, 3))

        dl_dXnorm = grad_outputs * self.gamma[None, :, None, None]
        dl_dv = np.sum(dl_dXnorm * self.forward_normalized_inputs, axis=(0, 2, 3))
        dl_dm = np.sum(dl_dXnorm, axis=(0, 2, 3))

        num = self.forward_normalized_inputs * dl_dv[None, :, None, None] + dl_dm[None, :, None, None]
        denom = grad_outputs.shape[0] * grad_outputs.shape[2] * grad_outputs.shape[3]

        return self.forward_inverse_std[None, :, None, None] * (dl_dXnorm - num / denom)
        # your code here /\


# =============================== 4.1.4 Flatten ==============================
class Flatten(Layer):
    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        self.output_shape = (np.prod(self.input_shape),)

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, h, w)), input values

            :return: np.array((n, (d * h * w))), output values

                n - batch size
                d - number of input channels
                (h, w) - image shape
        """
        # your code here \/
        return inputs.reshape(-1, self.output_shape[0])
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, (d * h * w))), dLoss/dOutputs

            :return: np.array((n, d, h, w)), dLoss/dInputs

                n - batch size
                d - number of units
                (h, w) - input image shape
        """
        # your code here \/
        return grad_outputs.reshape(grad_outputs.shape[0], *self.input_shape)
        # your code here /\


# =============================== 4.1.5 Dropout ==============================
class Dropout(Layer):
    def __init__(self, p, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p
        self.forward_mask = None

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, ...)), input values

            :return: np.array((n, ...)), output values

                n - batch size
                ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        if self.is_training:
            dist = np.random.uniform(0,1, size=inputs.shape)
            self.forward_mask = dist >= self.p
            return inputs * self.forward_mask
        else:
            return inputs * (1 - self.p)
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, ...)), dLoss/dOutputs

            :return: np.array((n, ...)), dLoss/dInputs

                n - batch size
                ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        return grad_outputs * self.forward_mask
        # your code here /\


# ====================== 2.3 Train and Test on CIFAR-10 ======================
def train_cifar10_model(x_train, y_train, x_valid, y_valid):
    # your code here \/
    # 1) Create a Model
    model = Model(loss=CategoricalCrossentropy(),
                  optimizer=SGDMomentum(lr=1e-2, momentum=0.9))

    # 2) Add layers to the model
    #   (don't forget to specify the input shape for the first layer)
    model.add(Conv2D(input_shape=(3, 32, 32), output_channels=64))
    model.add(ReLU())
    #model.add(BatchNorm())
    model.add(Pooling2D())

    model.add(Conv2D(output_channels=128))
    model.add(ReLU())
    #model.add(BatchNorm())
    model.add(Pooling2D())

    model.add(Conv2D(output_channels=256))
    model.add(ReLU())
    #model.add(BatchNorm())
    model.add(Pooling2D())

    model.add(Flatten())
    model.add(Dense(10))
    model.add(ReLU())
    model.add(Softmax())

    print(model)

    # 3) Train and validate the model using the provided data
    model.fit(x_train, y_train, batch_size=128, epochs=5, x_valid=x_valid, y_valid=y_valid)

    # your code here /\
    return model

# ============================================================================
# xx = np.random.randint(0,10,(256,3,32,32))
# train_cifar10_model(xx, xx, xx, xx)
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.nn import init
from torch.autograd import Function
#torch.manual_seed(42)


class hConvGRUCell(nn.Module):
    """
    Generate a convolutional GRU cell
    """

    def __init__(self, hidden_size, kernel_size, timesteps, batchnorm=True, grad_method='bptt', use_attention=False, no_inh=False, lesion_alpha=False, lesion_gamma=False, lesion_mu=False, lesion_kappa=False):
        super(hConvGRUCell, self).__init__()
        self.padding = kernel_size // 2
        self.hidden_size = hidden_size
        self.batchnorm = batchnorm
        self.timesteps = timesteps
        self.use_attention = use_attention
        self.no_inh = no_inh
        
        if self.use_attention:
            self.a_w_gate = nn.Conv3d(hidden_size, hidden_size, 1, padding=1 // 2)
            self.a_u_gate = nn.Conv3d(hidden_size, hidden_size, 1, padding=1 // 2)
            init.orthogonal_(self.a_w_gate.weight)
            init.orthogonal_(self.a_u_gate.weight)
            init.constant_(self.a_w_gate.bias, 1.)  # In future try setting to -1 -- originally set to 1
            init.constant_(self.a_u_gate.bias, 1.)

        self.i_w_gate = nn.Conv2d(hidden_size, hidden_size, 1)
        self.i_u_gate = nn.Conv2d(hidden_size, hidden_size, 1)
        
        self.e_w_gate = nn.Conv2d(hidden_size, hidden_size, 1)
        self.e_u_gate = nn.Conv2d(hidden_size, hidden_size, 1)

        spatial_h_size = kernel_size
        self.h_padding = spatial_h_size // 2
        self.w_exc = nn.Parameter(torch.empty(hidden_size, hidden_size, spatial_h_size, spatial_h_size)) # noqa
        init.orthogonal_(self.w_exc)

        if not no_inh:
            self.w_inh = nn.Parameter(torch.empty(hidden_size, hidden_size, spatial_h_size, spatial_h_size))  # noqa
            init.orthogonal_(self.w_inh)

        self.alpha = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.mu = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.gamma = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.kappa = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.w = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.bn = nn.ModuleList([nn.BatchNorm3d(hidden_size, eps=1e-03, affine=True, track_running_stats=False) for i in range(2)])

        init.orthogonal_(self.i_w_gate.weight)
        init.orthogonal_(self.i_u_gate.weight)
        init.orthogonal_(self.e_w_gate.weight)
        init.orthogonal_(self.e_u_gate.weight)

        for bn in self.bn:
            init.constant_(bn.weight, 0.1)

        if not no_inh:
            init.constant_(self.alpha, 1.)
            init.constant_(self.mu, 0.)
        init.constant_(self.gamma, 0.)
        init.constant_(self.kappa, 1.)

        if self.use_attention:
            self.i_w_gate.bias.data = -self.a_w_gate.bias.data
            self.e_w_gate.bias.data = -self.a_w_gate.bias.data
            self.i_u_gate.bias.data = -self.a_u_gate.bias.data
            self.e_u_gate.bias.data = -self.a_u_gate.bias.data
        else:
            init.uniform_(self.i_w_gate.bias.data, 1, self.timesteps - 1)
            self.i_w_gate.bias.data.log()
            self.i_u_gate.bias.data.log()
            self.e_w_gate.bias.data = -self.i_w_gate.bias.data
            self.e_u_gate.bias.data = -self.i_u_gate.bias.data
        if lesion_alpha:
            self.alpha.requires_grad = False
            self.alpha.weight = 0.
        if lesion_mu:
            self.mu.requires_grad = False
            self.mu.weight = 0.
        if lesion_gamma:
            self.gamma.requires_grad = False
            self.gamma.weight = 0.
        if lesion_kappa:
            self.kappa.requires_grad = False
            self.kappa.weight = 0.

    def forward(self, input_, inhibition, excitation,  activ=F.softplus, testmode=False):  # Worked with tanh and softplus
        if self.use_attention:
            att_gate = torch.sigmoid(self.a_w_gate(input_) + self.a_u_gate(excitation))  # Attention Spotlight -- MOST RECENT WORKING

        # Gate E/I with attention immediately
        if self.use_attention:
            gated_input = input_  # * att_gate  # In activ range
            gated_excitation = att_gate * excitation  # att_gate * excitation
            gated_inhibition = inhibition
        else:
            gated_input = input_

        if not self.no_inh:
            # Compute inhibition
            inh_intx = self.bn[0](F.conv2d(gated_excitation, self.w_inh, padding=self.h_padding))  # in activ range
            inhibition_hat = activ(input_ - activ(inh_intx * (self.alpha * gated_inhibition + self.mu)))

            # Integrate inhibition
            inh_gate = torch.sigmoid(self.i_w_gate(gated_input) + self.i_u_gate(gated_inhibition))
            inhibition = (1 - inh_gate) * inhibition + inh_gate * inhibition_hat  # In activ range
        else:
            inhibition, gated_inhibition = gated_excitation, excitation

        # Pass to excitatory neurons
        exc_gate = torch.sigmoid(self.e_w_gate(gated_inhibition) + self.e_u_gate(gated_excitation))
        exc_intx = self.bn[1](F.conv2d(inhibition, self.w_exc, padding=self.h_padding))  # In activ range
        excitation_hat = activ(exc_intx * (self.kappa * inhibition + self.gamma))  # Skip connection OR add OR add by self-sim

        excitation = (1 - exc_gate) * excitation + exc_gate * excitation_hat
        if testmode:
            return inhibition, excitation, att_gate
        else:
            return inhibition, excitation


class FFhGRU(nn.Module):

    def __init__(
            self,
            dimensions,
            timesteps=8,
            kernel_size=15,
            jacobian_penalty=False,
            grad_method='bptt',
            no_inh=False,
            lesion_alpha=False,
            lesion_mu=False,
            lesion_gamma=False,
            lesion_kappa=False,
            nl=F.softplus):
        '''
        '''
        super(FFhGRU, self).__init__()
        self.timesteps = timesteps
        self.jacobian_penalty = jacobian_penalty
        self.grad_method = grad_method
        self.nl = nl
        self.ff = lambda in_dim, out_dim, kernel: nn.Conv3d(
            in_dim, out_dim, kernel_size=kernel, padding=kernel // 2)
        self.nl_layer = nn.Softplus
        self.norm_layer = nn.BatchNorm3d
        self.pool_layer = lambda kernel, stride: nn.BatchNorm3d(kernel, stride)
        self.upsample_layer = lambda scale: torch.nn.Upsample(scale_factor=scale)  # noqa

        recurrent_settings = {
            "timesteps": timesteps,
            "use_attention": True,
            "no_inh": no_inh,
            "lesion_alpha": lesion_alpha,
            "lesion_mu": lesion_mu,
            "lesion_gamma": lesion_gamma,
            "lesion_kappa": lesion_kappa,
        }

        # Features, ff kern, r kern, pool kern, pool stride
        bu_hps = [
            [18, [1, 1, 1], [1, 9, 9], [1, 2, 2], [1, 2, 2]],
            [22, [1, 1, 1], [3, 5, 5], [1, 2, 2], [1, 2, 2]],
            [26, [1, 1, 1], [3, 3, 3], [1, 2, 2], [1, 2, 2]],
        ]

        # Features, ff kern, r kern, pool kern, pool stride
        td_hps = [  # Put in bottom-up order. Ordering is reversed below.
            [18, [1, 1, 1], [1, 1, 1], [1, 2, 2], [1, 2, 2]],
            [22, [1, 1, 1], [1, 1, 1], [1, 2, 2], [1, 2, 2]],
        ]

        # Create H layers
        self.layers = []
        self.layer_descriptions = []
        in_size = 2  # Input size. Move this to args eventually.
        for layer, hp in enumerate(bu_hps):
            out_size, ff_kernel_size, r_kernel_size, pool_kernel_size, pool_stride_size = hp  # noqa
            layer_ops = []

            # First an FF layer
            layer_ops.append(self.ff(in_size, out_size, ff_kernel_size))

            # Then a nonlinearity
            layer_ops.append(self.nl_layer)

            # Then a batchnorm
            layer_ops.append(self.norm_layer(out_size))

            # Then a recurrent layer
            layer_ops.append(
                hConvGRUCell(
                    out_size,
                    r_kernel_size,
                    **recurrent_settings))

            # Then a pooling layer
            layer_ops.append(self.pool_layer(pool_kernel_size, pool_stride_size))  # noqa

            # Add ops to a list
            self.layers.append(layer_ops)

            # Add a description
            self.layer_descriptions.append(["bottom-up", layer])

            # Update in_size
            in_size = out_size

        # Create TD layers
        for layer, hp in reversed(list(enumerate(td_hps))):
            out_size, ff_kernel_size, r_kernel_size, target_layer, pool_stride_size = hp  # noqa
            layer_ops = []

            # First upsampling
            layer_ops.append(self.upsample_layer(pool_stride_size))

            # Then an FF projection
            layer_ops.append(self.ff(in_size, out_size, ff_kernel_size))

            # Then a nonlinearity
            layer_ops.append(self.nl_layer)

            # Then a batchnorm
            layer_ops.append(self.norm_layer(out_size))

            # Then a recurrent layers
            layer_ops.append(
                hConvGRUCell(
                    out_size,
                    r_kernel_size,
                    **recurrent_settings))

            # Add ops to a list
            self.layers.append(layer_ops)

            # Add a description
            self.layer_descriptions.append(["top-down", layer])

            # Update in_size
            in_size = out_size

        self.readout_conv = nn.Conv3d(out_size, 1, 1)

    def forward(self, x, testmode=False):
        # Run bottom-up then top-down loops every timestep
        for t in range(self.timesteps):
            for layer in range(len(self.layers)):
                layer_type, target_layer = self.layer_descriptions[layer]
                layer_ops = self.layers[layer]
                for op in layer_ops:
                    if "hConvGRUCell" in op.__name__ and layer_type == "bottom-up":  # noqa
                        # This is the horizontal configuration
                        if t == 0:
                            # Create hiddens
                            x_shape = x.shape
                            excitation = torch.zeros(
                                (
                                    1,  # Broadcast along N
                                    x_shape[1],
                                    x_shape[2],
                                    x_shape[3],
                                    x_shape[4]), requires_grad=False).to(x.device)  # noqa
                            inhibition = torch.zeros(
                                (
                                    1,
                                    x_shape[1],
                                    x_shape[2],
                                    x_shape[3],
                                    x_shape[4]), requires_grad=False).to(x.device)  # noqa
                            setattr(self, "{}_{}_{}".format("excitation", layer, layer_type), excitation)  # noqa
                            setattr(self, "{}_{}_{}".format("inhibition", layer, layer_type), inhibition)  # noqa
                        else:
                            excitation = getattr(self, "{}_{}_{}".format("excitation", layer, layer_type))  # noqa
                            inhibition = getattr(self, "{}_{}_{}".format("inhibition", layer, layer_type))  # noqa
                        x = op(x, inhibition=inhibition, excitation=excitation)  # noqa
                    elif "hConvGRUCell" in op.__name__ and layer_type == "top-down":  # noqa
                        # For TD, inhibition is the l+1 excitation. Top-down inhibition.  # noqa
                        # Excitation is l excitation. Interpolation between the two.  # noqa
                        excitation = getattr(self, "{}_{}_{}".format("excitation", l, layer_type))  # noqa
                        inhibition = getattr(self, "{}_{}_{}".format("inhibition", target_layer, self.layer_descriptions[target_layer]))  # noqa

                    else:
                        x = op(x)
        output = self.readout_conv(x)
        return output

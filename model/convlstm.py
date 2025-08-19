import torch.nn as nn
import torch


class ConvLSTMCell(nn.Module):
    """
    A single ConvLSTM cell.
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        # Combined convolution for all gates
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        
        # Split into the 4 gates
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        # Apply activations
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        # Calculate next cell state
        c_next = f * c_cur + i * g
        # Calculate next hidden state
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):
    """
    A full ConvLSTM layer that can handle sequences.
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers=1, batch_first=False, bias=True):
        super(ConvLSTM, self).__init__()
        self.batch_first = batch_first
        self.num_layers = num_layers
        
        # Create a list of ConvLSTMCells
        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim
            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=hidden_dim,
                                          kernel_size=kernel_size,
                                          bias=bias))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        if self.batch_first:
            # (b, t, c, h, w) -> (t, b, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, c, h, w = input_tensor.size()

        # Initialize hidden state if not provided
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=(b), image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(0)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[t, :, :, :, :], cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=0)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append((h, c))

        # We only return the output of the last layer
        layer_output = layer_output_list[-1]
        
        # Following standard PyTorch LSTM return format
        final_h_list = [state[0] for state in last_state_list]
        final_c_list = [state[1] for state in last_state_list]
        
        h_n = torch.stack(final_h_list, dim=0)
        c_n = torch.stack(final_c_list, dim=0)

        return layer_output, (h_n, c_n)

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states
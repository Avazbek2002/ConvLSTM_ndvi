from model.convlstm import ConvLSTM
import torch
from torch import nn
import random

class Encoder(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(Encoder, self).__init__()
        self.conv_lstm = ConvLSTM(input_dim=input_channels,
                                  hidden_dim=hidden_channels,
                                  kernel_size=kernel_size,
                                  num_layers=1,
                                  batch_first=False)

    def forward(self, input_sequence):
        # conv_lstm returns: output, (h_n, c_n)
        # h_n and c_n have shape (num_layers, b, c, h, w)
        _, (h_n, c_n) = self.conv_lstm(input_sequence)
        # Return the state of the first (and only) layer, removing the 'num_layers' dimension
        return h_n[0], c_n[0]


class Decoder(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(Decoder, self).__init__()
        self.conv_lstm = ConvLSTM(input_dim=input_channels,
                                  hidden_dim=hidden_channels,
                                  kernel_size=kernel_size,
                                  num_layers=1,
                                  batch_first=False)
        # Final conv layer to map hidden channels to output channels
        self.final_conv = nn.Conv2d(in_channels=hidden_channels,
                                    out_channels=input_channels,
                                    kernel_size=(1, 1),
                                    padding='same')

    def forward(self, input_frame, hidden_state, cell_state):
        # The ConvLSTM expects hidden state in the format (h_n, c_n)
        # where h_n and c_n have shape (num_layers, b, c, h, w).
        # We add the num_layers dimension.
        h_n = hidden_state.unsqueeze(0)
        c_n = cell_state.unsqueeze(0)
        hidden_state = list(zip(h_n, c_n))
        
        # Run one step of the ConvLSTM
        output, (next_h, next_c) = self.conv_lstm(input_frame, hidden_state)

        # Squeeze the time dimension and pass to the final conv layer
        predicted_frame = self.final_conv(output.squeeze(0))
        
        # Squeeze the num_layers dimension from the output states
        return predicted_frame, next_h.squeeze(0), next_c.squeeze(0)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, input_sequence, target_sequence, teacher_forcing_ratio=0.5):
        # input_sequence shape: (t_in, b, c, h, w)
        # target_sequence shape: (t_out, b, c, h, w)
        
        batch_size = input_sequence.shape[1]
        target_len = target_sequence.shape[0]
        num_channels = input_sequence.shape[2]
        height = input_sequence.shape[3]
        width = input_sequence.shape[4]

        # 1. Encode the input sequence
        encoder_hidden, encoder_cell = self.encoder(input_sequence)

        # 2. Prepare for decoding
        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell
        
        # First input to the decoder is the last frame of the input sequence
        decoder_input = input_sequence[-1, :, :, :, :].unsqueeze(0)

        # 3. Initialize a tensor to store the predicted frames
        outputs = torch.zeros(target_len, batch_size, num_channels, height, width).to(self.device)

        # 4. Recursive Generation Loop
        for t in range(target_len):
            predicted_frame, decoder_hidden, decoder_cell = self.decoder(decoder_input, decoder_hidden, decoder_cell)
            outputs[t] = predicted_frame
            
            # Decide whether to use "teacher forcing"
            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
            
            if use_teacher_forcing:
                # Use the actual next frame from the target sequence as the next input
                decoder_input = target_sequence[t, :, :, :, :].unsqueeze(0)
            else:
                # Use the model's own prediction as the next input
                decoder_input = predicted_frame.unsqueeze(0)

        return outputs
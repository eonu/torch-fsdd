import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class DeepGRU(nn.Module):
    def __init__(self, n_features, n_classes, dims={'gru1': 512, 'gru2': 256, 'gru3': 128, 'fc': 256}):
        super().__init__()
        self.model = nn.ModuleDict({
            'enc': _EncoderNetwork(dims={'in': n_features, **{k:v for k, v in dims.items() if k.startswith('gru')}}),
            'attn': _AttentionModule(dims={'in': dims['gru3']}),
            'clf': _Classifier(dims={'in': dims['gru3']*2, 'fc': dims['fc'], 'out': n_classes}),
        })

    def forward(self, x, x_lengths):
        h, h_last = self.model['enc'](x, x_lengths)
        o_attn = self.model['attn'](h, h_last)
        return self.model['clf'](o_attn)

class _EncoderNetwork(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
        self.model = nn.ModuleDict({
           'gru1': nn.GRU(self.dims['in'], self.dims['gru1'], num_layers=2, batch_first=True),
           'gru2': nn.GRU(self.dims['gru1'], self.dims['gru2'], num_layers=2, batch_first=True),
           'gru3': nn.GRU(self.dims['gru2'], self.dims['gru3'], num_layers=1, batch_first=True)
        })

    def forward(self, x, x_lengths):
        # Pack the padded Tensor into a PackedSequence
        x_packed = pack_padded_sequence(x, x_lengths, batch_first=True)

        # Pass the PackedSequence through the GRUs
        h_packed, _ = self.model['gru1'](x_packed)
        h_packed, _ = self.model['gru2'](h_packed)
        h_packed, h_last = self.model['gru3'](h_packed)

        # Unpack the hidden state PackedSequence into a padded Tensor
        h_padded = pad_packed_sequence(h_packed, batch_first=True, padding_value=0.0, total_length=max(x_lengths))
        return h_padded[0], h_last
        # Shape: B x T_max x D_out, 1 x B x D_out

class _AttentionModule(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

        # Attentional context vector weights
        self.W_c = nn.Linear(self.dims['in'], self.dims['in'], bias=False)

        # Auxilliary context
        self.attn_gru = nn.GRU(input_size=self.dims['in'], hidden_size=self.dims['in'])

    def forward(self, h, h_last):
        h_last.transpose_(1, 0)
        # Shape: B x 1 x D_out

        # Calculate attentional context
        h.transpose_(1, 2)
        c = F.softmax(self.W_c(h_last) @ h, dim=0)
        c = (c @ h.transpose(2, 1)).transpose(1, 0)
        # Shape: 1 x B x D_out

        # Calculate auxilliary context
        c_aux, _ = self.attn_gru(c, h_last.transpose(1, 0))
        # Shape: 1 x B x D_out

        # Combine attentional and auxilliary context
        return torch.cat((c.squeeze(0), c_aux.squeeze(0)), dim=1)
        # Shape: B x D_out*2

class _Classifier(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
        self.model = nn.ModuleDict({
            'fc1': nn.Sequential(
                nn.BatchNorm1d(self.dims['in']),
                nn.Dropout(),
                nn.Linear(self.dims['in'], self.dims['fc'])
            ),
            'fc2': nn.Sequential(
                nn.BatchNorm1d(self.dims['fc']),
                nn.Dropout(),
                nn.Linear(self.dims['fc'], self.dims['out'])
            )
        })

    def forward(self, o_attn):
        f1 = self.model['fc1'](o_attn)
        f2 = self.model['fc2'](F.relu(f1))
        return F.log_softmax(f2, dim=1)
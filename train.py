import pandas as pd
import torch 
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import lightning as pl
from pytorch_lightning.loggers import WandbLogger
import random
import wandb

# This is the encoder model. It takes in the input size, hidden size, cell type, number of layers, dropout and bidirectional as parameters.
class Encoder(pl.LightningModule):
    def __init__(self, input_size, hidden_size, cell_type, num_layers=1, dropout=0, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.cell_type = cell_type
        if cell_type == 'LSTM':
            self.rnn = nn.LSTM
        elif cell_type == 'GRU':
            self.rnn = nn.GRU
        else:
            self.rnn = nn.RNN
        self.direction = 2 if bidirectional else 1
        self.first_cell = self.rnn(hidden_size, hidden_size, bidirectional=bidirectional, batch_first=True)
        self.rnns = nn.ModuleList([self.rnn(hidden_size*self.direction, hidden_size, bidirectional=bidirectional, batch_first=True)]*(num_layers-1))
        self.num_layers = num_layers

    def forward(self, input, hidden):
        # Getting an embedding of the character input
        embedded = self.embedding(input)
        # embedded = embedded.view(1, 1, -1)
        output = embedded

        # Passing the input to the RNNs
        output, hidden = self.first_cell(output, hidden)
        for i in range(self.num_layers-1):
            output, hidden = self.rnns[i](output, hidden)
        return output, hidden

    def init_hidden(self):
        if self.cell_type == 'LSTM':
            return torch.zeros(self.direction, self.hidden_size), torch.zeros(self.direction, self.hidden_size)
        return torch.zeros(self.direction, self.hidden_size, device=self.device)

# This is the decoder of the vanilla Seq2Seq model. It takes in the output size, hidden size, cell type, number of layers, dropout and bidirectional as parameters.
class Decoder(pl.LightningModule):
    def __init__(self, output_size, hidden_size, cell_type, num_layers=1, bidirectional=False, dropout=0):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        if cell_type == 'LSTM':
            self.cell_type = nn.LSTM
        elif cell_type == 'GRU':
            self.cell_type = nn.GRU
        else:
            self.cell_type = nn.RNN
        self.first_cell = self.cell_type(hidden_size, hidden_size, bidirectional=bidirectional, batch_first=True)
        self.direction = 2 if bidirectional else 1
        self.rnns= nn.ModuleList([self.cell_type(hidden_size*self.direction, hidden_size, bidirectional=bidirectional, batch_first=True)]*(num_layers-1))
        self.out = nn.Linear(hidden_size*self.direction, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.num_layers = num_layers

    def forward(self, input, hidden):
        # Getting an embedding of the character input
        output = self.embedding(input)
        output = nn.functional.relu(output)

        # Passing the input to the RNNs
        output, hidden = self.first_cell(output, hidden)
        for i in range(self.num_layers-1):
            output, hidden = self.rnns[i](output, hidden)
        linear_output = self.out(output)

        # Taking the softmax of the output
        output = self.softmax(self.out(output))
        if output.shape[0] == 1:
            output = output.squeeze(0)
        return output, hidden

# This is the overall Seq2Seq model. It takes in the encoder and decoder as parameters. 
class Seq2seq(pl.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)

    def forward(self, input):
        
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)
        
        # Get the batch size and the length of the input
        batched = True if len(input.shape) > 1 else False
        if not batched:
            input = input.unsqueeze(0)
        input = input.to(self.device)
        batch_size = input.shape[0]
        input_length = input.shape[1]

        # Store the outputs of the encoder. This is useful for the decoder and eventually the attention model.
        encoder_hidden = None
        encoder_hidden_outputs = torch.zeros(input_length, self.encoder.direction, batch_size, self.encoder.hidden_size, device=self.device)
        encoder_output_gate = torch.zeros(input_length, self.encoder.direction, batch_size, self.encoder.hidden_size, device=self.device)
        if self.encoder.cell_type == 'LSTM':
            a, b = [torch.zeros(self.encoder.direction, batch_size, self.encoder.hidden_size)]*2
            encoder_hidden = a.to(self.device), b.to(self.device)
        else:
            encoder_hidden = torch.zeros(self.encoder.direction, batch_size, self.encoder.hidden_size).to(self.device)

        # Pass the input through the encoder one character for every batch at a time
        for i in range(input_length):
            _, encoder_hidden_out = self.encoder(input[:, i].unsqueeze(1), encoder_hidden)
            encoder_hidden = encoder_hidden_out
            if self.encoder.cell_type == 'LSTM':
                encoder_hidden_outputs[i] = encoder_hidden_out[0]
                encoder_output_gate[i] = encoder_hidden_out[1]
            else:
                encoder_hidden_outputs[i] = encoder_hidden_out
        
        # Get the inital decoder hidden and decoder input
        if self.encoder.cell_type == 'LSTM':
            decoder_hidden = encoder_hidden_outputs[-1], encoder_output_gate[-1]
        else:
            decoder_hidden = encoder_hidden_outputs[-1]
        decoder_input = torch.zeros(batch_size, 1, dtype=torch.long, device=self.device)
        output_sequences    = []

        # Pass the input through the decoder one character for every batch at a time
        while decoder_input.item() != 1:
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            decoder_input = decoder_output.argmax(dim=-1)
            if not batched:
                decoder_input = decoder_input.unsqueeze(0)
            output_sequences.append(decoder_input)
        output_sequence = torch.tensor(output_sequences, device=self.device)
        if not batched:
            output_sequence = output_sequence.squeeze(0)
        return output_sequence
        
    # The function is similar to forward but due to modifications for the loss and for calculating the accuracy, the function
    # is reimplemented. The steps are the same as the forward function.
    def training_step(self, batch, batch_idx):
        input, target = batch
        
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)
        
        batched = True if len(input.shape) > 1 else False
        if not batched:
            input = input.unsqueeze(0)
            target = target.unsqueeze(0)
        input = input.to(self.device)
        target = target.to(self.device)
        batch_size = input.shape[0]
        input_length = input.shape[1]
        target_length = target.shape[1]

        encoder_hidden = None
        encoder_hidden_outputs = torch.zeros(input_length, self.encoder.direction, batch_size, self.encoder.hidden_size, device=self.device)
        encoder_output_gate = torch.zeros(input_length, self.encoder.direction, batch_size, self.encoder.hidden_size, device=self.device)
        if self.encoder.cell_type == 'LSTM':
            a, b = [torch.zeros(self.encoder.direction, batch_size, self.encoder.hidden_size)]*2
            encoder_hidden = a.to(self.device), b.to(self.device)
        else:
            encoder_hidden = torch.zeros(self.encoder.direction, batch_size, self.encoder.hidden_size).to(self.device)
        for i in range(input_length):
            _, encoder_hidden_out = self.encoder(input[:, i].unsqueeze(1), encoder_hidden)
            encoder_hidden = encoder_hidden_out
            if self.encoder.cell_type == 'LSTM':
                encoder_hidden_outputs[i] = encoder_hidden_out[0]
                encoder_output_gate[i] = encoder_hidden_out[1]
            else:
                encoder_hidden_outputs[i] = encoder_hidden_out
        loss = 0
        correct_words = 0
        if self.encoder.cell_type == 'LSTM':
            decoder_hidden = encoder_hidden_outputs[-1], encoder_output_gate[-1]
        else:
            decoder_hidden = encoder_hidden_outputs[-1]

        # Randomly choose whether to use teacher forcing or not
        if random.random() < 0.5: 
            # Teacher forcing

            decoder_input = target[:, 0].unsqueeze(1)
            correct = None
            for j in range(target_length):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                squeezed_output = decoder_output.squeeze(1)
                # Calculating the loss
                for i in range(batch_size):
                    loss += nn.functional.nll_loss(squeezed_output[i], target[i, j])
                decoder_input = target[:, j].unsqueeze(1)
                if correct is None:
                    correct = decoder_output.argmax(dim=-1) == target[:, j]
                else:
                    correct = (decoder_output.argmax(dim=-1) == target[:, j]) & correct
            correct_words = correct.sum()

        else:
            # Without teacher forcing

            decoder_input = target[:, 0].unsqueeze(1)
            correct = None
            for j in range(target_length):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                squeezed_output = decoder_output.squeeze(1)
                # Calculating the loss
                for i in range(batch_size):
                    loss += nn.functional.nll_loss(squeezed_output[i], target[i, j])
                decoder_input = decoder_output.argmax(dim=-1)
                if correct is None:
                    correct = decoder_input == target[:, j]
                else:
                    correct = (decoder_input == target[:, j]) & correct
            correct_words = correct.sum()

        reported_loss = loss / (batch_size * target_length)
        self.log('train_loss', reported_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', correct_words/batch_size, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        return loss
    
    # Like the training_step, due to the modifications needed, the validation_step is reimplemented.
    def validation_step(self, batch, batch_idx):
        input, target = batch
        
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)
        
        batched = True if len(input.shape) > 1 else False
        if not batched:
            input = input.unsqueeze(0)
            target = target.unsqueeze(0)
        input = input.to(self.device)
        target = target.to(self.device)
        batch_size = input.shape[0]
        input_length = input.shape[1]
        target_length = target.shape[1]

        encoder_hidden = None
        encoder_hidden_outputs = torch.zeros(input_length, self.encoder.direction, batch_size, self.encoder.hidden_size, device=self.device)
        encoder_output_gate = torch.zeros(input_length, self.encoder.direction, batch_size, self.encoder.hidden_size, device=self.device)
        if self.encoder.cell_type == 'LSTM':
            a, b = [torch.zeros(self.encoder.direction, batch_size, self.encoder.hidden_size)]*2
            encoder_hidden = a.to(self.device), b.to(self.device)
        else:
            encoder_hidden = torch.zeros(self.encoder.direction, batch_size, self.encoder.hidden_size).to(self.device)
        for i in range(input_length):
            _, encoder_hidden_out = self.encoder(input[:, i].unsqueeze(1), encoder_hidden)
            encoder_hidden = encoder_hidden_out
            if self.encoder.cell_type == 'LSTM':
                encoder_hidden_outputs[i] = encoder_hidden_out[0]
                encoder_output_gate[i] = encoder_hidden_out[1]
            else:
                encoder_hidden_outputs[i] = encoder_hidden_out
        loss = 0
        correct_words = 0
        if self.encoder.cell_type == 'LSTM':
            decoder_hidden = encoder_hidden_outputs[-1], encoder_output_gate[-1]
        else:
            decoder_hidden = encoder_hidden_outputs[-1]
        decoder_input = target[:, 0].unsqueeze(1)
        correct = None
        for j in range(target_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            squeezed_output = decoder_output.squeeze(1)
            for i in range(batch_size):
                loss += nn.functional.nll_loss(squeezed_output[i], target[i, j])
            decoder_input = decoder_output.argmax(dim=-1)
            if correct is None:
                correct = decoder_input == target[:, j]
            else:
                correct = (decoder_input == target[:, j]) & correct
        correct_words = correct.sum()
        reported_loss = loss / (batch_size * target_length)
        self.log('val_loss', reported_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', correct_words/batch_size, on_epoch=True, prog_bar=True, logger=True)
        return loss
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

# This is the decoder for the attention model. It takes in the output size, hidden size, attention size, cell type, number of layers, dropout and bidirectional as parameters.
class AttnDecoder(pl.LightningModule):
    def __init__(self, output_size, hidden_size, attention_size, cell_type, num_layers=1, bidirectional=False, dropout=0):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        if cell_type == 'LSTM':
            self.cell_type = nn.LSTM
        elif cell_type == 'GRU':
            self.cell_type = nn.GRU
        else:
            self.cell_type = nn.RNN
        self.first_cell = self.cell_type(hidden_size, hidden_size, bidirectional=bidirectional, batch_first=True)
        self.direction = 2 if bidirectional else 1
        self.rnns= nn.ModuleList([self.cell_type(hidden_size*self.direction, hidden_size, bidirectional=bidirectional, batch_first=True)]*(num_layers-1))
        self.out = nn.Linear(hidden_size*self.direction, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.num_layers = num_layers

        # These are the weights required for implementing the attention model.
        self.Uattn = nn.Linear(hidden_size*self.direction, attention_size)
        self.Wattn = nn.Linear(hidden_size*self.direction, attention_size)
        self.Vattn = nn.Linear(attention_size, 1)

        self.attn_combine = nn.Linear(hidden_size + hidden_size*self.direction, hidden_size)

    def forward(self, input, hidden, encoder_outputs):
        # Calculating at, the attention weights for the encoder outputs
        encoder_outputs_flat = encoder_outputs.transpose(1, 2).flatten(2)
        hidden_flat = None
        if self.cell_type == nn.LSTM:
            hidden_flat = hidden[0].transpose(0, 1).flatten(1)
        else:
            hidden_flat = hidden.transpose(0, 1).flatten(1)
        encoder_part = self.Uattn(encoder_outputs_flat)
        decoder_part = self.Wattn(hidden_flat.repeat(encoder_outputs.shape[0], 1, 1))

        # ejt = torch.tanh(Uattn*encoder_output + Wattn*decoder_hidden)
        ejt = torch.tanh(encoder_part + decoder_part)

        # at = Vattn(ejt)
        at = self.Vattn(ejt).squeeze(-1)
        at = at.transpose(0, 1).unsqueeze(1)

        # at is now the normalized attention weights
        at = nn.functional.softmax(at, dim=-1)
        encoder_outputs_flat = encoder_outputs_flat.transpose(0, 1)

        # getting the context vector
        context = torch.bmm(at, encoder_outputs_flat).squeeze(1)
        
        output = self.embedding(input)
        output = nn.functional.relu(output)

        # We concatenate the context vector with the output of the decoder and pass it through the linear layer
        # This is so that the RNNs can learn to use the context vector through a smaller dimension which is 
        # comparable with the seq2seq model.
        output = torch.cat((output.squeeze(1), context), dim=-1).unsqueeze(1)
        output = self.attn_combine(output)
        output, hidden = self.first_cell(output, hidden)
        for i in range(self.num_layers-1):
            output, hidden = self.rnns[i](output, hidden)
        linear_output = self.out(output)
        output = self.softmax(self.out(output))
        if output.shape[0] == 1:
            output = output.squeeze(0)
        return output, hidden, at

# This is the overall attention model. It takes in the encoder and decoder as parameters.
# This is very similar to the Seq2Seq model but with some changes to accomodate for the AttnDecoder
class AttnSeq2seq(pl.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)

    def forward(self, input):
        
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)
        
        batched = True if len(input.shape) > 1 else False
        if not batched:
            input = input.unsqueeze(0)
        input = input.to(self.device)
        batch_size = input.shape[0]
        input_length = input.shape[1]

        encoder_hidden = None
        encoder_hidden_outputs = torch.zeros(input_length, self.encoder.direction, batch_size, self.encoder.hidden_size, device=self.device)
        encoder_output_gate = torch.zeros(input_length, self.encoder.direction, batch_size, self.encoder.hidden_size, device=self.device)
        if self.encoder.cell_type == 'LSTM':
            a, b = [torch.zeros(self.encoder.direction, batch_size, self.encoder.hidden_size)]*2
            encoder_hidden = a.to(self.device), b.to(self.device)
        else:
            encoder_hidden = torch.zeros(self.encoder.direction, batch_size, self.encoder.hidden_size).to(self.device)
        for i in range(input_length):
            # print(input[:, i].shape, encoder_hidden.shape)
            _, encoder_hidden_out = self.encoder(input[:, i].unsqueeze(1), encoder_hidden)
            encoder_hidden = encoder_hidden_out
            if self.encoder.cell_type == 'LSTM':
                encoder_hidden_outputs[i] = encoder_hidden_out[0]
                encoder_output_gate[i] = encoder_hidden_out[1]
            else:
                encoder_hidden_outputs[i] = encoder_hidden_out
        if self.encoder.cell_type == 'LSTM':
            decoder_hidden = encoder_hidden_outputs[-1], encoder_output_gate[-1]
        else:
            decoder_hidden = encoder_hidden_outputs[-1]
        decoder_input = torch.zeros(batch_size, 1, dtype=torch.long, device=self.device)
        output_sequences = []
        attention_weights = [] 
        for j in range(24):
            decoder_output, decoder_hidden, at = self.decoder(decoder_input, decoder_hidden, encoder_hidden_outputs)
            decoder_input = decoder_output.argmax(dim=-1)
            output_sequences.append(decoder_input.item ())
            attention_weights.append(at)
        output_sequence = torch.tensor(output_sequences, device=self.device)
        # attention_weights = torch.tensor(attention_weights, device=self.device)
        if not batched:
            output_sequence = output_sequence.squeeze(0)
        return output_sequence, attention_weights
        
    def training_step(self, batch, batch_idx):
        input, target = batch
        
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)
        
        batched = True if len(input.shape) > 1 else False
        if not batched:
            input = input.unsqueeze(0)
            target = target.unsqueeze(0)
        input = input.to(self.device)
        target = target.to(self.device)
        batch_size = input.shape[0]
        input_length = input.shape[1]
        target_length = target.shape[1]

        encoder_hidden = None
        encoder_hidden_outputs = torch.zeros(input_length, self.encoder.direction, batch_size, self.encoder.hidden_size, device=self.device)
        encoder_output_gate = torch.zeros(input_length, self.encoder.direction, batch_size, self.encoder.hidden_size, device=self.device)
        if self.encoder.cell_type == 'LSTM':
            a, b = [torch.zeros(self.encoder.direction, batch_size, self.encoder.hidden_size)]*2
            encoder_hidden = a.to(self.device), b.to(self.device)
        else:
            encoder_hidden = torch.zeros(self.encoder.direction, batch_size, self.encoder.hidden_size).to(self.device)
        for i in range(input_length):
            # print(input[:, i].shape, encoder_hidden.shape)
            _, encoder_hidden_out = self.encoder(input[:, i].unsqueeze(1), encoder_hidden)
            encoder_hidden = encoder_hidden_out
            if self.encoder.cell_type == 'LSTM':
                encoder_hidden_outputs[i] = encoder_hidden_out[0]
                encoder_output_gate[i] = encoder_hidden_out[1]
            else:
                encoder_hidden_outputs[i] = encoder_hidden_out
        loss = 0
        correct_words = 0
        if self.encoder.cell_type == 'LSTM':
            decoder_hidden = encoder_hidden_outputs[-1], encoder_output_gate[-1]
        else:
            decoder_hidden = encoder_hidden_outputs[-1]
        if random.random() < 0.5: 
            decoder_input = target[:, 0].unsqueeze(1)
            correct = None
            for j in range(target_length):
                decoder_output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_hidden_outputs)
                squeezed_output = decoder_output.squeeze(1)
                for i in range(batch_size):
                    loss += nn.functional.nll_loss(squeezed_output[i], target[i, j])
                decoder_input = target[:, j].unsqueeze(1)
                if correct is None:
                    correct = decoder_output.argmax(dim=-1) == target[:, j]
                else:
                    correct = (decoder_output.argmax(dim=-1) == target[:, j]) & correct
            correct_words = correct.sum()

        else:
            decoder_input = target[:, 0].unsqueeze(1)
            correct = None
            for j in range(target_length):
                decoder_output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_hidden_outputs)
                squeezed_output = decoder_output.squeeze(1)
                for i in range(batch_size):
                    loss += nn.functional.nll_loss(squeezed_output[i], target[i, j])
                decoder_input = decoder_output.argmax(dim=-1)
                if correct is None:
                    correct = decoder_input == target[:, j]
                else:
                    correct = (decoder_input == target[:, j]) & correct
            correct_words = correct.sum()

        # print(correct_words, batch_size, correct_words/batch_size)
        reported_loss = loss / (batch_size * target_length)
        self.log('train_loss', reported_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', correct_words/batch_size, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        return loss
    def validation_step(self, batch, batch_idx):
        input, target = batch
        
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)
        
        batched = True if len(input.shape) > 1 else False
        if not batched:
            input = input.unsqueeze(0)
            target = target.unsqueeze(0)
        input = input.to(self.device)
        target = target.to(self.device)
        batch_size = input.shape[0]
        input_length = input.shape[1]
        target_length = target.shape[1]

        encoder_hidden = None
        encoder_hidden_outputs = torch.zeros(input_length, self.encoder.direction, batch_size, self.encoder.hidden_size, device=self.device)
        encoder_output_gate = torch.zeros(input_length, self.encoder.direction, batch_size, self.encoder.hidden_size, device=self.device)
        if self.encoder.cell_type == 'LSTM':
            a, b = [torch.zeros(self.encoder.direction, batch_size, self.encoder.hidden_size)]*2
            encoder_hidden = a.to(self.device), b.to(self.device)
        else:
            encoder_hidden = torch.zeros(self.encoder.direction, batch_size, self.encoder.hidden_size).to(self.device)
        for i in range(input_length):
            # print(input[:, i].shape, encoder_hidden.shape)
            _, encoder_hidden_out = self.encoder(input[:, i].unsqueeze(1), encoder_hidden)
            encoder_hidden = encoder_hidden_out
            if self.encoder.cell_type == 'LSTM':
                encoder_hidden_outputs[i] = encoder_hidden_out[0]
                encoder_output_gate[i] = encoder_hidden_out[1]
            else:
                encoder_hidden_outputs[i] = encoder_hidden_out
        loss = 0
        correct_words = 0
        if self.encoder.cell_type == 'LSTM':
            decoder_hidden = encoder_hidden_outputs[-1], encoder_output_gate[-1]
        else:
            decoder_hidden = encoder_hidden_outputs[-1]
        decoder_input = target[:, 0].unsqueeze(1)
        correct = None
        for j in range(target_length):
            decoder_output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_hidden_outputs)
            squeezed_output = decoder_output.squeeze(1)
            for i in range(batch_size):
                loss += nn.functional.nll_loss(squeezed_output[i], target[i, j])
            decoder_input = decoder_output.argmax(dim=-1)
            if correct is None:
                correct = decoder_input == target[:, j]
            else:
                correct = (decoder_input == target[:, j]) & correct
        correct_words = correct.sum()

        reported_loss = loss / (batch_size * target_length)
        self.log('val_loss', reported_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', correct_words/batch_size, on_epoch=True, prog_bar=True, logger=True)
        return loss
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

# Converts a given word to a tensor
# SOS : 0
# EOS : 1
# PAD : 2
# Rest of all characters will range from 3 to wherever the language ends
def convert_word_to_tensor(word, lang):
    lang_to_int = {'SOS': 0, 'EOS': 1, 'PAD': 2}
    if lang == 'eng':
        lang_to_int.update({chr(i): i-94 for i in range(97, 123)})
    elif lang == 'hin':
        lang_to_int.update({chr(i): i-2300 for i in range(2304, 2432)})
    elif lang == 'tam':
        lang_to_int.update({chr(i): i-2940 for i in range(2944, 3072)})
    
    a = [lang_to_int['SOS']]

    for i in word:
        a.append(lang_to_int[i])

    a.append(lang_to_int['EOS'])
    if len(a) < 24:
        a.extend([lang_to_int['PAD']]*(24-len(a)))
    
    return torch.tensor(a)

# A simple dataset class for the Seq2Seq model
# All words are padded to a length of 24
class AksharantarDataset(Dataset):
    def __init__(self, dataset, lang='hin'):
        super().__init__()
        self.dataset = dataset
        self.input = dataset[0]
        self.output = dataset[1]
        mask = np.array([len(elem) < 21 for elem in self.input]) & np.array([len(elem) < 21 for elem in self.output])
        self.input = self.input[mask]
        self.output = self.output[mask]
        self.len = len(self.input)
        self.lang = lang
    
    def __getitem__(self, index):
        return convert_word_to_tensor(self.input[index], 'eng'), convert_word_to_tensor(self.output[index], self.lang)
    
    def __len__(self):
        return self.len
    
# A simple datamodule for PyTorch Lightning
class CustomDataModule(pl.LightningDataModule):
    def __init__(self, dataset, val_dataset, batch_size=32, lang='hin'):
        super().__init__()
        self.dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.lang = lang

    def train_dataloader(self):
        dataset = AksharantarDataset(self.dataset, self.lang)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=2)
    def val_dataloader(self):
        dataset = AksharantarDataset(self.val_dataset, self.lang)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=2)


# Read data from a given path
def get_data(path):
    dataset = pd.read_csv(path, header=None)
    dataset = dataset.values
    input = dataset[:, 0]
    output = dataset[:, 1]
    return input, output

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-wp", "--wandb_project", help="Name of the wandb project", type=str, default="CS6910 Assignment 3")
    parser.add_argument("-we", "--wandb_entity", help="Name of the wandb entity", type=str, default="cs20b075")
    parser.add_argument("-e", "--epochs", help="Number of epochs", type=int, default=10)
    parser.add_argument("-b", "--batch_size", help="Batch size", type=int, default=32)
    parser.add_argument("-lr", "--learning_rate", help="Value of the learning rate", type=float, default=0.001)
    parser.add_argument("--train_path", help="Path to the train csv file", type=str, default="aksharantar_sampled/hin/hin_train.csv")
    parser.add_argument("--val_path", help="Path to the valid csv file", type=str, default="aksharantar_sampled/hin/hin_valid.csv")
    parser.add_argument("--attention", help="Use attention or not", type=bool, default=False)
    parser.add_argument("--save_path", help="Path to save the model", type=str, default="model.pt")
    parser.add_argument("-l", "--lang", help="Language to train on", type=str, default="hin")

    args = parser.parse_args()

    encoder = Encoder(30, 256, 'LSTM', num_layers=2, bidirectional=True)
    if args.attention:
        decoder = AttnDecoder(150, 256, 256, 'LSTM', num_layers=2, bidirectional=True)
        model = AttnSeq2seq(encoder, decoder)
    else:
        decoder = Decoder(150, 256, 'LSTM', num_layers=2, bidirectional=True)
        model = Seq2seq(encoder, decoder)
    train_dataset = get_data(args.train_path)
    val_dataset = get_data(args.val_path)
    data_module = CustomDataModule(train_dataset, val_dataset, batch_size=args.batch_size, lang=args.lang)
    
    wandb_logger = WandbLogger(project=args.wandb_project, entity=args.wandb_entity)
    trainer = pl.Trainer(max_epochs=args.epochs, logger=wandb_logger)

    trainer.fit(model, data_module)

    torch.save(model.state_dict(), args.save_path)
import mindspore
import mindspore.numpy as np
from mindspore import nn, ops, Tensor, Parameter
from mindspore.ops import operations as P
from mindspore.common.initializer import HeNormal, Normal

import random

def num2char(num):
    chars = "_0123456789abcdefghijklmnopqrstuvwxyz"
    char = chars[num]
    return char

def reduce_mul(l):
    out = 1.0
    for x in l:
        out *= x
    return out

def check_all_done(seqs):
    for seq in seqs:
        if not seq[-1]:
            return False
    return True

class SeqCharMaskRCNNC4Predictor(nn.Cell):
    def __init__(self, config):
        super(SeqCharMaskRCNNC4Predictor, self).__init__()
        # num_classes = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        num_classes = 1
        char_num_classes = config.roi.mask_head.char_class_count
        dim_reduced = config.roi.mask_head.conv_layers[-1]

        num_inputs = dim_reduced
        
        weight_init = HeNormal(mode="fan_out", nonlinearity="relu")
        bias_init = "zero"

        self.conv5_mask = nn.Conv2dTranspose(num_inputs, dim_reduced, 2, 2, weight_init=weight_init, bias_init=bias_init)
        self.mask_fcn_logits = nn.Conv2d(dim_reduced, num_classes, 1, 1, weight_init=weight_init, bias_init=bias_init)
        self.char_mask_fcn_logits = nn.Conv2d(dim_reduced, char_num_classes, 1, 1, weight_init=weight_init, bias_init=bias_init)
        self.seq = SequencePredictor(config, dim_reduced)

    def construct(self, x, decoder_targets=None, word_targets=None):
        x = nn.ReLU()(self.conv5_mask(x))
        if self.training:
            loss_seq_decoder = self.seq(
                x, decoder_targets=decoder_targets, word_targets=word_targets
            )
            return (
                self.mask_fcn_logits(x),
                self.char_mask_fcn_logits(x),
                loss_seq_decoder,
            )
        else:
            decoded_chars, decoded_scores, detailed_decoded_scores = self.seq(
                x, use_beam_search=True
            )
            return (
                self.mask_fcn_logits(x),
                self.char_mask_fcn_logits(x),
                decoded_chars,
                decoded_scores,
                detailed_decoded_scores,
            )

class SequencePredictor(nn.Cell):
    def __init__(self, config, dim_in):
        super(SequencePredictor, self).__init__()
        self.config = config
        weight_init = HeNormal(mode="fan_out", nonlinearity="relu")
        bias_init = "zero"

        self.seq_encoder = nn.CellList(
            nn.Conv2d(dim_in, 256, 3, pad_mode='pad', padding=1, weight_init=weight_init, bias_init=bias_init),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2), # TQ1: can not init weights and bias in this layer
        )
        x_onehot_size = int(config.sequence.resize_w / 2)
        y_onehot_size = int(config.sequence.resize_h / 2)
        self.seq_decoder = BahdanauAttnDecoderRNN(
            256, config.sequence.char_count, config.sequence.char_count, n_layers=1, dropout_p=0.1, onehot_size = (y_onehot_size, x_onehot_size)
        )
        # self.criterion_seq_decoder = nn.NLLLoss(ignore_index = -1, reduce=False)
        self.criterion_seq_decoder = nn.NLLLoss(ignore_index=-1, reduction="none")
        # self.rescale = nn.Upsample(size=(16, 64), mode="bilinear", align_corners=False)
        self.rescale = ops.ResizeBilinear(size=(config.sequence.resize_h, config.sequence.resize_w))

        x_weight_init = ops.eye(x_onehot_size, x_onehot_size, mindspore.float32)
        y_weight_init = ops.eye(y_onehot_size, y_onehot_size, mindspore.float32)
        self.x_onehot = nn.Embedding(x_onehot_size, x_onehot_size, embedding_table=x_weight_init)
        self.y_onehot = nn.Embedding(y_onehot_size, y_onehot_size, embedding_table=y_weight_init)

        self.concat = P.Concat(axis=1)


    def construct(self, x, decoder_targets=None, word_targets=None, use_beam_search=False):
        rescale_out = self.rescale(x)
        seq_decoder_input = self.seq_encoder(rescale_out)
        x_onehot_size = int(self.config.sequence.resize_w / 2)
        y_onehot_size = int(self.config.sequence.resize_h / 2)
        x_t, y_t = np.meshgrid(np.linspace(0, x_onehot_size - 1, x_onehot_size), np.linspace(0, y_onehot_size - 1, y_onehot_size))
        x_t = x_t.astype(mindspore.int64)
        y_t = x_t.astype(mindspore.int64)
        x_onehot_embedding = (
            self.x_onehot(x_t)
            .transpose(0, 2)
            .transpose(1, 2)
            .repeat(seq_decoder_input.shape(0), 1, 1, 1)
        )
        y_onehot_embedding = (
            self.y_onehot(y_t)
            .transpose(0, 2)
            .transpose(1, 2)
            .repeat(seq_decoder_input.shape(0), 1, 1, 1)
        )
        seq_decoder_input_loc = self.concat([seq_decoder_input, x_onehot_embedding, y_onehot_embedding])
        seq_decoder_input_reshape = (
            seq_decoder_input_loc.view(
                seq_decoder_input_loc.shape(0), seq_decoder_input_loc.shape(1), -1
            )
            .transpose(0, 2)
            .transpose(1, 2)
        )
        if self.training:
            bos_onehot = np.zeros(
                (seq_decoder_input_reshape.shape(1), 1), dtype=np.int32
            )
            bos_onehot[:, 0] = self.config.sequence.box_token
            decoder_input = Tensor(bos_onehot.tolist())
            decoder_hidden = np.zeros((seq_decoder_input_reshape.shape(1), 256))
            use_teacher_forcing = (
                True
                if random.random() < self.config.sequence.ratio
                else False
            )
            target_length = decoder_targets.shape(1)
            if use_teacher_forcing:
                # Teacher forcing: Feed the target as the next input
                for di in range(target_length):
                    decoder_output, decoder_hidden, decoder_attention = self.seq_decoder(
                        decoder_input, decoder_hidden, seq_decoder_input_reshape
                    )
                    if di == 0:
                        loss_seq_decoder = self.criterion_seq_decoder(
                            decoder_output, word_targets[:, di]
                        )
                    else:
                        loss_seq_decoder += self.criterion_seq_decoder(
                            decoder_output, word_targets[:, di]
                        )
                    decoder_input = decoder_targets[:, di]  # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                for di in range(target_length):
                    decoder_output, decoder_hidden, decoder_attention = self.seq_decoder(
                        decoder_input, decoder_hidden, seq_decoder_input_reshape
                    )
                    topv, topi = decoder_output.top_k(1)
                    decoder_input = topi.squeeze(
                        1
                    ).copy()  # detach from history as input
                    if di == 0:
                        loss_seq_decoder = self.criterion_seq_decoder(
                            decoder_output, word_targets[:, di]
                        )
                    else:
                        loss_seq_decoder += self.criterion_seq_decoder(
                            decoder_output, word_targets[:, di]
                        )
            loss_seq_decoder = loss_seq_decoder.sum() / loss_seq_decoder.shape(0)
            loss_seq_decoder = 0.2 * loss_seq_decoder
            return loss_seq_decoder
        else:
            words = []
            decoded_scores = []
            detailed_decoded_scores = []
            # real_length = 0
            if use_beam_search:
                for batch_index in range(seq_decoder_input_reshape.shape(1)):
                    decoder_hidden = np.zeros((1, 256))
                    word = []
                    char_scores = []
                    detailed_char_scores = []
                    top_seqs = self.beam_search(
                        seq_decoder_input_reshape[:, batch_index : batch_index + 1, :],
                        decoder_hidden,
                        beam_size=6,
                        max_len=self.config.sequence.max_len,
                    )
                    top_seq = top_seqs[0]
                    for character in top_seq[1:]:
                        character_index = character[0]
                        if character_index == self.config.sequence.char_count - 1:
                            char_scores.append(character[1])
                            detailed_char_scores.append(character[2])
                            break
                        else:
                            if character_index == 0:
                                word.append("~")
                                char_scores.append(0.0)
                            else:
                                word.append(num2char(character_index))
                                char_scores.append(character[1])
                                detailed_char_scores.append(character[2])
                    words.append("".join(word))
                    decoded_scores.append(char_scores)
                    detailed_decoded_scores.append(detailed_char_scores)
            else:
                for batch_index in range(seq_decoder_input_reshape.shape(1)):
                    bos_onehot = np.zeros((1, 1), dtype=np.int32)
                    bos_onehot[:, 0] = self.config.sequence.bos_token
                    decoder_input = Tensor(bos_onehot.tolist())
                    decoder_hidden = np.zeros((1, 256))
                    word = []
                    char_scores = []
                    for di in range(self.config.sequence.max_len):
                        decoder_output, decoder_hidden, decoder_attention = self.seq_decoder(
                            decoder_input,
                            decoder_hidden,
                            seq_decoder_input_reshape[
                                :, batch_index : batch_index + 1, :
                            ],
                        )
                        # decoder_attentions[di] = decoder_attention.data
                        topv, topi = decoder_output.data.top_k(1)
                        char_scores.append(topv.item())
                        if topi.item() == self.config.sequence.char_count - 1:
                            break
                        else:
                            if topi.item() == 0:
                                word.append("~")
                            else:
                                word.append(num2char(topi.item()))

                        # real_length = di
                        decoder_input = topi.squeeze(1).copy()
                    words.append("".join(word))
                    decoded_scores.append(char_scores)
            return words, decoded_scores, detailed_decoded_scores

    def beam_search_step(self, encoder_context, top_seqs, k):
        all_seqs = []
        for seq in top_seqs:
            seq_score = reduce_mul([_score for _, _score, _, _ in seq])
            if seq[-1][0] == self.config.sequence.char_count - 1:
                all_seqs.append((seq, seq_score, seq[-1][2], True))
                continue
            decoder_hidden = seq[-1][-1][0]
            onehot = np.zeros((1, 1), dtype=np.int32)
            onehot[:, 0] = seq[-1][0]
            decoder_input = Tensor(onehot.tolist())
            decoder_output, decoder_hidden, decoder_attention = self.seq_decoder(
                decoder_input, decoder_hidden, encoder_context
            )
            detailed_char_scores = decoder_output.asnumpy()
            # print(decoder_output.shape)
            scores, candidates = decoder_output.data[:, 1:].top_k(k)
            for i in range(k):
                character_score = scores[:, i]
                character_index = candidates[:, i]
                score = seq_score * character_score.item()
                char_score = seq_score * detailed_char_scores
                rs_seq = seq + [
                    (
                        character_index.item() + 1,
                        character_score.item(),
                        char_score,
                        [decoder_hidden],
                    )
                ]
                done = character_index.item() + 1 == self.config.sequence.char_count - 1
                all_seqs.append((rs_seq, score, char_score, done))
        all_seqs = sorted(all_seqs, key=lambda seq: seq[1], reverse=True)
        topk_seqs = [seq for seq, _, _, _ in all_seqs[:k]]
        all_done = check_all_done(all_seqs[:k])
        return topk_seqs, all_done

    def beam_search(self, encoder_context, decoder_hidden, beam_size=6, max_len=32):
        char_score = np.zeros(self.config.sequence.char_count)
        top_seqs = [[(self.config.sequence.bos_token, 1.0, char_score, [decoder_hidden])]]
        # loop
        for _ in range(max_len):
            top_seqs, all_done = self.beam_search_step(
                encoder_context, top_seqs, beam_size
            )
            if all_done:
                break
        return top_seqs

class BahdanauAttnDecoderRNN(nn.Cell):
    def __init__(
        self,
        hidden_size,
        embed_size,
        output_size,
        n_layers=1,
        dropout_p=0,
        bidirectional=False,
        onehot_size = (8, 32)
    ):
        super(BahdanauAttnDecoderRNN, self).__init__()
        # Define parameters
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        # Define layers
        embed_weight_init = ops.eye(embed_size,embed_size,mindspore.float32)
        self.embedding = nn.Embedding(output_size, embed_size, embedding_table=embed_weight_init)
        # self.dropout = nn.Dropout(dropout_p)
        self.word_linear = nn.Dense(embed_size, hidden_size)
        self.attn = Attn("concat", hidden_size, embed_size, onehot_size[0] + onehot_size[1])
        self.rnn = nn.GRUCell(2 * hidden_size + onehot_size[0] + onehot_size[1], hidden_size)
        self.out = nn.Dense(hidden_size, output_size)

    def construct(self, word_input, last_hidden, encoder_outputs):
        """
        :param word_input:
            word input for current time step, in shape (B)
        :param last_hidden:
            last hidden stat of the decoder, in shape (layers*direction*B, hidden_size)
        :param encoder_outputs:
            encoder outputs in shape (H*W, B, C)
        :return
            decoder output
        """
        # Get the embedding of the current input word (last output word)
        word_embedded_onehot = self.embedding(word_input).view(
            1, word_input.shape(0), -1
        )  # (1,B,embed_size)
        word_embedded = self.word_linear(word_embedded_onehot)  # (1, B, hidden_size)
        attn_weights = self.attn(last_hidden, encoder_outputs)  # (B, 1, H*W)
        context = attn_weights.bmm(
            encoder_outputs.transpose(0, 1)
        )  # (B, 1, H*W) * (B, H*W, C) = (B,1,C)
        context = context.transpose(0, 1)  # (1,B,C)
        # Combine embedded input word and attended context, run through RNN
        # 2 * hidden_size + W + H: 256 + 256 + 32 + 8 = 552
        rnn_input = P.Concat(2)((word_embedded, context))
        last_hidden = last_hidden.view(last_hidden.shape(0), -1)
        rnn_input = rnn_input.view(word_input.shape(0), -1)
        hidden = self.rnn(rnn_input, last_hidden)
        if not self.training:
            output = nn.Softmax(1)(self.out(hidden))
        else:
            output = nn.LogSoftmax(1)(self.out(hidden))
        # Return final output, hidden state
        # print(output.shape)
        return output, hidden, attn_weights

class Attn(nn.Cell):
    def __init__(self, method, hidden_size, embed_size, onehot_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.attn = nn.Dense(2 * self.hidden_size + onehot_size, hidden_size)
        # self.attn = nn.Linear(hidden_size, hidden_size)
        # self.v = Parameter(np.rand(hidden_size))
        stdv = 1.0 / np.sqrt(Tensor(hidden_size, mindspore.float32))
        self.v = Parameter(Tensor(dtype=mindspore.float32, shape=(hidden_size,),init=Normal(stdv)))
        # stdv = 1.0 / np.sqrt(self.v.shape(0))
        # self.v = mindspore.Parameter(Tensor(hidden_size, init=Normal(sigma=stdv)))

    def construct(self, hidden, encoder_outputs):
        """
        :param hidden:
            previous hidden state of the decoder, in shape (B, hidden_size)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (H*W, B, hidden_size)
        :return
            attention energies in shape (B, H*W)
        """
        max_len = encoder_outputs.shape(0)
        # this_batch_size = encoder_outputs.size(1)
        H = np.tile(hidden, (max_len, 1, 1)).transpose(0, 1)  # (B, H*W, hidden_size)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # (B, H*W, hidden_size)
        attn_energies = self.score(
            H, encoder_outputs
        )  # compute attention score (B, H*W)
        return nn.Softmax(axis=1)(attn_energies).unsqueeze(
            1
        )  # normalize with softmax (B, 1, H*W)

    def score(self, hidden, encoder_outputs):
        energy = ops.tanh(
            self.attn(P.Concat(2)([hidden, encoder_outputs]))
        )  # (B, H*W, 2*hidden_size+H+W)->(B, H*W, hidden_size)
        energy = energy.transpose(2, 1)  # (B, hidden_size, H*W)
        v = np.tile(self.v, (encoder_outputs.data.shape[0], 1)).unsqueeze(1)
        energy = ops.BatchMatMul()(v, energy)  # (B, 1, H*W)
        return energy.squeeze(1)  # (B, H*W)

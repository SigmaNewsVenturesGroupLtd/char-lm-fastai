from fastai import *
from fastai.text import *
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence


class CharTokenizer(Tokenizer):

    def process_all(self, texts):
        return [self.char_tokenize(t) for t in texts]

    def char_tokenize(self, text):
        prefix = ['bos']
        text = text.replace('xxfld 1 ', '')
        return prefix + [x for x in text if ord(x) < 128]


def get_space_preds(txts: list, model: SequentialRNN, bwd_model: SequentialRNN, vocab: Vocab, word_vocab: set):
    sequences = [torch.LongTensor(vocab.numericalize(['bos'] + [x for x in txt] + ['bos'])) for txt in txts ]
    fwd_sequences, lens = get_sequences(sequences)
    bwd_sequences, bwd_lens = get_sequences(sequences, reverse=True)

    forward_preds = predict(model, fwd_sequences, lens, vocab)
    backward_preds = predict(bwd_model, bwd_sequences, bwd_lens, vocab)

    dirty = []
    clean = []
    for seq_idx, (seq,seq_len, fwd_pred, bwd_pred) in enumerate(zip(fwd_sequences.t(), lens, forward_preds, backward_preds)):
        orig_txt = [vocab.itos[x] for x in seq[1:seq_len-1] if x != 1]
        fwd_pred = fwd_pred[:-1]
        bwd_pred = bwd_pred[:-1][::-1]
        for i in range(1, len(orig_txt)):
            if orig_txt[i] != ' ' and fwd_pred[i] == ' ' and bwd_pred[i] == ' ':
                concordance_len = 30
                start = max(0, i - concordance_len)
                end = min(len(orig_txt), i + concordance_len)
                r_num_chars = orig_txt[i:end].index(' ') if ' ' in orig_txt[i:end] else 0
                l_num_chars = orig_txt[start:i][::-1].index(' ') if ' ' in orig_txt[start:i] else 0

                wordlen = l_num_chars + r_num_chars

                if wordlen >= 5 and (r_num_chars >= 3 or l_num_chars >= 3):
                    word = ''.join(orig_txt[i-l_num_chars:i+r_num_chars])
                    l_word = ''.join(orig_txt[i-l_num_chars:i])
                    r_word = ''.join(orig_txt[i:i+r_num_chars])

                    if word not in word_vocab and l_word in word_vocab and r_word in word_vocab:
                        with_split = ''.join(orig_txt[:i]) + ' ' + ''.join(orig_txt[i:])
                        orig_score = score_txt(orig_txt, model, vocab) + score_txt(orig_txt[::-1], bwd_model, vocab)
                        split_score = score_txt(with_split, model, vocab) + score_txt(with_split[::-1], bwd_model, vocab)
                        if split_score > orig_score:
                            dirty.append(''.join(orig_txt[start:end]))
                            clean.append(''.join(orig_txt[start:i]) + ' ' + ''.join(orig_txt[i:end]))
    return pd.DataFrame({'dirty': dirty, 'clean': clean})


def get_sequences(sequences:list, reverse=False):
    """
    Transforms a list of LongTensors to a padded long tensor
    :param sequences:
    :param reverse:
    :return: a tuple res, lens where res is of shape TxB and lens is the lengths of the items in the batch.
    """
    if reverse:
        sequences = [torch.LongTensor(s.numpy()[::-1].copy()) for s in sequences]
    sorted_sequences = sorted(sequences, key=lambda x: x.size(), reverse=True)
    packed_sequences = pack_sequence(sorted_sequences)
    return pad_packed_sequence(packed_sequences, padding_value=1)


def score_txt(txt:str, model:nn.Module, vocab: Vocab):
    """
    Computes the average log likelihood of the text under the model
    :param txt:
    :param model:
    :param vocab:
    :return:
    """
    numericalized = vocab.numericalize(['bos'] + [x for x in txt] + ['bos'])
    model.reset()
    model.eval()
    inp = torch.LongTensor([numericalized]).t()
    preds = F.log_softmax(model(inp)[0], dim=0)
    score = 0.
    for pred, actual in zip(preds, numericalized[1:]):
        score += pred[actual]
    return score / len(txt)


def predict(model: nn.Module, txt:torch.LongTensor, lens:np.array, vocab: Vocab):
    """
    Applies the model and returns the results as a string
    :param model:
    :param txt:
    :param lens:
    :param vocab:
    :return:
    """
    model.eval()
    model.reset()
    forward_preds = model(txt)[0]
    forward_preds = forward_preds.argmax(-1).view(txt.size(0), -1).t()
    res = []
    for preds, length in zip(forward_preds, lens):
        res.append(''.join(vocab.itos[preds[i]] for i in range(length)))
    return res


if __name__ == '__main__':
    databunch = TextFileList.from_folder('/data/hp/0').label_const(0).split_by_folder().datasets(FilesTextDataset)\
        .tokenize(CharTokenizer())\
        .numericalize()\
        .databunch(TextLMDataBunch)
    databunch.save('tmp_char_lm')


def indexes_to_one_hot(indexes, n_dims=None):
    """Converts a vector of indexes to a batch of one-hot vectors. """
    indexes = indexes.type(torch.int64).contiguous().view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(indexes)) + 1
    one_hots = torch.zeros(indexes.size()[0], n_dims).cuda().scatter_(1, indexes, 1)
    one_hots = one_hots.view(*indexes.shape, -1)
    return one_hots


class CharRNNCore(nn.Module):
    "AWD-LSTM/QRNN inspired by https://arxiv.org/abs/1708.02182."

    initrange=0.1

    def __init__(self, vocab_sz:int, emb_sz:int, n_hid:int, n_layers:int, pad_token:int, bidir:bool=False,
                 hidden_p:float=0.2, input_p:float=0.6, embed_p:float=0.1, weight_p:float=0.5):

        super().__init__()
        self.bs,self.ndir = 1,(2 if bidir else 1)
        self.emb_sz,self.n_hid,self.n_layers = emb_sz,n_hid,n_layers
        self.encoder = nn.Embedding(vocab_sz, emb_sz, padding_idx=pad_token)
        self.encoder_dp = EmbeddingDropout(self.encoder, embed_p)
        self.rnns = [nn.LSTM(emb_sz if l == 0 else n_hid, n_hid // self.ndir,
            1, bidirectional=bidir) for l in range(n_layers)]
        self.rnns = [WeightDropout(rnn, weight_p) for rnn in self.rnns]
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.encoder.weight.data.uniform_(-self.initrange, self.initrange)
        self.input_dp = RNNDropout(input_p)
        self.hidden_dps = nn.ModuleList([RNNDropout(hidden_p) for l in range(n_layers)])

    def forward(self, input:LongTensor)->Tuple[Tensor,Tensor]:
        sl,bs = input.size()
        if bs!=self.bs:
            self.bs=bs
            self.reset()
        raw_output = self.input_dp(self.encoder_dp(input))
        new_hidden,raw_outputs,outputs = [],[],[]
        for l, (rnn,hid_dp) in enumerate(zip(self.rnns, self.hidden_dps)):
            raw_output, new_h = rnn(raw_output, self.hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.n_layers - 1: raw_output = hid_dp(raw_output)
            outputs.append(raw_output)
        self.hidden = to_detach(new_hidden)
        return raw_outputs, outputs

    def _one_hidden(self, l:int)->Tensor:
        "Return one hidden state."
        nh = self.n_hid//self.ndir
        return self.weights.new(self.ndir, self.bs, nh).zero_()

    def reset(self):
        "Reset the hidden states."
        [r.reset() for r in self.rnns if hasattr(r, 'reset')]
        self.weights = next(self.parameters()).data
        self.hidden = [(self._one_hidden(l), self._one_hidden(l)) for l in range(self.n_layers)]

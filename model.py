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


def get_space_preds(txts: list, model: SequentialRNN, bwd_model: SequentialRNN, vocab:Vocab, word_vocab:set):
    sequences = [ torch.LongTensor(vocab.numericalize(['bos'] + [x for x in txt] + ['bos'])) for txt in txts ]
    sorted_sequences = sorted(sequences, key=lambda x: x.size(), reverse=True)
    packed_sequences = pack_sequence(sorted_sequences)
    padded_sequences, lens = pad_packed_sequence(packed_sequences)
    forward_preds = predict(model, padded_sequences, lens, vocab)



    reversed_sorted_sequences = sorted([torch.LongTensor(s.numpy()[::-1].copy()) for s in sequences], key=lambda x:x.size(), reverse=True)
    r_packed_sequences = pack_sequence(reversed_sorted_sequences)
    r_padded_sequences, r_lens = pad_packed_sequence(r_packed_sequences)
    backward_preds = predict(bwd_model, r_padded_sequences, r_lens, vocab)

    for seq_idx, seq in enumerate(sorted_sequences):
        orig_txt = [vocab.itos[x] for x in seq[1:-1]]
        fwd_pred = forward_preds[seq_idx][:-2]
        bwd_pred = backward_preds[seq_idx][:-2][::-1]
        for i in range(1, len(orig_txt)):
            if orig_txt[i] != ' ' and fwd_pred[i] == ' ' and bwd_pred[i-1] == ' ':
                concordance_len = 20
                start = max(0, i - concordance_len)
                end = min(len(orig_txt), i + concordance_len)
                r_num_chars = orig_txt[i:end].index(' ') if ' ' in orig_txt[i:end] else 0
                l_num_chars = orig_txt[start:i][::-1].index(' ') if ' ' in orig_txt[start:i] else 0

                wordlen = l_num_chars + r_num_chars

                if wordlen > 6 and r_num_chars >= 3 and l_num_chars >= 3:
                    word = ''.join(orig_txt[i-l_num_chars:i+r_num_chars])
                    l_word = ''.join(orig_txt[i-l_num_chars:i])
                    r_word = ''.join(orig_txt[i:i+r_num_chars])

                    if word not in word_vocab and l_word in word_vocab and r_word in word_vocab:
                        with_split = ''.join(orig_txt[:i]) + ' ' + ''.join(orig_txt[i:])
                        orig_score = score_txt(orig_txt, model, vocab) + score_txt(orig_txt[::-1], bwd_model, vocab)
                        split_score = score_txt(with_split, model, vocab) + score_txt(with_split[::-1], bwd_model, vocab)
                        if split_score > orig_score:
                            print("splitting", ''.join(orig_txt[start:end]), '---->', ''.join(orig_txt[start:i]) + ' ' + ''.join(orig_txt[i:end]))
                        else:
                            pass
                        #print("score too low", orig_txt[start:end], 'to', orig_txt[start:i] + ' ' + orig_txt[i:end])


def score_txt(txt, model, vocab):
    numericalized = vocab.numericalize(['bos'] + [x for x in txt] + ['bos'])
    model.reset()
    model.eval()
    inp = torch.LongTensor([numericalized]).t().cpu()
    preds = F.log_softmax(model(inp)[0], dim=0)
    score = 0.
    for pred, actual in zip(preds, numericalized[1:]):
        score += pred[actual]
    return score / len(txt)

def predict(model, txt, lens, vocab):
    model.eval()
    model.reset()
    forward_preds = model(txt)[0]
    forward_preds = forward_preds.argmax(-1).view(txt.size(0), -1).t()
    res = []
    for row_idx in range(txt.size(1)):
        cur_len = lens[row_idx]
        cur_preds = forward_preds[row_idx]
        res.append(''.join(vocab.itos[cur_preds[i]] for i in range(cur_len)))
    return res


if __name__ == '__main__':
    databunch = TextFileList.from_folder('/data/hp/0').label_const(0).split_by_folder().datasets(FilesTextDataset)\
        .tokenize(CharTokenizer())\
        .numericalize()\
        .databunch(TextLMDataBunch)
    databunch.save('tmp_char_lm')

class CharRNNCore(nn.Module):
    "AWD-LSTM/QRNN inspired by https://arxiv.org/abs/1708.02182."

    initrange=0.1

    def __init__(self, vocab_sz:int, emb_sz:int, n_hid:int, n_layers:int, bidir:bool=False,
                 hidden_p:float=0.2, input_p:float=0.6, weight_p:float=0.5):

        super().__init__()
        self.bs,self.ndir = 1, (2 if bidir else 1)
        self.n_hid,self.n_layers = n_hid, n_layers
        self.rnns = [nn.LSTM(vocab_sz if l == 0 else n_hid, (n_hid if l != n_layers - 1 else emb_sz)//self.ndir,
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
        raw_output = input
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
        nh = (self.n_hid if l != self.n_layers - 1 else self.emb_sz)//self.ndir
        return self.weights.new(self.ndir, self.bs, nh).zero_()

    def reset(self):
        "Reset the hidden states."
        [r.reset() for r in self.rnns if hasattr(r, 'reset')]
        self.weights = next(self.parameters()).data
        if self.qrnn: self.hidden = [self._one_hidden(l) for l in range(self.n_layers)]
        else: self.hidden = [(self._one_hidden(l), self._one_hidden(l)) for l in range(self.n_layers)]

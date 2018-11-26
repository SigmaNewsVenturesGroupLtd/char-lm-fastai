from fastai import *
from fastai.text import *



class CharTokenizer(Tokenizer):

    def process_all(self, texts):
        return [self.char_tokenize(t) for t in texts]

    def char_tokenize(self, text):
        prefix = ['bos']
        text = text.replace('xxfld 1 ', '')
        return prefix + [x for x in text if ord(x) < 128]


def get_space_preds(orig_txt: str, model: SequentialRNN, bwd_model: SequentialRNN, vocab:Vocab, word_vocab:set):
    orig_txt = ['bos'] + [x for x in orig_txt] + ['bos']
    txt = vocab.numericalize(orig_txt)
    bwd_txt = list(reversed(txt))
    forward_preds = predict(model, txt, vocab)[0:-2]
    backward_preds = ''.join(x for x in predict(bwd_model, bwd_txt, vocab)[::-1])[2:]
    orig_txt = ''.join(orig_txt[1:-1])
    for i in range(1, len(orig_txt)):
        if orig_txt[i] != ' ' and forward_preds[i] == ' ' and backward_preds[i-1] == ' ':
            concordance_len = 20
            start = max(0, i - concordance_len)
            end = min(len(orig_txt), i + concordance_len)
            r_num_chars = orig_txt[i:end].find(' ')
            l_index = orig_txt[start:i-1].rfind(' ')
            l_num_chars = len(orig_txt[start:i - 1]) - l_index if l_index > 0 else 0

            wordlen = l_num_chars + r_num_chars

            if wordlen > 6 and r_num_chars >= 3 and l_num_chars >= 3:
                word = orig_txt[i-l_num_chars:i+r_num_chars]
                l_word = orig_txt[i-l_num_chars:i]
                r_word = orig_txt[i:i+r_num_chars]
                if word not in word_vocab and l_word in word_vocab and r_word in word_vocab:
                    with_split = orig_txt[:i] + ' ' + orig_txt[i:]
                    orig_score = score_txt(orig_txt, model, vocab) + score_txt(orig_txt[::-1], bwd_model, vocab)
                    split_score = score_txt(with_split, model, vocab) + score_txt(with_split[::-1], bwd_model, vocab)
                    if split_score > orig_score:
                        print("splitting", orig_txt[start:end], 'to', orig_txt[start:i] + ' ' + orig_txt[i:end])
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

def predict(model, txt, vocab):
    model.eval()
    model.reset()
    inp = torch.LongTensor([txt]).t().cpu()
    forward_preds = model(inp)[0]
    forward_preds = forward_preds.argmax(-1).view(inp.size(0), -1)
    forward_preds = ''.join(vocab.itos[x] for x in forward_preds[:, 0])
    return forward_preds


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

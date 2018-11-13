from fastai import *
from fastai.text import *


class CharTokenizer(Tokenizer):

    def process_all(self, texts):
        return [self.char_tokenize(t) for t in texts]

    def char_tokenize(self, text):
        prefix = ['bos']
        text = text.replace('xxfld 1 ', '')
        return prefix + [x for x in text if ord(x) < 128]


def get_space_preds(orig_txt: str, model: SequentialRNN, vocab:Vocab):
    orig_txt = ['bos'] + [x for x in orig_txt]
    txt = vocab.numericalize(orig_txt)
    model.eval()
    model.reset()
    inp = torch.LongTensor([txt, txt]).t().cuda()
    print(inp.size())
    forward_preds = model(inp)[0]
    print("PREDS", forward_preds.size())
    forward_preds = forward_preds.argmax(-1).view(inp.size(0), -1)
    print("ARGMAX", forward_preds.size())
    forward_preds = ''.join(vocab.itos[x] for x in forward_preds[:, 0])[:-1]
    orig_txt = ''.join(orig_txt[1:])
    print(orig_txt)
    print(forward_preds)
    for i in range(0, len(orig_txt)):
        if txt[i] != ' ' and forward_preds[i] == ' ':
            start = max(0, i-10)
            end = min(len(orig_txt), i+10)
            print("ACTUAL", orig_txt[start:end])
            print("PREDICTED", forward_preds[start:i-1] + '@ @' + forward_preds[i:end])

if __name__ == '__main__':
    databunch = TextLMDataBunch.from_csv('/data/char-lm-fastai',
                                         tokenizer=CharTokenizer(),
                                         chunksize=1024,
                                         max_vocab=128,
                                         clear_cache=True)


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

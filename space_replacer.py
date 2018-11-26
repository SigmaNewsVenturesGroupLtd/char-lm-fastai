from fastai import *
from fastai.text import *
from model import get_space_preds
import json


vocab = Vocab(np.load('/data/char-lm-fastai/tmp/itos.pkl'))
word_vocab = set(np.load('/data/suzi/v5/id2w.pkl'))

train_ids = np.zeros((512, 120))
valid_ids = np.zeros((512, 120))
databunch = TextLMDataBunch.from_ids('/data/char-lm-fastai/', vocab=vocab, train_ids=train_ids, valid_ids=valid_ids,
                                     bs=512, backwards=False)

fwd_learn = language_model_learner(databunch, emb_sz=100, nh=300, nl=1, drop_mult=0., tie_weights=False)
fwd_learn.load('/data/char-lm-fastai/models/fwd-1.049386')
fwd = fwd_learn.model.cpu()

bwd_learn = language_model_learner(databunch, emb_sz=100, nh=300, nl=1, drop_mult=0., tie_weights=False)
bwd_learn.load('/data/char-lm-fastai/models/bwd-1.044461')
bwd = bwd_learn.model.cpu()

bs=2048

for i, chunk in enumerate(pd.read_json('/data/char-lm-fastai/train.jsonl', lines=True, chunksize=bs)):
    print(f"Processing {i*bs}")
    text = chunk['tokens']
    get_space_preds(text, fwd, bwd, vocab, word_vocab)

# encoding=utf-8
import torchtext
import torch
import numpy as np
import random
import torch.nn as nn

USE_CUDA = torch.cuda.is_available()

# 为了保证实验结果可以复现，我们经常会把各种random seed固定在某一个值
random.seed(53113)
np.random.seed(53113)
torch.manual_seed(53113)

if USE_CUDA:
    torch.cuda.manual_seed(53113)

BATCH_SIZE = 32
EMBEDDING_SIZE = 100
HIDDEN_SIZE = 100
MAX_VOCAB_SIZE = 50000


TEXT = torchtext.data.Field(lower=True)
train, val, test = torchtext.datasets.LanguageModelingDataset.splits(
    path='../../data/text8',
    train='text8.train.txt',
    validation='text8.dev.txt',
    test='text8.test.txt',
    text_field=TEXT
)

TEXT.build_vocab(train, max_size=MAX_VOCAB_SIZE)
print(TEXT.vocab.itos[:10])
print(TEXT.vocab.stoi['apple'])


device = torch.device('cuda' if USE_CUDA else 'cpu')

train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits(
    (train, val, test),
    batch_size=BATCH_SIZE,
    device=device,
    bptt_len=50,
    repeat=False,
    shuffle=True
)
it = iter(train_iter)
batch = next(it)
print(batch)
print(batch.text)
print(' '.join(TEXT.vocab.itos[i] for i in batch.text[:, 0].data.cpu()))
print()
print(' '.join(TEXT.vocab.itos[i] for i in batch.target[:, 0].data.cpu()))


for i in range(5):
    batch = next(it)
    print(' '.join(TEXT.vocab.itos[i] for i in batch.text[:, 0].data.cpu()))
    print()
    print(' '.join(TEXT.vocab.itos[i] for i in batch.target[:, 0].data.cpu()))


class RNNModel(nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_size):
        super(RNNModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size

    def forward(self, text, hidden):
        # text: seq_length * batch_size
        emb = self.embed(text) # seq_length * batch_size * embed_size
        output, hidden = self.lstm(emb, hidden)
        # output: seq_length * batch_size * hidden_size
        # hidden: (1 * batch_size * hidden_size, 1 * batch_size * hidden_size)
        output = output.view(-1, output.shape[2]) # (seq_length * batch_size) * hidden_size
        out_vocab = self.linear(output) # (seq_length * batch_size) * vocab_size
        out_vocab = out_vocab.view(output.size(0), output.size(1), out_vocab.size(-1))
        return out_vocab

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros((1, bsz, self.hidden_size), requires_grad=True),
                weight.new_zeros((1, bsz, self.hidden_size), requires_grad=True))


model = RNNModel(vocab_size=len(TEXT.vocab), embed_size=EMBEDDING_SIZE, hidden_size=HIDDEN_SIZE)
if USE_CUDA:
    model = model.to(device)
print(model)
print(next(model.parameters()))


def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


loss_fn = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)
VOCAB_SIZE = len(TEXT.vocab)
GRAD_CLIP = 5.0
NUM_EPOCHS = 2
val_losses = []


def evaluate(model, data):
    model.eval()
    total_loss = 0.
    total_count = 0.
    it = iter(data)
    with torch.no_grad():
        hidden = model.init_hidden(BATCH_SIZE, requires_grad=False)
        for i, batch in enumerate(it):
            data, target = batch.text, batch.target
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)  # backpropgate through all iteration
            loss = loss_fn(output.view(-1, VOCAB_SIZE), target.view(-1))  # batch_size * target_dim, batch_size
            total_loss = loss.item() * np.multiply(*data.size())
            total_count = np.multiply(*data.size())
    loss = total_loss / total_count
    model.train()
    return loss


for epoch in range(NUM_EPOCHS):
    model.train()
    it = iter(train_iter)
    hidden = model.init_hidden(BATCH_SIZE)
    for i, batch in enumerate(it):
        data, target = batch.text, batch.target
        hidden = repackage_hidden(hidden)
        output, hidden = model(data, hidden)  # backpropgate through all iteration
        loss = loss_fn(output.view(-1, VOCAB_SIZE), target.view(-1))  # batch_size * target_dim, batch_size
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), GRAD_CLIP)
        optimizer.step()
        if i % 100 == 0:
            print('loss', loss.item())
        if i % 1000 == 0:
            val_loss = evaluate(model, val_iter)
            if len(val_loss) == 0 or val_loss < min(val_losses):
                torch.save(model.state_dict(), 'lm.pth')
                print('best model saved to lm.pth')
            else:
                # learning rate decay
                scheduler.step()


best_model = RNNModel(vocab_size=len(TEXT.vocab), embed_size=EMBEDDING_SIZE, hidden_size=HIDDEN_SIZE)
if USE_CUDA:
    model = model.to(device)
best_model.load_state_dict(torch.load('lm.pth'))

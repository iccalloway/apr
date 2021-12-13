import pickle
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from transformers import HubertModel, Wav2Vec2Processor


class ASRDataset(Dataset):
    def __init__(self):
        super().__init__()
        with open("./input.pkl", "rb") as f:
            self.in_ = pickle.load(f)

        with open("./output.pkl", "rb") as g:
            self.out_ = pickle.load(g).type(torch.LongTensor)

    def __len__(self):
        return self.in_.shape[0]

    def __getitem__(self, i):
        #return self.in_[i, :], self.out_[i, :]
        return self.in_[i % 10, :], self.out_[i % 10, :]


class ASR(nn.Module):
    def __init__(self):
        super(ASR, self).__init__()
        with open("./mapping.pkl", "rb") as h:
            self.d = pickle.load(h)
            self.rev_d = {v: k for k, v in self.d.items()}

        self.processor = Wav2Vec2Processor.from_pretrained(
            "facebook/hubert-large-ls960-ft"
        )
        self.hubert = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
        ##Freeze Hubert Weights
        for param in self.hubert.parameters():
            param.requires_grad = False
        self.linear1 = nn.Linear(249, 100)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(1024, len(self.d))

    def forward(self, x):
        input_ = self.processor(x.numpy(), sampling_rate=16000, return_tensors="pt")
        audio = torch.squeeze(input_["input_values"])
        attentions = torch.ones(audio.shape)
        hidden_states = self.hubert(audio, attentions).last_hidden_state
        hidden_states = torch.transpose(hidden_states, 1, 2)
        out1 = self.relu(self.linear1(hidden_states))
        out1 = torch.transpose(out1, 1, 2)
        output = self.linear2(out1)
        return output

    def parse(self, x):
        with torch.no_grad():
            maxes = torch.argmax(x, dim=2)
            preds = []
            for a in range(maxes.shape[0]):
                preds.append([self.rev_d[b] for b in maxes[a, :].tolist()])
            return preds


if __name__ == "__main__":
    epochs = 10
    split = 0.8

    ds = ASRDataset()
    asr = ASR()
    train, test = torch.utils.data.random_split(
        ds, [round(len(ds) * split), round(len(ds) * (1 - split))]
    )
    train_loader = DataLoader(train, batch_size=2, shuffle=True, drop_last=True)
    test_loader = DataLoader(test, batch_size=8, shuffle=True, drop_last=True)

    optimizer = torch.optim.Adam(asr.parameters(), lr=1e-4)

    for _ in range(epochs):
        for i, (in_, out_) in enumerate(train_loader):
            loss_fn = nn.CrossEntropyLoss()
            output = asr(in_)
            seg_pred = asr.parse(output)
            seg_actual = [
                [asr.rev_d[b.item()] for b in out_[a, :]] for a in range(out_.shape[0])
            ]
            for i, pred in enumerate(seg_pred):
                print("=====\n{}\n\n{}\n=====\n\n\n\n\n".format(pred, seg_actual[i]))
            output = torch.transpose(output, 1, 2)
            loss = loss_fn(output, out_)
            print(loss)
            loss.backward()
            optimizer.step()

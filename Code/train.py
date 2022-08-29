import json
import numpy as np
import pickle
import torch
import torch.nn as nn
import itertools

from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import HubertModel, Wav2Vec2Processor


def mean(l):
    return sum(l) / len(l)


def elcomp(*lists):
    return [1 if len(set(c)) == 1 else 0 for c in zip(*lists)]


class ASRDataset(Dataset):
    def __init__(self):
        super().__init__()
        with open("./data/output.tsv", "r") as g:
            self.out_ = [json.loads(line) for line in g.readlines()]
            joined = list(itertools.chain.from_iterable(self.out_))
            self.uniques = sorted(list(set(joined)))
            self.weights = torch.tensor(compute_class_weight(
                class_weight="balanced", classes=self.uniques, y=joined
            ))
        self.d = dict(zip(self.uniques, list(range(len(self.uniques)))))
        self.rev_d = {v: k for k, v in self.d.items()}

        self.in_ = np.memmap(
            "./data/input.bin",
            mode="r",
            dtype="float64",
            shape=(len(self.out_), 16000 * 5),
        )

    def __len__(self):
        return self.in_.shape[0]

    def __getitem__(self, i):
        return torch.tensor(self.in_[i, :]), torch.tensor([self.d[a] for a in self.out_[i]])


class ASR(nn.Module):
    def __init__(self, device, uniques, parameterization=None):
        super(ASR, self).__init__()

        self.device = device
        model_name = "facebook/hubert-large-ls960-ft"
        self.sr = 16000

        self.d = dict(zip(uniques, list(range(len(uniques)))))
        self.rev_d = {v: k for k, v in self.d.items()}

        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.hubert = HubertModel.from_pretrained(model_name)

        ##Freeze Hubert Weights
        # for param in self.hubert.parameters():
        #    param.requires_grad = False

        self.linear1 = nn.Linear(249, 100)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(1024, len(self.d))

    def forward(self, x):
        x = x.numpy()[0, :]
        input_ = self.processor(
            x, sampling_rate=self.sr, return_tensors="pt"
        ).input_values.to(self.device)
        hidden_states = self.hubert(input_).last_hidden_state
        hidden_states = torch.transpose(hidden_states, 1, 2)
        out = self.relu(self.linear1(hidden_states))
        out = torch.transpose(out, 1, 2)
        out = self.linear2(out)
        return out

    def encode(self, x):
        return torch.tensor([self.d[item] for item in x])

    def decode(self, x):
        with torch.no_grad():
            maxes = torch.argmax(x, dim=2)
            preds = []
            for a in range(maxes.shape[0]):
                preds.append([self.rev_d[b] for b in maxes[a, :].tolist()])
            return preds


def train(model, loss_fn, optimizer, loader, bs):
    metrics = {"accuracy": []}
    losses = []

    model.train()
    for i, (in_, actual_out) in enumerate(loader):
        pred_out = model(in_)
        acutal_out = actual_out.squeeze().to(model.device)
        loss = loss_fn(pred_out.transpose(1, 2).double(), actual_out.to(model.device))
        losses.append(loss.item())
        loss.backward()

        ##Metric Accuracy
        seg_pred = model.parse(pred_out)
        seg_actual = [
            [model.rev_d[b.item()] for b in actual_out[a, :]]
            for a in range(actual_out.shape[0])
        ]

        matches = [elcomp(seg_pred[k], seg_actual[k]) for k in range(len(seg_pred))]
        metrics["accuracy"].extend([sum(k) / len(k) for k in matches])
        # print("=====\n{}\n\n{}\n=====\n\n\n\n\n".format(pred, seg_actual[k]))
        if (i + 1) % bs == 0:
            optimizer.step()

    return mean(losses), {k: mean(v) for k, v in metrics.items()}, model


def test(model, loss_fn, loader):
    metrics = {"accuracy": []}
    losses = []

    model.eval()
    with torch.no_grad():
        for i, (in_, actual_out) in enumerate(loader):
            pred_out = model(in_)
            acutal_out = actual_out.squeeze().to(model.device)
            loss = loss_fn(pred_out.transpose(1, 2).double(), actual_out.to(model.device))
            losses.append(loss.item())

            ##Metric Accuracy
            seg_pred = model.decode(pred_out)
            seg_actual = [
                [model.rev_d[b.item()] for b in actual_out[a, :]]
                for a in range(actual_out.shape[0])
            ]

            matches = [elcomp(seg_pred[k], seg_actual[k]) for k in range(len(seg_pred))]
            metrics["accuracy"].extend([sum(k) / len(k) for k in matches])
    return mean(losses), {k: mean(v) for k, v in metrics.items()}, model


if __name__ == "__main__":
    writer = SummaryWriter("logs/debugging/")
    epochs = 30
    split = 0.8
    lr = 1e-4
    bs = 40
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    data = ASRDataset()
    print(data[0])
    model = ASR(device, data.uniques).to(device)

    train_data, test_data = torch.utils.data.random_split(
        data, [round(len(data) * split), round(len(data) * (1 - split))]
    )
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True, drop_last=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=data.weights)

    print("Baseline")
    val_loss, val_metrics, model = test(model, loss_fn, test_loader)
    for k, v in val_metrics.items():
        writer.add_scalar("{}/val".format(k), v, -1)
    writer.add_scalar("loss/val", val_loss, -1)

    for epoch in range(epochs):
        print("Starting Epoch {}".format(epoch))
        print("Training")
        train_loss, train_metrics, model = train(
            model, loss_fn, optimizer, train_loader, bs
        )
        print(train_loss, train_metrics)
        for k, v in train_metrics.items():
            writer.add_scalar("{}/train".format(k), v, epoch)
        writer.add_scalar("loss/train", train_loss, epoch)

        print("Validation")
        val_loss, val_metrics, model = test(model, loss_fn, test_loader)
        print(val_loss, val_metrics)
        for k, v in val_metrics.items():
            writer.add_scalar("{}/val".format(k), v, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)

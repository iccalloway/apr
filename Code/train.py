import json
import numpy as np
import pickle
import torch
import torch.nn as nn
import itertools

from ax import optimize
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import HubertModel, Wav2Vec2Processor, AutoModel


def mean(l):
    return sum(l) / len(l)


def elcomp(*lists):
    return [1 if len(set(c)) == 1 else 0 for c in zip(*lists)]


best = 0


class ASRDataset(Dataset):
    def __init__(self, in_path, out_path):
        super().__init__()
        with open(out_path, "r") as g:
            self.out_ = [json.loads(line) for line in g.readlines()]
            joined = list(itertools.chain.from_iterable(self.out_))
            self.uniques = sorted(list(set(joined)))
            self.weights = torch.tensor(
                compute_class_weight(
                    class_weight="balanced", classes=self.uniques, y=joined
                )
            )
        self.d = dict(zip(self.uniques, list(range(len(self.uniques)))))
        self.rev_d = {v: k for k, v in self.d.items()}

        self.in_ = np.memmap(
            in_path,
            mode="r",
            dtype="float64",
            shape=(len(self.out_), 16000 * 5),
        )

    def __len__(self):
        return self.in_.shape[0]

    def __getitem__(self, i):
        return torch.tensor(self.in_[i, :]), torch.tensor(
            [self.d[a] for a in self.out_[i]]
        )


class ASR(nn.Module):
    def __init__(self, device, uniques, parameterization=None):
        super(ASR, self).__init__()

        self.device = device
        model_name = "facebook/hubert-large-ll60k"
        self.sr = 16000

        self.d = dict(zip(uniques, list(range(len(uniques)))))
        self.rev_d = {v: k for k, v in self.d.items()}

        self.processor = Wav2Vec2Processor.from_pretrained(
            "facebook/hubert-large-ls960-ft"
        )
        self.hubert = AutoModel.from_pretrained(model_name)

        self.linear1 = nn.Linear(1024, 200)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(200, len(self.d))
        self.linear3 = nn.Linear(249, 100)

    def forward(self, x):
        input_ = self.processor(
            x.numpy(), sampling_rate=self.sr, return_tensors="pt"
        ).input_values.to(self.device)
        out = self.hubert(torch.squeeze(input_)).last_hidden_state
        out = self.relu(self.linear1(out))
        out = self.linear2(out)
        out = torch.transpose(out, 1, 2)
        out = self.linear3(out)
        out = torch.transpose(out, 1, 2)
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

    def collate_fn(self, data):
        i = []
        o = []
        for a, b in data:
            i.append(a)
            o.append(b)
        return torch.stack(i, dim=0), torch.stack(o, dim=0)


def train(model, loss_fn, optimizer, loader, bs, mode=None):
    if mode not in ["train", "test"]:
        raise ValueError('Mode must be set to "train" or "test"')

    if mode == "train":
        model.train()
    elif mode == "test":
        model.eval()

    metrics = {}
    losses = []

    for i, (in_, actual_out) in enumerate(loader):
        try:
            pred_out = model(in_)
            pred_temp = torch.transpose(pred_out, 1, 2).to(model.device)
            actual_out = actual_out.to(model.device)
            loss = loss_fn(pred_temp, actual_out)
            losses.append(loss.item())
            if mode == "train":
                loss.backward()
                if (i + 1) % bs == 0:
                    optimizer.step()
            del loss
        except:
            raise
            torch.cuda.empty_cache()
            continue

        ##Metric Accuracy
        seg_pred = model.decode(pred_out.cpu())
        seg_actual = [
            [model.rev_d[b.item()] for b in actual_out[a, :]]
            for a in range(actual_out.shape[0])
        ]

        matches = [elcomp(seg_pred[k], seg_actual[k]) for k in range(len(seg_pred))]
        if "accuracy" in metrics:
            metrics["accuracy"].extend([sum(k) / len(k) for k in matches])
        else:
            metrics["accuracy"] = [sum(k) / len(k) for k in matches]

    return mean(losses), {k: mean(v) for k, v in metrics.items()}, model


def train_evaluate(parameterization):
    in_path = "/outside/data/input.bin"
    out_path = "/outside/data/output.tsv"
    params = "stt-" + ",".join(
        ["{}={}".format(k, v) for k, v in parameterization.items()]
    )
    writer = SummaryWriter("logs/{}".format(params))
    epochs = 30
    split = 0.8
    lr = parameterization["lr"]
    bs = parameterization["bs"]
    best_metric = "accuracy"

    global best

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    data = ASRDataset(in_path, out_path)
    print(data[0])
    model = ASR(device, data.uniques).to(device)

    ##Layer Freezing
    prefix = "encoder.layers."
    turn_off_layers = list(
        range(parameterization["freeze"] - 1)
    )  ##Layers that should be frozen
    layer_check = [prefix + str(a) for a in turn_off_layers]
    for name, param in model.hubert.named_parameters():
        if any([a in name for a in layer_check]):
            print(name, param.grad)
            param.requires_grad = False

    train_data, test_data = torch.utils.data.random_split(
        data, [round(len(data) * split), round(len(data) * (1 - split))]
    )
    print('Train Data has {} elements'.format(len(train_data)))
    print('Test Data has {} elements'.format(len(test_data)))
    train_loader = DataLoader(
        train_data,
        batch_size=2,
        shuffle=True,
        drop_last=True,
        collate_fn=model.collate_fn,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=4,
        shuffle=True,
        drop_last=True,
        collate_fn=model.collate_fn,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=data.weights.float().to(model.device))

    print("Baseline")
    val_loss, val_metrics, model = train(
        model, loss_fn, None, test_loader, None, "test"
    )
    for k, v in val_metrics.items():
        writer.add_scalar("{}/val".format(k), v, -1)
    writer.add_scalar("loss/val", val_loss, -1)

    best = val_metrics[best_metric]
    for epoch in range(epochs):
        print("Starting Epoch {}".format(epoch))
        print("Training")
        train_loss, train_metrics, model = train(
            model, loss_fn, optimizer, train_loader, bs, "train"
        )
        print(train_loss, train_metrics)
        for k, v in train_metrics.items():
            writer.add_scalar("{}/train".format(k), v, epoch)
        writer.add_scalar("loss/train", train_loss, epoch)

        print("Validation")
        val_loss, val_metrics, model = test(model, loss_fn, None, test_loader, "test")
        print(val_loss, val_metrics)
        for k, v in val_metrics.items():
            writer.add_scalar("{}/val".format(k), v, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)

        if val_metrics[best_metric] > best:
            best = val_metrics[best_metric]
            print("New Best: {}".format(best))
            torch.save(model.state_dict, "./stt.model")


def ax_optimize():
    best_parameters, values, experiment, model = optimize(
        parameters=[
            {"name": "lr", "type": "range", "bounds": [1e-7, 1e-2], "log_scale": True},
            {"name": "freeze", "type": "range", "bounds": [0, 23], "log_scale": False},
            {
                "name": "bs",
                "type": "range",
                "bounds": [1, 128],
                "log_scale": True,
            },
        ],
        evaluation_function=train_evaluate,
        objective_name="accuracy",
        minimize=False,
        total_trials=100,
    )
    with open("./results.log", "r") as f:
        f.write(logging.info(best_parameters))
    return best_parameters, values


if __name__ == "__main__":
    ax_optimize()

import os
import random
import torch
import torch.nn as nn

from argparse import ArgumentParser
from transformers import AdamW

from src.parameters import DEVICE, SAVED_CHECKPOINTS

from src.model import ColBERT
from src.utils import print_message, save_checkpoint


def autoopen(filename, mode='rb', **kwargs):
    if filename.endswith(".gz"):
        print(mode)
        import gzip
        return gzip.open(filename, mode, **kwargs)
    elif filename.endswith(".bz2"):
        import bz2
        return bz2.open(filename, mode, **kwargs)
    return open(filename, mode)



class TrainReader:
    def __init__(self, data_file):
        print_message("#> Training with the triples in", data_file, "...\n\n")
        self.reader = autoopen(data_file, mode='rt', encoding="utf-8")

    # def get_minibatch(self, bsize):
    #     return [self.reader.readline().split('\t') for _ in range(bsize)]
    def get_minibatch(self, bsize):
        arr = []
        for _ in  range(bsize):
           parts = self.reader.readline().split('\t')
           if len(parts) == 3:
              arr.append(parts)
        return arr


def manage_checkpoints(colbert, optimizer, batch_idx):
    if batch_idx % 100 == 0:
        save_checkpoint("colbert.dnn", 0, batch_idx, colbert, optimizer)

    if batch_idx in SAVED_CHECKPOINTS:
        save_checkpoint("colbert-" + str(batch_idx) + ".dnn", 0, batch_idx, colbert, optimizer)


def train(args):
    colbert = ColBERT.from_pretrained(args.bert,
                                      query_maxlen=args.query_maxlen,
                                      doc_maxlen=args.doc_maxlen,
                                      dim=args.dim,
                                      similarity_metric=args.similarity, 
                                      tokenizer=args.bert_tokenizer)
    colbert = colbert.to(DEVICE)
    colbert.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(colbert.parameters(), lr=args.lr, eps=1e-8)

    optimizer.zero_grad()
    labels = torch.zeros(args.bsize, dtype=torch.long, device=DEVICE)

    reader = TrainReader(args.triples)
    train_loss = 0.0

    for batch_idx in range(args.maxsteps):
        Batch = reader.get_minibatch(args.bsize)
        Batch = sorted(Batch, key=lambda x: max(len(x[1]), len(x[2])))

        for B_idx in range(args.accumsteps):
            size = args.bsize // args.accumsteps
            B = Batch[B_idx * size: (B_idx+1) * size]
            if len(B) == 0:
                continue
            Q, D1, D2 = zip(*B)

            colbert_out = colbert(Q + Q, D1 + D2)
            colbert_out1, colbert_out2 = colbert_out[:len(Q)], colbert_out[len(Q):]

            out = torch.stack((colbert_out1, colbert_out2), dim=-1)

            positive_score, negative_score = round(colbert_out1.mean().item(), 2), round(colbert_out2.mean().item(), 2)
            print("#>>>   ", positive_score, negative_score, '\t\t|\t\t', positive_score - negative_score)

            loss = criterion(out, labels[:out.size(0)])
            loss = loss / args.accumsteps
            loss.backward()

            train_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(colbert.parameters(), 2.0)

        optimizer.step()
        optimizer.zero_grad()

        print_message(batch_idx, train_loss / (batch_idx+1))

        manage_checkpoints(colbert, optimizer, batch_idx+1)

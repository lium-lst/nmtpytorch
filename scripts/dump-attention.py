#!/usr/bin/env python
import argparse
import pickle as pkl
from pathlib import Path

import numpy as np
import torch
import tqdm

from nmtpytorch.translator import Translator
from nmtpytorch.utils.data import make_dataloader



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='nmtpy-dump-attention',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="generate attention pkl",
        argument_default=argparse.SUPPRESS)

    parser.add_argument('-m', '--model', type=str, required=True,
                        help=".ckpt model file")
    parser.add_argument('-s', '--split', type=str,
                       help='test_set name given as in configuration file')
    parser.add_argument('-o', '--output', type=str,
                       help='output file name.')

    args = parser.parse_args()
    translator = Translator(models=[args.model], splits=args.split,
                            source=None, disable_filters=True, override=None,
                            task_id=None)

    model = translator.instances[0]

    dataset = model.load_data(args.split, 96, mode='beam')
    loader = make_dataloader(dataset)
    data = []

    torch.set_grad_enabled(False)

    # Greedy search
    for batch in tqdm.tqdm(loader, unit='batch'):
        # Visual attention (may not be available)
        img_att = [[] for i in range(batch.size)]

        # Textual attention
        main_att = [[] for i in range(batch.size)]

        # Hierarchical attention
        hie_att = [[] for i in range(batch.size)]

        hyps = [[] for i in range(batch.size)]

        fini = torch.zeros(batch.size, dtype=torch.long)
        ctx_dict = model.encode(batch)

        # Get initial hidden state
        h_t = model.dec.f_init(ctx_dict)

        y_t = model.get_bos(batch.size)

        # Iterate for 100 timesteps
        for t in range(100):
            logp, h_t = model.dec.f_next(ctx_dict, model.dec.get_emb(y_t, t).squeeze(), h_t)

            # text attention
            tatt = model.dec.txt_alpha_t.data.clone().numpy()

            # If decoder has .img_alpha_t
            if hasattr(model.dec, 'img_alpha_t'):
                iatt = model.dec.img_alpha_t.data.clone().numpy()
            else:
                iatt = None

            if hasattr(model.dec, 'h_att'):
                hatt = model.dec.h_att.data.clone().numpy()
            else:
                hatt = None

            top_scores, y_t = logp.data.topk(1, largest=True)
            hyp = y_t.numpy().tolist()
            for idx, w in enumerate(hyp):
                if 2 not in hyps[idx]:
                    hyps[idx].append(w[0])
                    main_att[idx].append(tatt[:, idx])
                    if iatt is None:
                        img_att[idx].append(None)
                    else:
                        img_att[idx].append(iatt[:, idx])

                    if hatt is None:
                        hie_att[idx].append(None)
                    else:
                        hie_att[idx].append(hatt[:, idx])


            # Did we finish? (2 == <eos>)
            fini = fini | y_t.eq(2).squeeze().long()
            if fini.sum() == batch.size:
                break

        for h, sa, ia, ha in zip(hyps, main_att, img_att, hie_att):
            d = [model.trg_vocab.idxs_to_sent(h), sa]
            if ia:
                d.extend(ia)
            if ha:
                d.extend(ha)
            data.append(d)

    # Put into correct order
    data = [data[i] for i, j in sorted(
        enumerate(loader.batch_sampler.orig_idxs), key=lambda k: k[1])]

    src_lines = []
    with open(model.opts.data['{}_set'.format(args.split)][model.sl]) as sf:
        for line in sf:
            src_lines.append(line.strip())

    for d, line in zip(data, src_lines):
        d.insert(0, line)

    with open('{}.greedy.pkl'.format(args.split), 'wb') as f:
        pkl.dump(data, f)

    with open('{}.greedy.txt'.format(args.split), 'w') as f:
        for line in data:
            f.write(line[0] + '\n')



# -*- coding: utf-8 -*-
import logging

import torch
from torch import nn

from ..layers import TextEncoder, MaxMargin, FF
from ..utils.misc import get_n_params
from ..vocabulary import Vocabulary
from ..utils.topology import Topology
from ..utils.ml_metrics import Loss
from ..datasets import MultimodalDataset
from ..metrics import Metric

import numpy as np

logger = logging.getLogger('nmtpytorch')


class Ranking(nn.Module):
    """In the config, this model expects the text as the source and the
    image as the tgt."""
    supports_beam_search = True

    def set_defaults(self):
        self.defaults = {
            'emb_dim': 128,             # Source and target embedding sizes
            'emb_maxnorm': None,        # Normalize embeddings l2 norm to 1
            'emb_gradscale': False,     # Scale embedding gradients w.r.t. batch frequency
            'enc_dim': 256,             # Encoder hidden size
            'enc_type': 'gru',          # Encoder type (gru|lstm)
            'n_encoders': 1,            # Number of stacked encoders
            'image_dim': 2048,          # Decoder hidden size
            'proj_dim': 256,            # Image projection size
            'ranking_dim': 256,         # Expected max-margin ranking size
            'dropout_img': 0,           # Simple dropout to the image embeddings
            'dropout_enc': 0,           # Intra-encoder dropout if n_encoders > 1
            'dropout_emb': 0,           # Intra-encoder dropout if n_encoders > 1
            'max_len': 80,              # Reject sentences where 'bucket_by' length > 80
            'bucket_order': None,       # A key like 'en' to define w.r.t which dataset
            'bucket_by': 'en',          # A key like 'en' to define w.r.t which dataset
                                        # the batches will be sorted
            'pooling_type': 'mean',     # How to pool over the input features
            'sim_function': 'cosine',   # Which similarity function to use for ranking
            'margin': 0.1,              # margin in the max-margin. Must be > 0.
            'direction': None,          # Network directionality, i.e. en->de
            'max_violation': False,     # Max-of-hinges or sum-of-hinges loss?
        }

    def __init__(self, opts):
        super().__init__()

        # opts -> config file sections {.model, .data, .vocabulary, .train}
        self.opts = opts

        # Vocabulary objects
        self.vocabs = {}

        # Each auxiliary loss should be stored inside this dictionary
        # in order to be taken into account by the mainloop for multi-tasking
        self.aux_loss = {}

        # Setup options
        self.opts.model = self.set_model_options(opts.model)

        # Parse topology & languages
        self.topology = Topology(self.opts.model['direction'])

        # Load vocabularies here
        for name, fname in self.opts.vocabulary.items():
            self.vocabs[name] = Vocabulary(fname, name=name)

        # nmtpytorch expects a src -> tgt format.
        # We don't have that so we need to abuse the config.
        # the source NEEDS to be text, the target needs to be the image
        slangs = self.topology.get_src_langs()
        if slangs:
            self.sl = slangs[0]
            self.src_vocab = self.vocabs[self.sl]
            self.n_src_vocab = len(self.src_vocab)

    def __repr__(self):
        s = super().__repr__() + '\n'
        for vocab in self.vocabs.values():
            s += "{}\n".format(vocab)
        s += "{}\n".format(get_n_params(self))
        return s

    def set_model_options(self, model_opts):
        self.set_defaults()
        for opt, value in model_opts.items():
            if opt in self.defaults:
                # Override defaults from config
                self.defaults[opt] = value
            else:
                logger.info('Warning: unused model option: {}'.format(opt))
        return self.defaults

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad and 'bias' not in name:
                nn.init.kaiming_normal(param.data)

    def setup(self, is_train=True):
        """Sets up NN topology by creating the layers."""
        ########################
        # Create Textual Encoder
        ########################
        self.text_enc = TextEncoder(
            input_size=self.opts.model['emb_dim'],
            hidden_size=self.opts.model['enc_dim'],
            n_vocab=self.n_src_vocab,
            rnn_type=self.opts.model['enc_type'],
            dropout_emb=self.opts.model['dropout_emb'],
            dropout_rnn=self.opts.model['dropout_enc'],
            num_layers=self.opts.model['n_encoders'],
            emb_maxnorm=self.opts.model['emb_maxnorm'],
            emb_gradscale=self.opts.model['emb_gradscale'])

        ################
        # Create Image Encoder
        # feed-forward (FF) layer to project feat_dim to emb_dim
        # which is what concat/prepend
        ################
        self.img_enc = FF(self.opts.model['image_dim'], self.opts.model['proj_dim'])

        ############################
        # Create the max-margin loss
        ############################
        self.loss = MaxMargin(margin=self.opts.model['margin'],
                              sim_function=self.opts.model['sim_function'],
                              max_violation=self.opts.model['max_violation'])

    def load_data(self, split, batch_size, mode='train'):
        """Loads the requested dataset split."""
        dataset = MultimodalDataset(
            data=self.opts.data['{}_set'.format(split)],
            mode=mode, batch_size=batch_size,
            vocabs=self.vocabs, topology=self.topology,
            bucket_by=self.opts.model['bucket_by'],
            max_len=self.opts.model['max_len'],
            bucket_order=self.opts.model['bucket_order'])
        logger.info(dataset)
        return dataset

    def encode(self, batch, **kwargs):
        """Encodes all inputs and returns a dictionary.

        Arguments:
            batch (dict): A batch of samples with keys designating the
                information sources.

        Returns:
            dict:
                A dictionary where keys are source modalities compatible
                with the data loader and the values are tuples where the
                elements are encodings and masks. The mask can be ``None``
                if the relevant modality does not require a mask.
        """

        # Pass the projected sentence and image features
        # to the max-margin ranking layer

        return {str(self.sl): self.text_enc(batch[self.sl]),
                'feats':      self.img_enc(batch['feats'])}

    def forward(self, batch, **kwargs):
        """Computes the forward-pass of the network and returns batch loss.

        Arguments:
            batch (dict): A batch of samples with keys designating the source
                and target modalities.

        Returns:
            Variable:
                A scalar loss normalized w.r.t batch size and token counts.
        """
        encoders = self.encode(batch)
        text_data = encoders[str(self.sl)]
        text = text_data[0]
        text_mask = text_data[1]
        if text_mask is not None:
            # pool the text and divide by the correct mask
            text = text.sum(0) / text_mask.sum(0).unsqueeze(1)
        else:
            text = text.mean(0)
        image = encoders['feats']

        # Get loss dict
        result = self.loss(text, image)
        result['n_items'] = torch.nonzero(batch[self.sl][1:]).shape[1]
        return result

    def encode_for_test(self, batch):
        """ Encodes the images and sentences exclusively for measuring the
            image--sentence retrieval performance of a model for evaluation.
            Does not forward or calculate a loss
        """
        encoders = self.encode(batch)
        text_data = encoders[str(self.sl)]
        text = text_data[0]
        text_mask = text_data[1]
        if text_mask is not None:
            # pool the text and divide by the correct mask
            text = text.sum(0) / text_mask.sum(0).unsqueeze(1)
        else:
            text = text.mean(0)
        image = encoders['feats']
        return text, image

    def test_performance(self, data_loader, dump_file=None):
        """Computes test set loss over the given DataLoader instance.

           This function also computes the retrieval performance after
           concatenating all of the batches together. We need to encode
           all of the data in the split at the same time, hence the
           acummulators of the encoder stacks.
        """
        loss = Loss()

        # These stacks save the rank-2 matrices that represent the output
        # of the encoder functions over the two data sources that we are
        # trying to project into the same space.
        enc1_stack = []
        enc2_stack = []

        for batch in data_loader:
            batch.to_gpu(volatile=True)
            out = self.forward(batch)
            loss.update(out['loss'], out['n_items'])

            # Encode the batch data here and append it to the stacks
            text, image = self.encode_for_test(batch)
            enc1_stack.append(text)
            enc2_stack.append(image)

        # Concatenate each stack into a single matrix
        enc1 = torch.cat(enc1_stack)
        enc2 = torch.cat(enc2_stack)

        # Calculcate the Image2Text and T2I retrieval performance
        i2t_scores = self.i2t(enc1, enc2)
        t2i_scores = self.t2i(enc2, enc1)

        return [
            Metric('LOSS', loss.get(), higher_better=False),
            Metric('Image2Text R@1', i2t_scores[0], higher_better=True),
            Metric('Image2Text R@5', i2t_scores[1], higher_better=True),
            Metric('Image2Text R@10', i2t_scores[2], higher_better=True),
            Metric('Image2Text SumR', sum(i2t_scores[0:3]), higher_better=True),
            Metric('Image2Text MedR', i2t_scores[3], higher_better=False),
            Metric('Text2Image R@1', t2i_scores[0], higher_better=True),
            Metric('Text2Image R@5', t2i_scores[1], higher_better=True),
            Metric('Text2Image R@10', t2i_scores[2], higher_better=True),
            Metric('Text2Image SumR', sum(t2i_scores[0:3]), higher_better=True),
            Metric('Text2Image MedR', t2i_scores[3], higher_better=False),
            Metric('SUMRS', sum(i2t_scores[0:3] + t2i_scores[0:3]), higher_better=True)
        ]

    def get_decoder(self, task_id=None):
        """Compatibility function for multi-tasking architectures."""
        return self.dec

    def i2t(self, images, captions, npts=None, measure='cosine', return_ranks=False):
        """ Computes the image--text retrieval performance on the held-out
            data. Expects all of the images and all of the sentences to have
            been encoded. """

        if npts is None:
            npts = images.shape[0]
        index_list = []

        ranks = np.zeros(npts)
        top1 = np.zeros(npts)
        for index in range(npts):

            # Get query image
            im = images[index].view(1, images.shape[1])

            # Compute scores
            d = torch.mm(im, captions.t())
            ordered, inds = torch.sort(d, descending=True)
            index_list.append(inds[0][0])

            # Score
            rank = 1e20
            tmp = (inds == index).nonzero()[0][1].data[0]
            if tmp < rank:
                rank = tmp
            ranks[index] = rank

        # Compute metric
        r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
        medr = np.floor(np.median(ranks)) + 1
        if return_ranks:
            return (r1, r5, r10, medr), (ranks, top1)
        else:
            return (r1, r5, r10, medr)

    def t2i(self, captions, images, npts=None, measure='cosine', return_ranks=False):
        """ Computes the text--image retrieval performance on the held-out
            data. Expects all of the images and all of the sentences to have
            been encoded. """

        if npts is None:
            npts = captions.shape[0]
        index_list = []

        ranks = np.zeros(npts)
        top1 = np.zeros(npts)
        for index in range(npts):

            # Get query sentence
            sent = captions[index].view(1, captions.shape[1])

            # Compute scores
            d = torch.mm(sent, images.t())
            ordered, inds = torch.sort(d, descending=True)
            index_list.append(inds[0][0])

            # Score
            rank = 1e20
            tmp = (inds == index).nonzero()[0][1].data[0]
            if tmp < rank:
                rank = tmp
            ranks[index] = rank

        # Compute metric
        r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
        medr = np.floor(np.median(ranks)) + 1
        if return_ranks:
            return (r1, r5, r10, medr), (ranks, top1)
        else:
            return (r1, r5, r10, medr)

# -*- coding: utf-8 -*-
import logging

import torch
from torch import nn

from ..layers import TextEncoder, ImageEncoder, VectorDecoder
from ..layers import ZSpace
from ..utils.misc import get_n_params
from ..vocabulary import Vocabulary
from ..utils.topology import Topology
from ..utils.ml_metrics import Loss
from ..utils.device import DEVICE
from ..utils.misc import pbar
from ..datasets import MultimodalDataset
from ..metrics import Metric
from ..utils.scheduler import Scheduler

logger = logging.getLogger('nmtpytorch')


class Multitask(nn.Module):
    supports_beam_search = True

    def set_defaults(self):
        self.defaults = {
            # Text related options
            'emb_dim': 128,             # Source and target embedding sizes
            'emb_maxnorm': None,        # Normalize embeddings l2 norm to 1
            'emb_gradscale': False,     # Scale embedding gradients w.r.t. batch frequency
            'enc_dim': 256,             # Encoder hidden size
            'enc_type': 'gru',          # Encoder type (gru|lstm)
            'n_encoders': 1,            # Number of stacked encoders
            'dec_dim': 256,             # Decoder hidden size
            'dec_type': 'gru',          # Decoder type (gru|lstm)
            'dec_init': 'mean_ctx',     # How to initialize decoder (zero/mean_ctx/feats)
            'dec_init_size': None,      # feature vector dimensionality for
                                        # dec_init == 'feats'
            'att_type': 'mlp',          # Attention type (mlp|dot)
            'att_temp': 1.,             # Attention temperature
            'att_activ': 'tanh',        # Attention non-linearity (all torch nonlins)
            'att_mlp_bias': False,      # Enables bias in attention mechanism
            'att_bottleneck': 'ctx',    # Bottleneck dimensionality (ctx|hid)
            'att_transform_ctx': True,  # Transform annotations before attention
            'dropout_emb': 0,           # Simple dropout to source embeddings
            'dropout_ctx': 0,           # Simple dropout to source encodings
            'dropout_out': 0,           # Simple dropout to decoder output
            'dropout_enc': 0,           # Intra-encoder dropout if n_encoders > 1
            'tied_emb': False,          # Share embeddings: (False|2way|3way)
            'direction': None,          # Network directionality, i.e. en->de
            'max_len': 80,              # Reject sentences where 'bucket_by' length > 80
            'bucket_by': None,          # A key like 'en' to define w.r.t which dataset
                                        # the batches will be sorted
            'bucket_order': None,       #
            # Image related options
            'cnn_type': 'resnet50',     # A variant of VGG or ResNet
            'cnn_layer': 'res5c_relu',  # From where to extract features
            'cnn_pretrained': True,     # Should we use pretrained imagenet weights
            'cnn_finetune': None,       # Should we finetune part or all of CNN
            'pool': None,               # ('Avg|Max', kernel_size, stride_size)
            'dropout_img': 0.,          # a 2d dropout over conv features
            'l2_norm': False,           # L2 normalize features
            'l2_norm_dim': -1,          # Which dimension to L2 normalize
            'fusion_type': 'concat',    # Multimodal context fusion (sum|mul|concat)
            'resize': 256,              # resize width, height for images
            'crop': 224,                # center crop size after resize
            # Latent space related options
            'z_type': None,             # whether to do simple combination or projections (None|ff)
            'z_activ': None,            # which transformation: (linear|tanh|sigmoid)
            # Scheduler related options
            'schedule_type_enc': None,  # drop encoder(s) randomly (None|random|random_1)
            'schedule_type_dec': None,  # drop decoder(s) randomly (None|random|random_1)
            'droptask_prob': 1,         # probability of dropping encoder(s)/decoder(s)
                                        # (only used for non-None schedule_type_enc/dec)
            'droptask_e_delay': None,   # number of completed epochs before droptask
            'manual_schedule': None,    # dictionary of {id:direction@num_batches} pairs to cycle thru (None|{})
            'val_tasks': None           # dictionary of {id:direction} pairs for validation (None|{})
        }

    def __init__(self, opts):
        super().__init__()

        # opts -> config file sections {.model, .data, .vocabulary, .train}
        self.opts = opts

        # Langs, Vocabulary and Vocab Length objects
        self.vocabs = {}
        self.slangs = []
        self.svocabs = {}       # source vocabs
        self.n_svocabs = {}
        self.tlangs = []        # target languages IDs
        self.tvocabs = {}       # target vocabs
        self.n_tvocabs = {}
        self.val_refs = {}
        self.ctx_sizes = {}

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

        # Inherently non multi-lingual aware <-- Let's change that!
        slangs = self.topology.get_src_langs()
        tlangs = self.topology.get_trg_langs()
        for sl in slangs:
            self.slangs.append(sl)
            self.svocabs[sl] = self.vocabs[sl]
            self.n_svocabs[sl] = len(self.svocabs[sl])
        for tl in tlangs:
            self.tlangs.append(tl)
            self.tvocabs[tl] = self.vocabs[tl]
            self.n_tvocabs[tl] = len(self.tvocabs[tl])
            # Need to be set for early-stop evaluation
            # NOTE: This should come from config or elsewhere
            self.val_refs[tl] = self.opts.data['val_set'][tl]

        # Textual context size is always equal to enc_dim * 2 since
        # it is the concatenation of forward and backward hidden states
        if 'enc_dim' in self.opts.model:
            for sl in slangs:
                self.ctx_sizes[str(sl)] = self.opts.model['enc_dim'] * 2

        # add the context size for the latent space
        # TODO: add config file entry for this size + change z.py to project to this size
        self.ctx_sizes['z'] = self.opts.model['enc_dim'] * 2

        # Check tying option
        if self.opts.model['tied_emb'] not in [False, '2way']:
            raise RuntimeError(
                "'{}' not recognized for tied_emb.".format(self.opts.model['tied_emb']))

        # TODO: VISION init

        # Latent space init
        self.z_type = self.opts.model['z_type']
        self.z_activ = self.opts.model['z_activ']

        # Scheduler options init
        self.schedule_type_enc = self.opts.model['schedule_type_enc']
        self.schedule_type_dec = self.opts.model['schedule_type_dec']
        self.droptask_prob = self.opts.model['droptask_prob']
        self.droptask_e_delay = self.opts.model['droptask_e_delay']
        self.manual_schedule = self.opts.model['manual_schedule']

        self.val_tasks_config = self.opts.model['val_tasks']

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
                nn.init.kaiming_normal_(param.data)

    def setup(self, is_train=True):
        """Sets up NN topology by creating the layers."""

        # create encoders
        self.encs = nn.ModuleDict()
        self.encs_type = {}
        enc_switcher = {
            "Text": self.create_text_encoder,
            "Image": self.create_image_encoder,
            #"Speech": create_speech_encoder,
        }

        for e in self.topology.srcs.values():
            logger.info("Creating {} encoder for {}".format(e._type, e))
            create_enc = enc_switcher.get(e._type, "Invalid encoder {} for {}".format(e._type, e))
            self.encs[str(e)] = create_enc(str(e))
            self.encs_type[str(e)] = e._type

        # create latent space
        # NOTE: for now, let's do a simple sum
        # TODO: Karl will provide better latent spaces ;)
        self.z_space = ZSpace(
            ctx_size_dict=self.ctx_sizes, z_size=self.ctx_sizes['z'],
            z_type=self.z_type, activ=self.z_activ)

        # create decoders
        self.decs = nn.ModuleDict()
        self.dec_types = {}
        dec_switcher = {
            "Text": self.create_text_decoder,
            #"Image": create_image_decoder,
            #"Speech": SpeechDecoder,
        }

        for d in self.topology.trgs.values():
            logger.info("Creating {} decoder for {}".format(d._type, d))
            create_dec = dec_switcher.get(d._type, "Invalid decoder {} for {}".format(d._type, d))
            self.decs[str(d)] = create_dec(str(d))
            self.dec_types[str(d)] = d._type

        if is_train:
            # Create the scheduler
            self.scheduler = Scheduler(
                self.topology, self.schedule_type_enc, self.schedule_type_dec,
                self.droptask_prob, self.droptask_e_delay, self.manual_schedule)

        # Create the val_tasks which is a dic {'0': Topology}
        if self.val_tasks_config is not None:
            self.val_tasks = {
                k: Topology(self.val_tasks_config[k]) for k in self.val_tasks_config.keys()}

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

    def get_bos(self, batch_size):
        """Returns a representation for <bos> embeddings for decoding."""
        # NOTE: This thing is not actually useful. always return 0 i.e. <bos>
        bos = next(iter(self.vocabs.values()))['<bos>']
        return torch.LongTensor(batch_size).fill_(bos)

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
        enc_ids = kwargs.get('enc_ids', None)

        #logger.info("encode: batch is {}".format(batch))
        if enc_ids is None:
            raise Exception('Encoders not given')
        else:
            enc_results = {}
            for e in enc_ids:
                #logger.info("encoding batch {} with {} ".format(batch[e].shape, e))
                #the encoders() return a tuple (values, mask) where mask can be None if sent have same length
                enc_results[e] = self.encs[e](batch[e])
                #logger.info("enc_res[{}] size is {}".format(e, enc_results[e][0].shape))

        assert(enc_results), "For some reason, the encoding results are empty!"
        # project into latent space (single vector for now) and return the vector
        # Dictionnary format:  key => (features, mask)
        # NOTE: in the case of single vector Z space, no need for a mask
        return {'z': (self.z_space(enc_results), None)}

    def decode(self, enc_results, batch, dec_ids):
        # Get loss dict
        dec_results = {}
        for d in dec_ids:
            dec_results[d] = self.decs[d](enc_results, batch[d])
            if self.dec_types[d] == 'Text':
                dec_results[d]['n_items'] = torch.nonzero(batch[d][1:]).shape[0]
        return dec_results

    def forward(self, batch, **kwargs):
        """Computes the forward-pass of the network and returns batch loss.

        Arguments:
            batch (dict): A batch of samples with keys designating the source
                and target modalities.

        Returns:
            Tensor:
                A scalar loss normalized w.r.t batch size and token counts.
        """
        #uctr = kwargs['uctr']
        #ectr = kwargs['ectr']
        val_task = kwargs.get('val_task', None)

        dec_results = {}
        # encode the batch and project it to latent space
        if val_task is not None:
            enc_results = self.encode(batch, enc_ids=val_task.srcs)
            dec_results = self.decode(enc_results, batch, val_task.trgs)
        else:
            enc_ids, dec_ids, aux_ = self.scheduler.get_encs_and_decs()
            #logger.info("enc results: {}".format(enc_results))
            enc_results = self.encode(batch, enc_ids=enc_ids)
            dec_results = self.decode(enc_results, batch, dec_ids)
        return dec_results

    def test_performance(self, data_loader, dump_file=None):
        """Computes test set loss over the given DataLoader instance."""
        loss = Loss()

        for batch in pbar(data_loader, unit='batch'):
            batch.device(DEVICE)
            for taskid in self.val_tasks:
                out = self.forward(batch, val_task=self.val_tasks[taskid])
                for d in out:
                    loss.update(out[d]['loss'], out[d]['n_items'])

        return [
            Metric('LOSS', loss.get(), higher_better=False),
        ]

    ######
    # Create a text encoder with default parameters
    ######
    def create_text_encoder(self, id):
        return TextEncoder(
            input_size=self.opts.model['emb_dim'],
            hidden_size=self.opts.model['enc_dim'],
            n_vocab=self.n_svocabs[id],
            rnn_type=self.opts.model['enc_type'],
            dropout_emb=self.opts.model['dropout_emb'],
            dropout_ctx=self.opts.model['dropout_ctx'],
            dropout_rnn=self.opts.model['dropout_enc'],
            num_layers=self.opts.model['n_encoders'],
            emb_maxnorm=self.opts.model['emb_maxnorm'],
            emb_gradscale=self.opts.model['emb_gradscale'])

    ######
    # Create a CNN image encoder with default parameters
    ######
    def create_image_encoder(self):
        cnn_encoder = ImageEncoder(
            cnn_type=self.opts.model['cnn_type'],
            pretrained=self.opts.model['cnn_pretrained'])
        # Set truncation point
        cnn_encoder.setup(layer=self.opts.model['cnn_layer'],
                          dropout=self.opts.model['dropout_img'],
                          pool=self.opts.model['pool'])

        # By default the CNN is not tuneable
        if self.opts.model['cnn_finetune'] is not None:
            assert not self.opts.model['l2_norm'], \
                "finetuning and l2 norm does not work together."
            cnn_encoder.set_requires_grad(
                value=True, layers=self.opts.model['cnn_finetune'])

        # Number of channels defines the spatial vector dim for us
        self.ctx_sizes['image'] = cnn_encoder.get_output_shape()[1]
        return cnn_encoder

    ######
    # Create a text decoder with default parameters
    ######
    def create_text_decoder(self, id):
        return VectorDecoder(
            input_size=self.opts.model['emb_dim'],
            hidden_size=self.opts.model['dec_dim'],
            n_vocab=self.n_tvocabs[id],
            #rnn_type=self.opts.model['dec_type'],
            ctx_size_dict=self.ctx_sizes,
            ctx_name='z',   # TODO this should be changed to z (check with Ozan)
            tied_emb=self.opts.model['tied_emb'],
            dropout_out=self.opts.model['dropout_out'],
            emb_maxnorm=self.opts.model['emb_maxnorm'],
            emb_gradscale=self.opts.model['emb_gradscale'])

    def get_decoder(self, task_id=None):
        return self.decs[task_id]

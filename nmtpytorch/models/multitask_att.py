# -*- coding: utf-8 -*-
import logging
import numpy as np

import torch
from torch import nn

from ..layers import TextEncoder, ImageEncoder, VectorDecoder
from ..layers import FeatureEncoder, MaxMargin, FF
from ..layers import BiLSTMp
from ..layers import SimpleGRUDecoder, ConditionalDecoder, ZSpaceAtt
from ..utils.misc import get_n_params
from ..vocabulary import Vocabulary
from ..utils.topology import Topology
from ..utils.ml_metrics import Loss
from ..utils.device import DEVICE
from ..utils.misc import pbar
from ..datasets import MultimodalDataset
from ..metrics import Metric
from ..utils.nn import mean_pool
from ..utils.scheduler import Scheduler

logger = logging.getLogger('nmtpytorch')


class MultitaskAtt(nn.Module):
    supports_beam_search = True

    def set_defaults(self):
        self.defaults = {
            # ------------- Model generic options
            'direction': None,              # Network directionality, i.e. en->de
            'max_len': 80,                  # Reject sentences where 'bucket_by' length > 80
            'bucket_by': None,              # A key like 'en' to define w.r.t which dataset
                                            # the batches will be sorted
            'bucket_order': None,           # None, ascending or descending for curriculum learning
            'val_tasks': None,               # dictionary of {id:direction} pairs for validation (None|{})
            # ------------- Options for text encoder (bidir RNN)
            'te_emb_dim': 128,              # Source and target embedding sizes
            'te_enc_dim': 128,              # Encoder hidden size
            'te_enc_type': 'gru',           # Encoder type (gru|lstm)
            'te_dropout_emb': 0,            # Simple dropout to source embeddings
            'te_dropout_ctx': 0,            # Simple dropout to source encodings
            'te_dropout_enc': 0,            # Intra-encoder dropout if n_encoders > 1
            'te_n_encoders': 1,             # Number of stacked encoders
            'te_emb_maxnorm': None,         # Normalize embeddings l2 norm to 1
            'te_emb_gradscale': False,      # Scale embedding gradients w.r.t. batch frequency
            # ------------- Options for decoder with attention
            'td_type': 'simple',            # Decoder type (simple/conditional)
            'td_emb_dim': 128,              # Input size
            'td_dec_dim': 128,              # Decoder hidden size
            'td_tied_emb': False,           # Share decoder embeddings
            'td_dec_init': 'mean_ctx',      # How to initialize decoder (zero/mean_ctx/feats)
            'td_att_type': 'mlp',           # Attention type (mlp|dot)
            'td_att_temp': 1.,              # Attention temperature
            'td_att_activ': 'tanh',         # Attention non-linearity (all torch nonlins)
            'td_att_transform_ctx': True,   # Transform annotations before attention
            'td_att_mlp_bias': False,       # Enables bias in attention mechanism
            'td_att_bottleneck': 'ctx',     # Bottleneck dimensionality (ctx|hid)
            'td_dropout_out': 0,            # Simple dropout to decoder output
            'td_emb_maxnorm': None,         # Normalize embeddings l2 norm to 1
            'td_emb_gradscale': False,      # Scale embedding gradients w.r.t. batch frequency
            # ------------- Additional options for conditional decoder
            'td_dec_type': 'gru',           # Decoder type (gru|lstm)
            'td_dec_init_size': None,       # feature vector dimensionality for dec_init == 'feats'
            'td_dec_init_activ': 'tanh',    # Decoder initialization activation func
            'td_dropout': 0,                # Generic dropout overall the architecture
            # ------------- Options for image CNN encoder
            'ie_cnn_type': 'resnet50',      # A variant of VGG or ResNet
            'ie_cnn_pretrained': True,      # Should we use pretrained imagenet weights
            'ie_cnn_layer': 'res5c_relu',   # From where to extract features
            'ie_dropout_img': 0.,           # a 2d dropout over conv features
            'ie_pool': None,                # ('Avg|Max', kernel_size, stride_size)
            'ie_cnn_finetune': None,        # Should we finetune part or all of CNN
            'ie_l2_norm': False,            # L2 normalize features
            # NOTE those options are not provided to create the image encoder but found initialized in amnmt.py
            #'ie_l2_norm_dim': -1,          # Which dimension to L2 normalize
            #'ie_resize': 256,              # resize width, height for images
            #'ie_crop': 224,                # center crop size after resize
            # ------------- Options for video encoder
            've_dim': 2048,                 # Video frame input size
            've_proj_size': 512,            # Video frame embedding size
            've_enc_dim': 256,              # Encoder hidden size
            've_enc_type': 'gru',           # Encoder type (gru|lstm)
            've_dropout_emb': 0,            # Simple dropout to source embeddings
            've_dropout_ctx': 0,            # Simple dropout to source encodings
            've_dropout_enc': 0,            # Intra-encoder dropout if n_encoders > 1
            've_n_encoders': 1,             # Number of stacked encoders
            've_bidirectional': True,       # Enable bidirectional encoder
            # ------------- Options for video decoder
            'vd_emb_dim': 256,             # Source and target embedding sizes
            'vd_vid_dim': 2048,            # Video frame input size
            'vd_proj_size': 512,           # Video frame embedding size
            'vd_emb_maxnorm': None,        # Normalize embeddings l2 norm to 1
            'vd_emb_gradscale': False,     # Scale embedding gradients w.r.t. batch frequency
            'vd_dec_dim': 512,             # Decoder hidden size
            'vd_dec_type': 'gru',          # Decoder type (gru|lstm)
            'vd_dec_init': 'mean_ctx',     # How to initialize decoder (zero/mean_ctx/feats)
            'vd_dec_init_size': None,      # feature vector dimensionality for
                                           # dec_init == 'feats'
            'vd_att_type': 'mlp',          # Attention type (mlp|dot)
            'vd_att_temp': 1.,             # Attention temperature
            'vd_att_activ': 'tanh',        # Attention non-linearity (all torch nonlins)
            'vd_att_mlp_bias': False,      # Enables bias in attention mechanism
            'vd_att_bottleneck': 'ctx',    # Bottleneck dimensionality (ctx|hid)
            'vd_att_transform_ctx': True,  # Transform annotations before attention
            'vd_bidirectional': True,      # Whether the encoder is bidirectional or not
            'vd_dropout_emb': 0,           # Simple dropout to source embeddings
            'vd_dropout_out': 0,           # Simple dropout to decoder output
            'vd_loss_type': 'SmoothL1',    # Loss type (MSE_loss | SmoothL1)
            # ------------- Options for BiLSTMp speech encoder
            'se_feat_dim': 43,              # Speech features dimensionality
            'se_enc_dim': 256,              # Encoder hidden size
            'se_dropout': 0,                # Generic dropout overall the architecture
            'se_enc_layers': '1_1_2_2_1_1', # Subsampling & layer architecture
            'se_proj_dim': 320,             # Intra-LSTM projection layer dim
            # ------------- Options for the shared z-space
            'z_size': 256,                  # size of hidden state of z-space
            'z_len': 10,                    # how many latent states to produce
            'z_transform': None,            # how to transform input contexts (None|linear|tanh|sigmoid)
            'z_in_size': 256,               # input size of the ZSpace layer
            'z_merge': 'sum',               # How to merge the attended vector to feed the ZSpace layer
            # ------------- Options for the scheduler
            'schedule_type_enc': None,      # drop encoder(s) randomly (None|random|random_1)
            'schedule_type_dec': None,      # drop decoder(s) randomly (None|random|random_1)
            'droptask_prob': 1,             # probability of dropping encoder(s)/decoder(s)
                                            # (only used for non-None schedule_type_enc/dec)
            'droptask_e_delay': None,       # number of completed epochs before droptask
            'manual_schedule': None,        # dictionary of {id:direction@num_batches} pairs to cycle thru (None|{})
            'loss_scaling': None,           # dictionary with same keys as manual_schedule for loss scaling constants
            # ------------- Options for mutual projection networks
            'use_z': True,                  # whether to use z-space or decode directly from encoders
            'use_mpn': False,               # whether to use auxiliary max-margin loss objective
            'use_decmpn': False,            # use auxiliary max-margin objective in the decoder
            'pooling_type': 'mean',         # pooling method to be used before max-margin layer (mean)
            'margin': 0.1,                  # max-margin layer "alpha"
            'max_violation': False,         # max-margin hinge type (True: max-of-hinges, False: sum-of-hinges)
            'sim_function': 'cosine'        # max-margin similarity function
        }

    def __init__(self, opts):
        super().__init__()

        # opts -> config file sections {.model, .data, .vocabulary, .train}
        self.opts = opts

        # Langs, Vocabulary and Vocab Length objects
        self.vocabs = {}        # all vocabularies
        self.slangs = []        # source languages IDs
        self.svocabs = {}       # source vocabs
        self.n_svocabs = {}     # sizes of source vocabs
        self.tlangs = []        # target languages IDs
        self.tvocabs = {}       # target vocabs
        self.n_tvocabs = {}     # sizes of sources vocabs
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
            # NOTE: for language-specific evaluation metrics (e.g. BLEU),
            # this will be overwritten by the 0th topology in 'val_tasks' in the conf file
            self.val_refs[tl] = self.opts.data['val_set'][tl]

        # Textual context size is always equal to enc_dim * 2 since
        # it is the concatenation of forward and backward hidden states
        if 'te_enc_dim' in self.opts.model:
            for sl in slangs:
                self.ctx_sizes[str(sl)] = self.opts.model['te_enc_dim'] * 2

        # Check tying option
        if self.opts.model['td_tied_emb'] not in [False, '2way']:
            raise RuntimeError(
                "'{}' not recognized for td_tied_emb.".format(self.opts.model['td_tied_emb']))

        self.td_type = self.opts.model['td_type']
        # FIXME: this small hack because of string mismatch between Simple and Cond decoder
        # FIXME: this should be changed in cond_decoder.py
        if self.td_type == 'conditional' and self.opts.model['td_dec_init'] == 'mean':
            self.opts.model['td_dec_init'] = 'mean_ctx'

        # TODO: VISION generic init
        # TODO: SPEECH generic init

        # MPN options init
        self.use_z = self.opts.model['use_z']
        self.use_mpn = self.opts.model['use_mpn']
        self.use_decmpn = self.opts.model['use_decmpn']
        self.pooling_type = self.opts.model['pooling_type']
        ############################
        # Create the max-margin loss
        ############################
        if self.use_mpn or self.use_decmpn:
            assert len(self.topology.srcs) >= 2, \
                "For MPN, there must be at least two different encoders defined in the overall topology."
            self.mm_loss = MaxMargin(
                margin=self.opts.model['margin'],
                # sim_function=self.opts.model['sim_function'],
                max_violation=self.opts.model['max_violation'])

        # Latent space options init
        self.z_size = self.opts.model['z_size']
        self.z_len = self.opts.model['z_len']
        self.z_transform = self.opts.model['z_transform']
        self.z_in_size = self.opts.model['z_in_size']
        self.z_merge = self.opts.model['z_merge']

        # Scheduler options init
        self.schedule_type_enc = self.opts.model['schedule_type_enc']
        self.schedule_type_dec = self.opts.model['schedule_type_dec']
        self.droptask_prob = self.opts.model['droptask_prob']
        self.droptask_e_delay = self.opts.model['droptask_e_delay']
        self.manual_schedule = self.opts.model['manual_schedule']
        self.loss_scaling = self.opts.model['loss_scaling']
        self.val_tasks_config = self.opts.model['val_tasks']

        # FIXME: if no val_task is given, use the topology as val_task
        # Create val_tasks which is a dict {'0': Topology}
        if self.val_tasks_config is not None:
            self.val_tasks = {}
            self.val_tasks = {k: Topology(self.val_tasks_config[k]) for k in self.val_tasks_config.keys()}
            # If val_tasks is given, override the general val_refs which by default uses all decoders
            # Convention: the 0th val_task is the one used for language-specific evaluation
            assert len(self.val_tasks[0].get_trg_langs()) == 1, \
                "0th val_task must have only one decoder for picking an evaluation reference."
            ref_override = self.val_tasks[0].get_trg_langs()[0]
            logger.info(
                "Note: for language-specific evaluation metrics, if enabled, using {}.\n"
                "(This reference is specified by the first decoder in the '0' Topology from val_tasks)".format(ref_override))
            self.val_refs = self.opts.data['val_set'][ref_override]

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
            "Kaldi": self.create_speech_encoder,
            "Shelve": self.create_video_encoder
        }
        self.single_ffs = nn.ModuleDict()
        ff_switcher = {
            "Text": self.create_text_ff,
            "Kaldi": self.create_speech_ff,
            "Shelve": self.create_video_ff
        }

        for e in self.topology.srcs.values():
            logger.info("Creating {} encoder for {}".format(e._type, e))
            create_enc = enc_switcher.get(e._type, "Invalid encoder {} for {}".format(e._type, e))
            self.encs[str(e)] = create_enc(str(e))
            self.encs_type[str(e)] = e._type
            create_ff = ff_switcher.get(e._type, "Invalid FF transform {} for {}".format(e._type, e))
            self.single_ffs[str(e)] = create_ff(str(e))

            if e._type.startswith('Shelve'):
                if 've_enc_dim' in self.opts.model:
                    if self.opts.model['ve_bidirectional']:
                        self.ctx_sizes[str(e)] = self.opts.model['ve_enc_dim'] * 2
                    else:
                        self.ctx_sizes[str(e)] = self.opts.model['ve_enc_dim']
            elif e._type.startswith('Kaldi'):
                self.ctx_sizes[str(e)] = self.opts.model['se_enc_dim'] * 2

        # create shared space
        # NOTE: let's do a more complex z-space generating several states with attention a la Lu et al. 2018
        if self.use_z:
            self.z_space = ZSpaceAtt(
                ctx_size_dict=self.ctx_sizes, z_size=self.z_size,
                z_len=self.z_len, z_transform=self.z_transform,
                z_in_size=self.z_in_size, z_merge=self.z_merge)
        self.ctx_sizes['z'] = self.z_size

        # create decoders
        self.decs = nn.ModuleDict()
        self.dec_types = {}
        dec_switcher = {
            "Image": self.create_image_decoder,
            "Text": self.create_attentional_text_decoder,
            "Kaldi": self.create_speech_decoder,
            "Shelve": self.create_video_decoder
        }
        dec_ff_switcher = {
            "Text": self.create_dec_text_ff,
        }

        for d in self.topology.trgs.values():
            logger.info("Creating {} decoder for {}".format(d._type, d))
            create_dec = dec_switcher.get(d._type, "Invalid decoder {} for {}".format(d._type, d))
            self.decs[str(d)] = create_dec(str(d))
            self.dec_types[str(d)] = d._type
            create_ff = dec_ff_switcher.get(d._type, "Invalid FF transform {} for {}".format(d._type, d))
            self.single_ffs[str(d)] = create_ff(str(d))

        if is_train:
            # create scheduler
            self.scheduler = Scheduler(
                self.topology, self.schedule_type_enc, self.schedule_type_dec,
                self.droptask_prob, self.droptask_e_delay, self.manual_schedule)
            if self.use_mpn:
                self.scheduler.check_mpn()
            if self.manual_schedule is not None and self.loss_scaling is not None:
                assert self.manual_schedule.keys() == self.loss_scaling.keys(), \
                    "Keys for manual_schedule and loss_scaling must match."

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
        if self.use_z:
            return {'z': (self.z_space(enc_results), None)}
        else:
            # TODO: implement a "do nothing" Z space option that simply returns the encoder results given it
            # for now, change encoder key to be 'z' so that decoders don't complain
            # ... and make sure encoder hidden states for all encoders are the same size (a la z_size, e.g. 256)
            # ... and make sure you're only using one encoder per task as defined in manual_schedule
            enc_results['z'] = enc_results.pop([*enc_results][0])
            return enc_results

    def decode(self, enc_results, batch, dec_ids):
        # Get loss dict
        dec_results = {}
        for d in dec_ids:
            dec_results[d] = self.decs[d](enc_results, batch[d])
            if 'n_items' not in dec_results[d]:
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
        # uctr = kwargs['uctr']
        # ectr = kwargs['ectr']
        val_task = kwargs.get('val_task', None)

        dec_results = {}
        # encode the batch and project it to latent space
        if val_task is not None:
            # i.e., this forward pass is being called for evaluation
            enc_results = self.encode(batch, enc_ids=val_task.srcs)
            dec_results = self.decode(enc_results, batch, val_task.trgs)
        else:
            enc_ids, dec_ids, aux_enc_ids = self.scheduler.get_encs_and_decs()
            enc_results = self.encode(batch, enc_ids=enc_ids)
            dec_results = self.decode(enc_results, batch, dec_ids)
            if self.use_mpn:
                # i.e. use MPN setup
                # Randomly sample an auxiliary encoder
                aux_enc = aux_enc_ids[np.random.randint(0, len(aux_enc_ids))]
                aux_results = self.encode(batch, enc_ids=[aux_enc])

                # pool and project the encoder
                enc_pool = mean_pool(enc_results['z'])
                enc_proj = self.single_ffs[[*enc_ids.keys()][0]](enc_pool)

                # pool and project the aux encoder
                aux_pool = mean_pool(aux_results['z'])
                aux_proj = self.single_ffs[aux_enc](aux_pool)

                # collect the encoder MPN term
                enc_mpn = self.mm_loss(enc_proj, aux_proj)['loss']

                if self.use_decmpn:
                    # collect the decoder hidden states and the mask
                    decoder_key = [*dec_ids][0]
                    decoder_hiddens = torch.stack(self.decs[decoder_key].hiddens)
                    decoder_mask = (batch[decoder_key] != 0).float()

                    # pool and project the decoder
                    dec_pool = mean_pool((decoder_hiddens, decoder_mask))
                    dec_proj = self.single_ffs[decoder_key](dec_pool)

                    # collect the decoder MPN term
                    dec_mpn = self.mm_loss(dec_proj, aux_proj)['loss']
                    self.aux_loss['mpn'] = (enc_mpn + dec_mpn) * self.opts.train['mpn_scale']
                else:
                    self.aux_loss['mpn'] = enc_mpn * self.opts.train['mpn_scale']

            if self.loss_scaling is not None:
                curr_task = self.scheduler.curr_key
                dec_results[[*dec_ids][0]]['loss'] *= self.loss_scaling[curr_task]
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

    """
        Naming convention for config variables for various modalities
        <first letter modality><e for enc, d for dec>_<variable name>
        Ex: te_emb_dim: emb_dim for text encoder
            ie_cnn_type: cnn_type for image encoder
            td_att_type: att_type for text decoder
    """

    ######
    # Functions to create a text encoder and decoder with default parameters
    ######
    def create_text_encoder(self, id):
        return TextEncoder(
            input_size=self.opts.model['te_emb_dim'],
            hidden_size=self.opts.model['te_enc_dim'],
            n_vocab=self.n_svocabs[id],
            rnn_type=self.opts.model['te_enc_type'],
            dropout_emb=self.opts.model['te_dropout_emb'],
            dropout_ctx=self.opts.model['te_dropout_ctx'],
            dropout_rnn=self.opts.model['te_dropout_enc'],
            num_layers=self.opts.model['te_n_encoders'],
            emb_maxnorm=self.opts.model['te_emb_maxnorm'],
            emb_gradscale=self.opts.model['te_emb_gradscale'])

    def create_text_ff(self, id):
        ''' Only used to create an additional non-linearity between
            a pooled layer and the max-margin layer '''
        return FF(self.opts.model['te_enc_dim'] * 2,
                  self.opts.model['z_size'])

    def create_dec_text_ff(self, id):
        ''' Only used to create an additional non-linearity between
            a pooled layer and the max-margin layer '''
        return FF(self.opts.model['te_enc_dim'],
                  self.opts.model['td_dec_dim'] * 2)

    def create_text_decoder(self, id):
        return VectorDecoder(
            input_size=self.opts.model['td_emb_dim'],
            hidden_size=self.opts.model['td_dec_dim'],
            n_vocab=self.n_tvocabs[id],
            ctx_size_dict=self.ctx_sizes,
            ctx_name='z',
            tied_emb=self.opts.model['td_tied_emb'],
            dropout_out=self.opts.model['td_dropout_out'],
            emb_maxnorm=self.opts.model['td_emb_maxnorm'],
            emb_gradscale=self.opts.model['td_emb_gradscale'])

    def create_attentional_text_decoder(self, id):
        if self.td_type == 'simple':
            return self.create_simple_attentional_text_decoder(id)
        elif self.td_type == 'conditional':
            if self.use_decmpn:
                return self.create_mpn_cond_attentional_text_decoder(id)
            return self.create_cond_attentional_text_decoder(id)

        raise Exception('Unknown text decoder type {}, should be one of simple/conditional'.format(self.td_type))

    def create_simple_attentional_text_decoder(self, id):
        return SimpleGRUDecoder(
            input_size=self.opts.model['td_emb_dim'],
            hidden_size=self.opts.model['td_dec_dim'],
            n_vocab=self.n_tvocabs[id],
            #rnn_type=self.opts.model['dec_type'],
            ctx_size_dict=self.ctx_sizes,
            ctx_name='z',
            tied_emb=self.opts.model['td_tied_emb'],
            dec_init=self.opts.model['td_dec_init'],
            att_type=self.opts.model['td_att_type'],
            att_temp=self.opts.model['td_att_temp'],
            att_activ=self.opts.model['td_att_activ'],
            transform_ctx=self.opts.model['td_att_transform_ctx'],
            mlp_bias=self.opts.model['td_att_mlp_bias'],
            att_bottleneck=self.opts.model['td_att_bottleneck'],
            dropout_out=self.opts.model['td_dropout_out'],
            emb_maxnorm=self.opts.model['td_emb_maxnorm'],
            emb_gradscale=self.opts.model['td_emb_gradscale'])

    def create_cond_attentional_text_decoder(self, id):
        return ConditionalDecoder(
            input_size=self.opts.model['td_emb_dim'],
            hidden_size=self.opts.model['td_dec_dim'],
            n_vocab=self.n_tvocabs[id],
            rnn_type=self.opts.model['td_dec_type'],
            ctx_size_dict=self.ctx_sizes,
            ctx_name='z',
            tied_emb=self.opts.model['td_tied_emb'],
            dec_init=self.opts.model['td_dec_init'],
            dec_init_size=self.opts.model['td_dec_init_size'],
            dec_init_activ=self.opts.model['td_dec_init_activ'],
            att_type=self.opts.model['td_att_type'],
            att_temp=self.opts.model['td_att_temp'],
            att_activ=self.opts.model['td_att_activ'],
            transform_ctx=self.opts.model['td_att_transform_ctx'],
            mlp_bias=self.opts.model['td_att_mlp_bias'],
            att_bottleneck=self.opts.model['td_att_bottleneck'],
            dropout_out=self.opts.model['td_dropout'])

#     def create_mpn_cond_attentional_text_decoder(self, id):
        # return MPNConditionalDecoder(
            # input_size=self.opts.model['td_emb_dim'],
            # hidden_size=self.opts.model['td_dec_dim'],
            # n_vocab=self.n_tvocabs[id],
            # rnn_type=self.opts.model['td_dec_type'],
            # ctx_size_dict=self.ctx_sizes,
            # ctx_name='z',
            # tied_emb=self.opts.model['td_tied_emb'],
            # dec_init=self.opts.model['td_dec_init'],
            # dec_init_size=self.opts.model['td_dec_init_size'],
            # dec_init_activ=self.opts.model['td_dec_init_activ'],
            # att_type=self.opts.model['td_att_type'],
            # att_temp=self.opts.model['td_att_temp'],
            # att_activ=self.opts.model['td_att_activ'],
            # transform_ctx=self.opts.model['td_att_transform_ctx'],
            # mlp_bias=self.opts.model['td_att_mlp_bias'],
            # att_bottleneck=self.opts.model['td_att_bottleneck'],
            # dropout_out=self.opts.model['td_dropout'])

    ######
    # Functions to create a CNN image encoder with default parameters
    ######
    def create_image_encoder(self, id):
        cnn_encoder = ImageEncoder(
            cnn_type=self.opts.model['ie_cnn_type'],
            pretrained=self.opts.model['ie_cnn_pretrained'])
        # Set truncation point
        cnn_encoder.setup(layer=self.opts.model['ie_cnn_layer'],
                          dropout=self.opts.model['ie_dropout_img'],
                          pool=self.opts.model['ie_pool'])

        # By default the CNN is not tuneable
        if self.opts.model['ie_cnn_finetune'] is not None:
            assert not self.opts.model['ie_l2_norm'], \
                "finetuning and l2 norm does not work together."
            cnn_encoder.set_requires_grad(
                value=True, layers=self.opts.model['ie_cnn_finetune'])

        # Number of channels defines the spatial vector dim for us
        self.ctx_sizes[id] = cnn_encoder.get_output_shape()[1]
        return cnn_encoder

    def create_image_decoder(self, id):
        raise Exception('No image decoder available...(yet!)')
        #return ImageDecoder()

    ######
    # Functions to create a video encoder and decoder with default parameters
    ######
    def create_video_encoder(self, id):
        return FeatureEncoder(
            input_size=self.opts.model['ve_dim'],
            proj_size=self.opts.model['ve_proj_size'],
            hidden_size=self.opts.model['ve_enc_dim'],
            rnn_type=self.opts.model['ve_enc_type'],
            dropout_emb=self.opts.model['ve_dropout_emb'],
            dropout_ctx=self.opts.model['ve_dropout_ctx'],
            dropout_rnn=self.opts.model['ve_dropout_enc'],
            num_layers=self.opts.model['ve_n_encoders'],
            bidirectional=self.opts.model['ve_bidirectional'])

    def create_video_ff(self, id):
        ''' Only used to create an additional non-linearity between
            a pooled layer and the max-margin layer '''
        input_size = (self.opts.model['ve_enc_dim'] * 2
                      if self.opts.model['ve_bidirectional']
                      else self.opts.model['ve_enc_dim'])
        return FF(input_size,
                  self.opts.model['z_size'])

#     def create_video_decoder(self, id):
        # return ReverseVideoDecoder(
            # input_size=self.opts.model['vd_emb_dim'],
            # hidden_size=self.opts.model['vd_dec_dim'],
            # rnn_type=self.opts.model['vd_dec_type'],
            # video_dim=self.opts.model['vd_vid_dim'],
            # ctx_size_dict=self.ctx_sizes,
            # ctx_name='z',
            # dec_init=self.opts.model['vd_dec_init'],
            # dec_init_size=self.opts.model['vd_dec_init_size'],
            # att_type=self.opts.model['vd_att_type'],
            # att_temp=self.opts.model['vd_att_temp'],
            # att_activ=self.opts.model['vd_att_activ'],
            # transform_ctx=self.opts.model['vd_att_transform_ctx'],
            # mlp_bias=self.opts.model['vd_att_mlp_bias'],
            # att_bottleneck=self.opts.model['vd_att_bottleneck'],
            # dropout_out=self.opts.model['vd_dropout_out'],
            # use_smoothL1=self.opts.model['vd_loss_type'])

    ######
    # Functions to create a speech encoder and decoder with default parameters
    ######
    def create_speech_encoder(self, id):
        return BiLSTMp(
            input_size=self.opts.model['se_feat_dim'],
            hidden_size=self.opts.model['se_enc_dim'],
            proj_size=self.opts.model['se_proj_dim'],
            layers=self.opts.model['se_enc_layers'],
            dropout=self.opts.model['se_dropout'])

    def create_speech_ff(self, id):
        ''' Only used to create an additional non-linearity between
            a pooled layer and the max-margin layer '''
        return FF(self.opts.model['se_enc_dim'] * 2,
                  self.opts.model['z_size'])

    def create_speech_decoder(self, id):
        raise Exception('No speech decoder available...(yet!)')

    def get_decoder(self, task_id=None):
        return self.decs[task_id]

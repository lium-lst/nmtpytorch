import time
from .utils.misc import load_pt_file
from .utils.filterchain import FilterChain


class Translator(object):
    def __init__(self, logger, model,
                 batch_size, beam_size, max_len,
                 avoid_rep=False, avoid_unk=False, apply_filters=True):
        self.logger = logger
        self.model = model
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.max_len = max_len
        self.avoid_rep = avoid_rep
        self.avoid_unk = avoid_unk
        if apply_filters and self.model.opts.train.get('eval_filters', ''):
            self.filter = FilterChain(self.model.opts.train['eval_filters'])
        else:
            self.filter = lambda s: s

        # Setup the model
        self.model.setup()
        self.is_ready = False
        self.results = {}

    def load(self, model_file):
        self.model_file = model_file
        weights = load_pt_file(model_file)[0]
        self.model.load_state_dict(weights)

        # Only on the first model is enough
        if not self.is_ready:
            self.model.train(False)
            self.model.cuda()
            self.is_ready = True

    def translate(self, split, dump=False):
        self.model.load_data(split)
        loader = self.model.datasets[split].get_iterator(
            self.batch_size, only_source=True)
        self.logger.info('Starting translation on "{}"'.format(split))
        start = time.time()

        if self.beam_size == 1 and hasattr(self.model, 'greedy_search'):
            hyps = self.model.greedy_search(loader, self.max_len,
                                            self.avoid_rep, self.avoid_unk)
        else:
            hyps = self.model.beam_search(loader, self.beam_size, self.max_len,
                                          self.avoid_rep, self.avoid_unk)

        up_time = time.time() - start
        self.logger.info('Took {:.3f} seconds, {:.4f} sec/hyp'.format(
            up_time, up_time / len(hyps)))

        self.results[split] = self.filter(hyps)

        if dump:
            self.dump(split)

    def dump(self, split):
        """Writes the results into a file. File name is set as
        '<model_file>.<split>.beam<beamsize>'.
        """

        output = "{}.{}.beam{}".format(self.model_file, split, self.beam_size)

        with open(output, 'w') as f:
            for line in self.results[split]:
                f.write(line + '\n')

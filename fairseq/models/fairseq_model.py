# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import FairseqDecoder, FairseqEncoder#, FactoredCompositeEncoder
from .factored_composite_encoder import FactoredCompositeEncoder
from fairseq.data import Dictionary


class BaseFairseqModel(nn.Module):
    """Base class for fairseq models."""

    def __init__(self):
        super().__init__()
        self._is_generation_fast = False

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        pass

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        raise NotImplementedError

    def get_targets(self, sample, net_output):
        """Get targets from either the sample or the net's output."""
        return sample['target']

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        if hasattr(self, 'decoder'):
            return self.decoder.get_normalized_probs(net_output, log_probs, sample)
        elif torch.is_tensor(net_output):
            logits = net_output.float()
            if log_probs:
                return F.log_softmax(logits, dim=-1)
            else:
                return F.softmax(logits, dim=-1)
        raise NotImplementedError

    def max_positions(self):
        """Maximum length supported by the model."""
        return None

    def max_decoder_positions(self):
        """Maximum length supported by the decoder."""
        return self.decoder.max_positions()

    def load_state_dict(self, state_dict, strict=True):
        """Copies parameters and buffers from *state_dict* into this module and
        its descendants.

        Overrides the method in :class:`nn.Module`. Compared with that method
        this additionally "upgrades" *state_dicts* from old checkpoints.
        """
        self.upgrade_state_dict(state_dict)
        super().load_state_dict(state_dict, strict)

    def upgrade_state_dict(self, state_dict):
        """Upgrade old state dicts to work with newer code."""
        self.upgrade_state_dict_named(state_dict, '')

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade old state dicts to work with newer code.

        Args:
            state_dict (dict): state dictionary to upgrade, in place
            name (str): the state dict key corresponding to the current module
        """
        assert state_dict is not None

        def do_upgrade(m, prefix):
            if len(prefix) > 0:
                prefix += '.'

            for n, c in m.named_children():
                name = prefix + n
                if hasattr(c, 'upgrade_state_dict_named'):
                    c.upgrade_state_dict_named(state_dict, name)
                elif hasattr(c, 'upgrade_state_dict'):
                    c.upgrade_state_dict(state_dict)
                do_upgrade(c, name)

        do_upgrade(self, name)

    def make_generation_fast_(self, **kwargs):
        """Optimize model for faster generation."""
        if self._is_generation_fast:
            return  # only apply once
        self._is_generation_fast = True

        # remove weight norm from all modules in the network
        def apply_remove_weight_norm(module):
            try:
                nn.utils.remove_weight_norm(module)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(apply_remove_weight_norm)

        seen = set()

        def apply_make_generation_fast_(module):
            if module != self and hasattr(module, 'make_generation_fast_') \
                    and module not in seen:
                seen.add(module)
                module.make_generation_fast_(**kwargs)

        self.apply(apply_make_generation_fast_)

        def train(mode=True):
            if mode:
                raise RuntimeError('cannot train after make_generation_fast')

        # this model should no longer be used for training
        self.eval()
        self.train = train

    def prepare_for_onnx_export_(self, **kwargs):
        """Make model exportable via ONNX trace."""
        seen = set()

        def apply_prepare_for_onnx_export_(module):
            if module != self and hasattr(module, 'prepare_for_onnx_export_') \
                    and module not in seen:
                seen.add(module)
                module.prepare_for_onnx_export_(**kwargs)

        self.apply(apply_prepare_for_onnx_export_)


class FairseqModel(BaseFairseqModel):
    """Base class for encoder-decoder models.

    Args:
        encoder (FairseqEncoder): the encoder
        decoder (FairseqDecoder): the decoder
    """

    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        assert isinstance(self.encoder, FairseqEncoder)
        assert isinstance(self.decoder, FairseqDecoder)

    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        """
        Run the forward pass for an encoder-decoder model.

        First feed a batch of source tokens through the encoder. Then, feed the
        encoder output and previous decoder outputs (i.e., input feeding/teacher
        forcing) to the decoder to produce the next outputs::

            encoder_out = self.encoder(src_tokens, src_lengths)
            return self.decoder(prev_output_tokens, encoder_out)

        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing

        Returns:
            the decoder's output, typically of shape `(batch, tgt_len, vocab)`
        """
        encoder_out = self.encoder(src_tokens, src_lengths)
        decoder_out = self.decoder(prev_output_tokens, encoder_out)
        return decoder_out

    def max_positions(self):
        """Maximum length supported by the model."""
        return (self.encoder.max_positions(), self.decoder.max_positions())


class FairseqMultiModel(BaseFairseqModel):
    """Base class for combining multiple encoder-decoder models."""
    def __init__(self, encoders, decoders):
        super().__init__()
        assert encoders.keys() == decoders.keys()
        self.keys = list(encoders.keys())
        for key in self.keys:
            assert isinstance(encoders[key], FairseqEncoder)
            assert isinstance(decoders[key], FairseqDecoder)

        self.models = nn.ModuleDict({
            key: FairseqModel(encoders[key], decoders[key])
            for key in self.keys
        })

    @staticmethod
    def build_shared_embeddings(
        dicts: Dict[str, Dictionary],
        langs: List[str],
        embed_dim: int,
        build_embedding: callable,
        pretrained_embed_path: Optional[str] = None,
    ):
        """
        Helper function to build shared embeddings for a set of languages after
        checking that all dicts corresponding to those languages are equivalent.

        Args:
            dicts: Dict of lang_id to its corresponding Dictionary
            langs: languages that we want to share embeddings for
            embed_dim: embedding dimension
            build_embedding: callable function to actually build the embedding
            pretrained_embed_path: Optional path to load pretrained embeddings
        """
        shared_dict = dicts[langs[0]]
        if any(dicts[lang] != shared_dict for lang in langs):
            raise ValueError(
                '--share-*-embeddings requires a joined dictionary: '
                '--share-encoder-embeddings requires a joined source '
                'dictionary, --share-decoder-embeddings requires a joined '
                'target dictionary, and --share-all-embeddings requires a '
                'joint source + target dictionary.'
            )
        return build_embedding(
            shared_dict, embed_dim, pretrained_embed_path
        )

    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        decoder_outs = {}
        for key in self.keys:
            encoder_out = self.models[key].encoder(src_tokens, src_lengths)
            decoder_outs[key] = self.models[key].decoder(prev_output_tokens, encoder_out)
        return decoder_outs

    def max_positions(self):
        """Maximum length supported by the model."""
        return {
            key: (self.models[key].encoder.max_positions(), self.models[key].decoder.max_positions())
            for key in self.keys
        }

    def max_decoder_positions(self):
        """Maximum length supported by the decoder."""
        return min(model.decoder.max_positions() for model in self.models.values())

    @property
    def encoder(self):
        return self.models[self.keys[0]].encoder

    @property
    def decoder(self):
        return self.models[self.keys[0]].decoder


class FairseqLanguageModel(BaseFairseqModel):
    """Base class for decoder-only models.

    Args:
        decoder (FairseqDecoder): the decoder
    """

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder
        assert isinstance(self.decoder, FairseqDecoder)

    def forward(self, src_tokens, src_lengths):
        """
        Run the forward pass for a decoder-only model.

        Feeds a batch of tokens through the decoder to predict the next tokens.

        Args:
            src_tokens (LongTensor): tokens on which to condition the decoder,
                of shape `(batch, tgt_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`

        Returns:
            the decoder's output, typically of shape `(batch, seq_len, vocab)`
        """
        return self.decoder(src_tokens)

    def max_positions(self):
        """Maximum length supported by the model."""
        return self.decoder.max_positions()

    @property
    def supported_targets(self):
        return {'future'}

    def remove_head(self):
        """Removes the head of the model (e.g. the softmax layer) to conserve space when it is not needed"""
        raise NotImplementedError()

'''
# For factored transformer
class FairseqFactoredMultiModel2(BaseFairseqModel):
    """Base class for combining multiple encoder-decoder models."""
    def __init__(self, encoders, decoders):
        super().__init__()
        assert encoders.keys() == decoders.keys()
        self.keys = list(encoders.keys())
        for key in self.keys:
            assert isinstance(encoders[key], FairseqEncoder)
            assert isinstance(decoders[key], FairseqDecoder)

        self.models = nn.ModuleDict({
            key: FairseqModel(encoders[key], decoders[key])
            for key in self.keys
        })

    @staticmethod
    def build_shared_embeddings(
        dicts: Dict[str, Dictionary],
        langs: List[str],
        embed_dim: int,
        build_embedding: callable,
        pretrained_embed_path: Optional[str] = None,
    ):
        """
        Helper function to build shared embeddings for a set of languages after
        checking that all dicts corresponding to those languages are equivalent.

        Args:
            dicts: Dict of lang_id to its corresponding Dictionary
            langs: languages that we want to share embeddings for
            embed_dim: embedding dimension
            build_embedding: callable function to actually build the embedding
            pretrained_embed_path: Optional path to load pretrained embeddings
        """
        shared_dict = dicts[langs[0]]
        if any(dicts[lang] != shared_dict for lang in langs):
            raise ValueError(
                '--share-*-embeddings requires a joined dictionary: '
                '--share-encoder-embeddings requires a joined source '
                'dictionary, --share-decoder-embeddings requires a joined '
                'target dictionary, and --share-all-embeddings requires a '
                'joint source + target dictionary.'
            )
        return build_embedding(
            shared_dict, embed_dim, pretrained_embed_path
        )

    def forward2(self, src_tokens, src_lengths, prev_output_tokens):
        decoder_outs = {}
        concat_encoder = None
        for key in self.keys:
            encoder_out = self.models[key].encoder(src_tokens, src_lengths)
            if concat_encoder is None:
                concat_encoder = encoder_out
            else:
                concat_encoder = torch.cat((concat_encoder, encoder_out))
            # decoder_outs[key] = self.models[key].decoder(prev_output_tokens, encoder_out)
        for key in self.keys:
            #return self.models[key].decoder(prev_output_tokens, concat_encoder)
            decoder_outs[key] = self.models[key].decoder(prev_output_tokens, concat_encoder)
        return decoder_outs

    def forward(self, inputs):
        print(inputs)
        decoder_outs = {}
        concat_encoder = None
        for key in self.keys:
            src_tokens = inputs[key][src_tokens]
            src_lengths = inputs[key][src_lengths]
            prev_output_tokens = inputs[key][src_lengths]
            encoder_out = self.models[key].encoder(src_tokens, src_lengths)
            if concat_encoder is None:
                concat_encoder = encoder_out
            else:
                concat_encoder = torch.cat((concat_encoder, encoder_out))
        for key in self.keys:
            #return self.models[key].decoder(prev_output_tokens, concat_encoder)
            decoder_outs[key] = self.models[key].decoder(prev_output_tokens, concat_encoder)
        return decoder_outs

    def max_positions(self):
        """Maximum length supported by the model."""
        return {
            key: (self.models[key].encoder.max_positions(), self.models[key].decoder.max_positions())
            for key in self.keys
        }

    def max_decoder_positions(self):
        """Maximum length supported by the decoder."""
        return min(model.decoder.max_positions() for model in self.models.values())

    @property
    def encoder(self):
        return self.models[self.keys[0]].encoder

    @property
    def decoder(self):
        return self.models[self.keys[0]].decoder
    '''


class FairseqFactoredMultiModel(BaseFairseqModel):
    """Base class for combining multiple encoder-decoder models."""
    def __init__(self, encoders, decoder):
        #super().__init__(FactoredCompositeEncoder(encoders),decoder)
        super().__init__()
        self.encoder = FactoredCompositeEncoder(encoders)
        self.decoder = decoder
        #assert encoders.keys() == decoders.keys()
        self.keys = list(encoders.keys())
        for key in self.keys:
            assert isinstance(encoders[key], FairseqEncoder)
            #assert isinstance(decoders[key], FairseqDecoder)
        assert isinstance(decoder, FairseqDecoder)
        '''
        self.models = nn.ModuleDict({
            key: FairseqModel(encoders[key], decoders[key])
            for key in self.keys
        })
        '''

    @staticmethod
    def build_shared_embeddings(
        dicts: Dict[str, Dictionary],
        langs: List[str],
        embed_dim: int,
        build_embedding: callable,
        pretrained_embed_path: Optional[str] = None,
    ):
        """
        Helper function to build shared embeddings for a set of languages after
        checking that all dicts corresponding to those languages are equivalent.

        Args:
            dicts: Dict of lang_id to its corresponding Dictionary
            langs: languages that we want to share embeddings for
            embed_dim: embedding dimension
            build_embedding: callable function to actually build the embedding
            pretrained_embed_path: Optional path to load pretrained embeddings
        """
        shared_dict = dicts[langs[0]]
        if any(dicts[lang] != shared_dict for lang in langs):
            raise ValueError(
                '--share-*-embeddings requires a joined dictionary: '
                '--share-encoder-embeddings requires a joined source '
                'dictionary, --share-decoder-embeddings requires a joined '
                'target dictionary, and --share-all-embeddings requires a '
                'joint source + target dictionary.'
            )
        return build_embedding(
            shared_dict, embed_dim, pretrained_embed_path
        )

    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        encoder_out = self.encoder(src_tokens, src_lengths)
        #print(src_tokens.size())
        #print(encoder_out['encoder_out'].size())
        #print(encoder_out)
        #''''
        if encoder_out['encoder_padding_mask'] is not None:
            pass
            #print(xx)
            #print(encoder_out)
            #input()
        #'''
        #encoder_out['encoder_padding_mask'] = None # Hack...
        decoder_out = self.decoder(prev_output_tokens, encoder_out)
        #print(decoder_out[0].size())
        #exit()
        return decoder_out
    '''
    def forward2(self, src_tokens, src_lengths, prev_output_tokens):
        decoder_outs = {}
        concat_encoder = None
        for key in self.keys:
            encoder_out = self.models[key].encoder(src_tokens, src_lengths)
            if concat_encoder is None:
                concat_encoder = encoder_out
            else:
                concat_encoder = torch.cat((concat_encoder, encoder_out))
            # decoder_outs[key] = self.models[key].decoder(prev_output_tokens, encoder_out)
        for key in self.keys:
            #return self.models[key].decoder(prev_output_tokens, concat_encoder)
            decoder_outs[key] = self.models[key].decoder(prev_output_tokens, concat_encoder)
        return decoder_outs
    '''
    '''
    def forward3(self, inputs):
        print(inputs)
        decoder_outs = {}
        concat_encoder = None
        for key in self.keys:
            src_tokens = inputs[key][src_tokens]
            src_lengths = inputs[key][src_lengths]
            prev_output_tokens = inputs[key][src_lengths]
            encoder_out = self.models[key].encoder(src_tokens, src_lengths)
            if concat_encoder is None:
                concat_encoder = encoder_out
            else:
                concat_encoder = torch.cat((concat_encoder, encoder_out))
        for key in self.keys:
            #return self.models[key].decoder(prev_output_tokens, concat_encoder)
            decoder_outs[key] = self.models[key].decoder(prev_output_tokens, concat_encoder)
        return decoder_outs
    '''
    def max_positions(self):
        """Maximum length supported by the model."""
        return {
            key: (self.encoder.encoders[key].max_positions(), self.decoder.max_positions())
            for key in self.keys
        }
        '''
        return {
            key: (self.models[key].encoder.max_positions(), self.models[key].decoder.max_positions())
            for key in self.keys
        }
        '''
    '''
    def max_decoder_positions____(self):
        """Maximum length supported by the decoder."""
        return min(model.decoder.max_positions() for model in self.models.values())
    '''
    '''
    @property
    def encoder(self):
        return self.models[self.keys[0]].encoder

    @property
    def decoder(self):
        return self.models[self.keys[0]].decoder
    '''

class FairseqFactoredMultiModel3(BaseFairseqModel):
    """Base class for combining multiple encoder-decoder models."""
    def __init__(self, encoders, decoders):
        super().__init__()
        assert encoders.keys() == decoders.keys()
        self.keys = list(encoders.keys())
        for key in self.keys:
            assert isinstance(encoders[key], FairseqEncoder)
            assert isinstance(decoders[key], FairseqDecoder)

        self.models = nn.ModuleDict({
            key: FairseqModel(encoders[key], decoders[key])
            for key in self.keys
        })

    @staticmethod
    def build_shared_embeddings(
        dicts: Dict[str, Dictionary],
        langs: List[str],
        embed_dim: int,
        build_embedding: callable,
        pretrained_embed_path: Optional[str] = None,
    ):
        """
        Helper function to build shared embeddings for a set of languages after
        checking that all dicts corresponding to those languages are equivalent.

        Args:
            dicts: Dict of lang_id to its corresponding Dictionary
            langs: languages that we want to share embeddings for
            embed_dim: embedding dimension
            build_embedding: callable function to actually build the embedding
            pretrained_embed_path: Optional path to load pretrained embeddings
        """
        shared_dict = dicts[langs[0]]
        if any(dicts[lang] != shared_dict for lang in langs):
            raise ValueError(
                '--share-*-embeddings requires a joined dictionary: '
                '--share-encoder-embeddings requires a joined source '
                'dictionary, --share-decoder-embeddings requires a joined '
                'target dictionary, and --share-all-embeddings requires a '
                'joint source + target dictionary.'
            )
        return build_embedding(
            shared_dict, embed_dim, pretrained_embed_path
        )

    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        '''
        decoder_outs = {}
        for key in self.keys:
            encoder_out = self.models[key].encoder(src_tokens, src_lengths)
            decoder_outs[key] = self.models[key].decoder(prev_output_tokens, encoder_out)
        return decoder_outs
        '''
        concat_encoder = None
        # encoder_out = {}
        for index, key in enumerate(self.encoders):
            encoder_out = self.encoders[key](src_tokens[index], src_lengths)
            # encoder_out[key] = self.encoders[key](src_tokens[index], src_lengths
            if concat_encoder is None:
                concat_encoder = encoder_out  # ['encoder_out']
            else:
                concat = torch.cat((concat_encoder['encoder_out'], encoder_out['encoder_out']))
                concat_encoder['encoder_out'] = concat

    def max_positions(self):
        """Maximum length supported by the model."""
        return {
            key: (self.models[key].encoder.max_positions(), self.models[key].decoder.max_positions())
            for key in self.keys
        }

    def max_decoder_positions(self):
        """Maximum length supported by the decoder."""
        return min(model.decoder.max_positions() for model in self.models.values())

    @property
    def encoder(self):
        return self.models[self.keys[0]].encoder

    @property
    def decoder(self):
        return self.models[self.keys[0]].decoder




class FairseqFactoredOneEncoderModel(BaseFairseqModel):
    """Base class for encoder-decoder models.

    Args:
        encoder (FairseqEncoder): the encoder
        decoder (FairseqDecoder): the decoder
    """

    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        assert isinstance(self.encoder, FairseqEncoder)
        assert isinstance(self.decoder, FairseqDecoder)

    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        """
        Run the forward pass for an encoder-decoder model.

        First feed a batch of source tokens through the encoder. Then, feed the
        encoder output and previous decoder outputs (i.e., input feeding/teacher
        forcing) to the decoder to produce the next outputs::

            encoder_out = self.encoder(src_tokens, src_lengths)
            return self.decoder(prev_output_tokens, encoder_out)

        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing

        Returns:
            the decoder's output, typically of shape `(batch, tgt_len, vocab)`
        """
        encoder_out = self.encoder(src_tokens, src_lengths)
        decoder_out = self.decoder(prev_output_tokens, encoder_out)
        return decoder_out

    @staticmethod
    def build_shared_embeddings(
            dicts: Dict[str, Dictionary],
            langs: List[str],
            embed_dim: int,
            build_embedding: callable,
            pretrained_embed_path: Optional[str] = None,
    ):
        """
        Helper function to build shared embeddings for a set of languages after
        checking that all dicts corresponding to those languages are equivalent.

        Args:
            dicts: Dict of lang_id to its corresponding Dictionary
            langs: languages that we want to share embeddings for
            embed_dim: embedding dimension
            build_embedding: callable function to actually build the embedding
            pretrained_embed_path: Optional path to load pretrained embeddings
        """
        shared_dict = dicts[langs[0]]
        if any(dicts[lang] != shared_dict for lang in langs):
            raise ValueError(
                '--share-*-embeddings requires a joined dictionary: '
                '--share-encoder-embeddings requires a joined source '
                'dictionary, --share-decoder-embeddings requires a joined '
                'target dictionary, and --share-all-embeddings requires a '
                'joint source + target dictionary.'
            )
        return build_embedding(
            shared_dict, embed_dim, pretrained_embed_path
        )

    def max_positions(self):
        """Maximum length supported by the model."""
        return (self.encoder.max_positions(), self.decoder.max_positions())



class FairseqFactoredNewMultiModel(BaseFairseqModel):
    """Base class for combining multiple encoder-decoder models."""
    def __init__(self, encoders, decoder):
        super().__init__()
        #assert encoders.keys() == decoders.keys()
        self.keys = list(encoders.keys())
        for key in self.keys:
            assert isinstance(encoders[key], FairseqEncoder)
        assert isinstance(decoder, FairseqDecoder)
        self.encoders = encoders
        self.decoder = decoder
        '''
        self.models = nn.ModuleDict({
            key: FairseqModel(encoders[key], decoders[key])
            for key in self.keys
        })
        '''

    @staticmethod
    def build_shared_embeddings(
        dicts: Dict[str, Dictionary],
        langs: List[str],
        embed_dim: int,
        build_embedding: callable,
        pretrained_embed_path: Optional[str] = None,
    ):
        """
        Helper function to build shared embeddings for a set of languages after
        checking that all dicts corresponding to those languages are equivalent.

        Args:
            dicts: Dict of lang_id to its corresponding Dictionary
            langs: languages that we want to share embeddings for
            embed_dim: embedding dimension
            build_embedding: callable function to actually build the embedding
            pretrained_embed_path: Optional path to load pretrained embeddings
        """
        shared_dict = dicts[langs[0]]
        if any(dicts[lang] != shared_dict for lang in langs):
            raise ValueError(
                '--share-*-embeddings requires a joined dictionary: '
                '--share-encoder-embeddings requires a joined source '
                'dictionary, --share-decoder-embeddings requires a joined '
                'target dictionary, and --share-all-embeddings requires a '
                'joint source + target dictionary.'
            )
        return build_embedding(
            shared_dict, embed_dim, pretrained_embed_path
        )

    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        total_encoder_out = None
        for key in self.keys:
            encoder_out = self.encoders[key](src_tokens, src_lengths)
            if total_encoder_out is None:
                total_encoder_out['encoder_out'] = torch.cat((total_encoder_out['encoder_out'], encoder_out['encoder_out']))
                if total_encoder_out['encoder_padding_mask'] is not None:
                    total_encoder_out['encoder_padding_mask'] = torch.cat(
                        (total_encoder_out['encoder_padding_mask'], encoder_out['encoder_padding_mask']), 1)
        decoder_out = self.decoder(prev_output_tokens, total_encoder_out)
        return decoder_out

    def max_positions(self):
        """Maximum length supported by the model."""
        return {
            key: (self.encoders[key].max_positions(), self.decoder.max_positions())
            for key in self.keys
        }

    def max_decoder_positions(self):
        """Maximum length supported by the decoder."""
        return self.decoder.max_positions()
    @property
    def encoder(self):
        return self.encoders[self.keys[0]]

    @property
    def decoder(self):
        return self.decoder

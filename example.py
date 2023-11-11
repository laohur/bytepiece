# -*- coding: utf-8 -*-
import os
import json

import bytepiece
import tokenizers
import base64
import numpy as np
from transformers import PreTrainedTokenizerFast, AutoTokenizer


"""
file: bytepiece.model
    "Ag==": [
        5,
        "\u0002",
        38748
    ],
"""


def bp2sp():
    from bytepiece import Tokenizer
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

    tokenizer1 = Tokenizer("bytepiece.model")
    tokenizer1.convert_to_sentencepiece("bytepiece_sp.model")

    import sentencepiece as spm

    tokenizer2 = spm.SentencePieceProcessor("bytepiece_sp.model")

    tokenizer1.encode("今天天气不错")
    tokenizer2.encode("今天天气不错")


# https://github.com/bojone/bytepiece/issues/7
def bp2vocab(bp_model_file):
    pieces = json.load(open(bp_model_file))
    pieces = {base64.b64decode(k): v for k, v in pieces.items()}
    log_total = np.log(sum(v[2] for v in pieces.values()))

    vocab = [("<unk>", 0), ("<s>", 1), ("</s>", 2), ("<pad>", 3)]
    vocab += [(f"<0x{p:02X}>", 0.0) for p in range(256)]
    for k, v in pieces.items():
        if len(k) == 1:
            word = chr(k[0])
        else:
            word = k.decode("utf-8")
        pair = (word, np.log(v[2]) - log_total)
        vocab.append(pair)
    return vocab


# https://github.com/huggingface/tokenizers/blob/main/bindings/python/py_src/tokenizers/implementations/sentencepiece_unigram.py#L146
def init_tokenizer(hf_save_dir=None, vocab=[("<unk>", 0)]):
    model = tokenizers.models.Unigram(vocab, 0, True)
    tokenizer = tokenizers.Tokenizer(model)
    tokenizer.normalizer = tokenizers.normalizers.Sequence(
        [
            tokenizers.normalizers.NFC(),
        ]
    )
    pre_tokenizer = tokenizers.pre_tokenizers.Sequence(
        [
            # pre_tokenizers.Split(" ", "isolated"),
            # pre_tokenizers.Whitespace(),
            # pre_tokenizers.PreTokenizer.custom(LanguagePreTokenizer()),
            # pre_tokenizers.UnicodeScripts(),
            # pre_tokenizers.Split(" ?[^(\\s|[.,!?…。，、।۔،])]+", "isolated"),
            # tokenizers.pre_tokenizers.CharDelimiterSplit("\x00"),
            # tokenizers.pre_tokenizers.Punctuation(behavior="contiguous"),
            # tokenizers.pre_tokenizers.Digits(individual_digits=True),
            # pre_tokenizers.Metaspace(add_prefix_space=False),
            # pre_tokenizers.ByteLevel(add_prefix_space=False,use_regex=False)
            tokenizers.pre_tokenizers.ByteLevel(replace_ment="\x00", add_prefix_space=False),
        ]
    )
    tokenizer.pre_tokenizer = pre_tokenizer
    # tokenizer.post_processor = tokenizers.processors.ByteLevel(trim_offsets=False, add_prefix_space=False)
    # tokenizer.decoder = tokenizers.decoders.ByteLevel(add_prefix_space=False)

    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        # cls_token="<cls>",
        # sep_token="<sep>",
        # mask_token="<mask>",
        padding_side="left",
        clean_up_tokenization_spaces=False,
    )
    if hf_save_dir:
        wrapped_tokenizer.save_pretrained(hf_save_dir)
    return wrapped_tokenizer


def bp2hf(bp_model_file, hf_save_dir):
    vocab = bp2vocab(bp_model_file)
    tokenizer = init_tokenizer(hf_save_dir, vocab)
    print(tokenizer.encode("今天天气不错"))

    tokenizer = AutoTokenizer.from_pretrained(hf_save_dir, byte_fallback=True)
    print(tokenizer.encode("今天天气不错"))
    tokenizer.save_pretrained(hf_save_dir)


# https://www.huaxiaozhuan.com/%E5%B7%A5%E5%85%B7/huggingface_transformer/chapters/1_tokenizer.html
def train():
    baseTokenizer = init_tokenizer().backend_tokenizer
    normalizer = baseTokenizer.normalizer
    preTokenizer = baseTokenizer.pre_tokenizer

    class corpus:
        def __iter__(self):
            for l in open("data_sample.json"):
                text = json.loads(l)["text"]
                # normalized_str = normalizer.normalize_str(text)
                for word, (start, end) in preTokenizer.pre_tokenize_str(text):
                    yield word

    from bytepiece import Trainer

    trainer = Trainer(order=6, max_vocab_size=65536, min_count=32)
    trainer.train(corpus(), workers=64, batch_size=1000)
    trainer.save("bytepiece.model")


if __name__ == "__main__":
    train()
    bp2hf("bytepiece.model", "hf_save_dir")
    bp2sp()

# coding=utf-8
# Copyright 2021 Artem Ploujnikov


# Lint as: python3
import json

import datasets

_DESCRIPTION = """\
Grapheme-to-Phoneme training, validation and test sets
"""

_BASE_URL = "https://raw.githubusercontent.com/flexthink/librig2p-nostress-space/develop/dataset"


_HOMEPAGE_URL = "https://github.com/flexthink/librig2p-nostress-space/tree/develop"

_PHONEMES = [
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",
    "B",
    "CH",
    "D",
    "DH",
    "EH",
    "ER",
    "EY",
    "F",
    "G",
    "HH",
    "IH",
    "IY",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "OW",
    "OY",
    "P",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "UH",
    "UW",
    "V",
    "W",
    "Y",
    "Z",
    "ZH",
    " "
]
_ORIGINS = ["librispeech", "librispeech-lex", "wikipedia-homograph"]
_NA = "N/A"
_SPLIT_TYPES = ["train", "valid", "test"]
_DATA_TYPES = ["lexicon", "sentence", "homograph"]
_SPLITS = [
    f"{data_type}_{split_type}"
    for data_type in _DATA_TYPES
    for split_type in _SPLIT_TYPES]

class GraphemeToPhoneme(datasets.GeneratorBasedBuilder):
    def __init__(self, base_url=None, splits=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_url = base_url or _BASE_URL
        self.splits = splits or _SPLITS

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "speaker_id": datasets.Value("string"),
                    "origin": datasets.ClassLabel(names=_ORIGINS),
                    "char": datasets.Value("string"),
                    "phn": datasets.Sequence(datasets.ClassLabel(names=_PHONEMES)),
                    "homograph": datasets.Value("string"),
                    "homograph_wordid": datasets.Value("string")
                },
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE_URL,
        )

    def _get_url(self, split):
        return f'{self.base_url}/{split}.json'

    def _split_generator(self, dl_manager, split):
        url = self._get_url(split)
        path = dl_manager.download_and_extract(url)
        return datasets.SplitGenerator(
            name=split,
            gen_kwargs={"datapath": path, "datatype": split},
        )

    def _split_generators(self, dl_manager):
        return [
            self._split_generator(dl_manager, split)
            for split in self.splits
        ]

    def _generate_examples(self, datapath, datatype):
        with open(datapath, encoding="utf-8") as f:
            data = json.load(f)

        for sentence_counter, (item_id, item) in enumerate(data.items()):
            resp = {
                "id": item_id,
                "speaker_id": str(item.get("speaker_id") or _NA),
                "homograph": item.get("homograph", _NA),
                "homograph_wordid": item.get("homograph_wordid", _NA),
                "origin": item["origin"],
                "char": item["char"],
                "phn": item["phn"],
            }
            yield sentence_counter, resp

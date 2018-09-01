from typing import Dict
import json
import logging

from overrides import overrides

import tqdm

from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

from sklearn.datasets import fetch_20newsgroups

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("20newsgroups")
class NewsgroupsDatasetReader(DatasetReader):
    """
    Reads a JSON-lines file containing papers from the Semantic Scholar database, and creates a
    dataset suitable for document classification using these papers.

    Expected format for each input line: {"paperAbstract": "text", "title": "text", "venue": "text"}

    The JSON could have other fields, too, but they are ignored.

    The output of ``read`` is a list of ``Instance`` s with the fields:
        title: ``TextField``
        abstract: ``TextField``
        label: ``LabelField``

    where the ``label`` is derived from the venue of the paper.

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the title and abstrct into words or other kinds of tokens.
        Defaults to ``WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input token representations. Defaults to ``{"tokens":
        SingleIdTokenIndexer()}``.
    """
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        instances = []
        if file_path == "train":
            logger.info("Reading instances from: %s", file_path)
            categories = ["comp.graphics","sci.space","rec.sport.baseball"]
            newsgroups_data = fetch_20newsgroups(subset='train',categories=categories)
            
        elif file_path == "test":
            logger.info("Reading instances from: %s", file_path)
            categories = ["comp.graphics","sci.space","rec.sport.baseball"]
            newsgroups_data = fetch_20newsgroups(subset='test',categories=categories)
            
        else:
            raise ConfigurationError("Path string not specified in read method")
            
        for i,text in enumerate(newsgroups_data.data):
            if file_path == "validate":
                if i == 400:
                    break
            text = newsgroups_data.data[i]
            target = newsgroups_data.target[i]
            yield self.text_to_instance(text, target)

    @overrides
    def text_to_instance(self, text: str, target: str = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokenized_text = self._tokenizer.tokenize(text)
        text_field = TextField(tokenized_text, self._token_indexers)
        fields = {'text': text_field}
        if target is not None:
            fields['label'] = LabelField(int(target),skip_indexing=True)
        return Instance(fields)



# Training a deep learning model with AllenNLP

In this tutorial we’ll use the 20 newsgroups provided by scikit-learn. For more details check out [this article](https://medium.com/swlh/deep-learning-for-text-made-easy-with-allennlp-62bc79d41f31) 😉

With AllenNLP we define the model architecture in a JSON file ([experiments/newsgroups_without_cuda.json](https://github.com/dmesquita/easy-deep-learning-with-AllenNLP/blob/master/experiments/newsgroups_without_cuda.json)). This ``Model`` performs text classification for the newsgroup files. The basic model structure: we'll embed the text and encode it with a Seq2VecEncoder.  We'll then pass the result through a feedforward network, the output of which we'll use as our scores for each label.

## 1 —Data inputs
To set the input dataset and how to read from it we use the ``'dataset_reader'`` key in the JSON file. We specify how to read the data [here](https://github.com/dmesquita/easy-deep-learning-with-AllenNLP/blob/2b2cf0176404346f7713d72fd34f78f645f6d7cf/newsgroups/dataset_readers/fetch_newsgroups.py#L55) by creating a [``DatasetReader`` class](https://github.com/dmesquita/easy-deep-learning-with-AllenNLP/blob/master/newsgroups/dataset_readers/fetch_newsgroups.py)

## 2 — The model
To specify the model we’ll set the ``'model'`` key. There are three more parameters inside: ``'model_text_field_embedder'``, ``'internal_text_encoder'`` and ``'classifier_feedforward'``. The internals of the model is defined in the [``Fetch20NewsgroupsClassifier`` class](https://github.com/dmesquita/easy-deep-learning-with-AllenNLP/blob/master/newsgroups/models/newsgroups_classifier.py)

## 3 — The data iterator
AllenNLP provides an iterator called BucketIterator that makes the computations (padding) more efficient by padding batches with respect to the maximum input lengths per batch. To do that it sorts the instances by the number of tokens in each text. We set these parameters in the ``'iterator'`` key of the JSON file.

## 4 — Training the model
The trainer uses the AdaGrad optimizer for 30 epochs, stopping if validation accuracy has not increased for the last 3 epochs. This is also specified in the [JSON file](https://github.com/dmesquita/easy-deep-learning-with-AllenNLP/blob/master/experiments/newsgroups_without_cuda.json).

To train the model locally we need to run this:

```python3 run.py train experiments/newsgroups_without_cuda.json  --include-package newsgroups.dataset_readers --include-package newsgroups.models```


### Train the model: colaboratory notebook
[https://colab.research.google.com/drive/1q3b5HAkcjYsVd6yhrwnxL2ByqGK08jhQ](https://colab.research.google.com/drive/1q3b5HAkcjYsVd6yhrwnxL2ByqGK08jhQ)





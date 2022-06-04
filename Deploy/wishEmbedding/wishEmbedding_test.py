"""wishEmbedding dataset."""

import tensorflow_datasets as tfds
from . import wishEmbedding


class WishembeddingTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for wishEmbedding dataset."""
  DATASET_CLASS = wishEmbedding.Wishembedding
  SPLITS = {
      'train': 3,  # Number of fake train example
      'test': 1,  # Number of fake test example
  }


if __name__ == '__main__':
  tfds.testing.test_main()

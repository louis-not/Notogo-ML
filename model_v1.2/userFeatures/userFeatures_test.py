<<<<<<< HEAD
"""userFeatures dataset."""

import tensorflow_datasets as tfds
from . import userFeatures


class UserfeaturesTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for userFeatures dataset."""
  # TODO(userFeatures):
  DATASET_CLASS = userFeatures.Userfeatures
  SPLITS = {
      'train': 3,  # Number of fake train example
      'test': 1,  # Number of fake test example
  }

  # If you are calling `download/download_and_extract` with a dict, like:
  #   dl_manager.download({'some_key': 'http://a.org/out.txt', ...})
  # then the tests needs to provide the fake output paths relative to the
  # fake data directory
  # DL_EXTRACT_RESULT = {'some_key': 'output_file1.txt', ...}


if __name__ == '__main__':
  tfds.testing.test_main()
=======
"""userFeatures dataset."""

import tensorflow_datasets as tfds
from . import userFeatures


class UserfeaturesTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for userFeatures dataset."""
  # TODO(userFeatures):
  DATASET_CLASS = userFeatures.Userfeatures
  SPLITS = {
      'train': 3,  # Number of fake train example
      'test': 1,  # Number of fake test example
  }

  # If you are calling `download/download_and_extract` with a dict, like:
  #   dl_manager.download({'some_key': 'http://a.org/out.txt', ...})
  # then the tests needs to provide the fake output paths relative to the
  # fake data directory
  # DL_EXTRACT_RESULT = {'some_key': 'output_file1.txt', ...}


if __name__ == '__main__':
  tfds.testing.test_main()
>>>>>>> fe385ef1b97c4ff5f39f6cb3fc228f2fe62857e7

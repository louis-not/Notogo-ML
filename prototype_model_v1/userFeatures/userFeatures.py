"""my_dataset dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
import os
import csv

# TODO(my_dataset): Markdown description  that will appear on the catalog page.
_DESCRIPTION = None
"""
# Description is **formatted** as markdown.

# It should also contain any processing which has been applied (if any),
# (e.g. corrupted example skipped, images cropped,...):
# """

# TODO(my_dataset): BibTeX citation
# _CITATION = """
# """


class Userfeatures(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for my_dataset dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(my_dataset): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder= self,
        description= _DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'user_id': tf.string,
            'location_name' : tf.string,
            'location_id' : tf.string
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=None,  # Set to `None` to disable
        homepage=None,
        citation=None
    )

  # def _split_generators(self, dl_manager: tfds.download.DownloadManager):
  #   """Returns SplitGenerators."""
  #   # TODO(my_dataset): Downloads the data and defines the splits
  #   # path = dl_manager.download_and_extract('https://todo-data-url')

  #   # TODO(my_dataset): Returns the Dict[split names, Iterator[Key, Example]]
  #   return {
  #       'train': self._generate_examples(path / 'train_imgs'),
  #   }
  def _split_generators(self, dl_manager):
  # Download source data
  # extracted_path = dl_manager.download_and_extract(...)
  # Specify the splits
    pathf = '/content/drive/Othercomputers/My Laptop/Bangkit/Capstone/Recommender system/userFeatures'
    return {
        'train': self._generate_examples(
            path= os.path.join(pathf,'userFeaturesV2.csv')
            # label_path=extracted_path / 'train_labels.csv',
        ),
        # 'test': self._generate_examples(
        #     path= os.path.join(pathf,'rating.csv')
        #     # label_path=extracted_path / 'test_labels.csv',
        # ),
    }

  def _generate_examples(self, path):
    """Yields examples."""
    
    # TODO(my_dataset): Yields (key, example) tuples from the dataset
    # for f in path.glob('*.jpeg'):
    with open(path) as csv_file:
      csv_reader = csv.reader(csv_file)
      for i,row in enumerate(csv_reader):
        yield i, {
            'user_id': row[0],
            'location_name' : row[1], # -> update V2
            'location_id': row[2]
        }

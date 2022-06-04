"""my_dataset dataset."""

import tensorflow as tf
import pandas as pd
import mysql.connector
import tensorflow_datasets as tfds


_DESCRIPTION = None


class Wishembedding(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for Wishembedding dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return tfds.core.DatasetInfo(
      builder= self,
      description= _DESCRIPTION,
      features=tfds.features.FeaturesDict({
          # These are the features of your dataset like images, labels ...
          'location_id' : tf.string,
          'location_name' : tf.string
      }),
      supervised_keys=None,  # Set to `None` to disable
      homepage=None,
      citation=None
    )

  def _split_generators(self, dl_manager):
  # Download source data
    cnx = mysql.connector.connect(user = 'root', password = '1234', host = '34.101.251.5', database = 'notogo')
    cursor = cnx.cursor()
    query = ("select * from wish_embedding")
    cursor.execute(query)
    dfWishEmbedding = pd.DataFrame(cursor.fetchall())
    dfWishEmbedding = dfWishEmbedding.values
    return {
      'train': self._generate_examples(
          dfWishEmbedding
          # label_path=extracted_path / 'train_labels.csv',
      ),
    }

  def _generate_examples(self, path):
    """Yields examples."""
    for i, data in enumerate(path):
      yield i, {
          'location_id' : data[0],
          'location_name': data[1]
      }

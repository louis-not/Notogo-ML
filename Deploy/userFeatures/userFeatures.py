"""my_dataset dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
import csv
import pandas as pd
import sqlite3
# from google.cloud import storage
from datetime import datetime
import mysql.connector

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
            'like': tf.int32,
            'add' : tf.int32,
            'category' : tf.string,
            'location_id' : tf.string
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=None,  # Set to `None` to disable
        homepage=None,
        citation=None
    )

  def _split_generators(self, dl_manager):
    cnx = mysql.connector.connect(user = 'root', password = '1234', host = '34.101.251.5', database = 'notogo')
    cursor = cnx.cursor()

    query = ("select * from user_features")
    cursor.execute(query)

    columns =['id','user_id', 'like', 'add', 'category', 'location_name','location_id']
    dfUserFeat = pd.DataFrame(cursor.fetchall(), columns = columns)

    dfUserFeat['location_id'] = dfUserFeat['location_id'].astype(str)
    dfUserFeat['like'] = dfUserFeat['like'].astype(int)
    dfUserFeat['add'] = dfUserFeat['add'].astype(int)
    dfUserFeat['user_id'] = dfUserFeat['user_id'].astype(str)

    dfFeature = dfUserFeat[['user_id',"like", 'add','category','location_name','location_id']].values
    return {
        'train': self._generate_examples(dfFeature),
        # 'test': self._generate_examples(
        #     path= os.path.join(pathf,'rating.csv')
        #     # label_path=extracted_path / 'test_labels.csv',
        # ),
    }

  def _generate_examples(self, path):
    """Yields examples."""
    
    # TODO(my_dataset): Yields (key, example) tuples from the dataset
    for i, data in enumerate(path):
      yield i, {
          'user_id': data[1],
          'like' : int(data[2]),
          'add' : int(data[3]),
          'category' : data[4],
          'location_id': data[6]
      }

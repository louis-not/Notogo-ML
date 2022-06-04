"""my_dataset dataset."""

import tensorflow as tf
import pandas as pd
import mysql.connector
import tensorflow_datasets as tfds


_DESCRIPTION = None


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
      supervised_keys=None,  # Set to `None` to disable
      homepage=None,
      citation=None
    )

  def _split_generators(self, dl_manager):

    # import gspread
    # from google.auth import default
    # from gspread_dataframe import get_as_dataframe, set_with_dataframe

    # creds, _ = default()
    # gc = gspread.authorize(creds)
    # feature = gc.open('capstone_dataset').worksheet('userFeatures(coldstartsol)')
    # rows = feature.get_all_values()
    # dfFeature = pd.DataFrame.from_records(rows[1:], columns=rows[0])
    # dfFeature = dfFeature[['user_id',"like", 'add','category','location_id']].values

    cnx = mysql.connector.connect(user = 'root', password = '1234', host = '34.101.251.5', database = 'notogo')
    cursor = cnx.cursor()

    query = ("select * from user_features")
    cursor.execute(query)

    columns =['user_id', 'like', 'add', 'category','location_id']
    dfUserFeat = pd.DataFrame(cursor.fetchall(), columns = columns)

    dfUserFeat['location_id'] = dfUserFeat['location_id'].astype(str)
    dfUserFeat['user_id'] = dfUserFeat['user_id'].astype(str)

    dfFeature = dfUserFeat[['user_id',"like", 'add','category','location_id']].values

    return {
        'train': self._generate_examples(
          dfFeature
            # label_path=extracted_path / 'train_labels.csv',
        )
    }

  def _generate_examples(self, path):
    """Yields examples."""
    for i, data in enumerate(path):
      yield i, {
          'user_id': data[0],
          'like' : int(data[1]),
          'add' : int(data[2]),
          'category' : data[3],
          'location_id': data[5]
      }

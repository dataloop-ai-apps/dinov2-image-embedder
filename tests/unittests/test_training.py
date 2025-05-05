
import unittest
import os
import dtlpy as dl
from PIL import Image
from adapter import DINOv2Adapter


class TestModelAdapter(unittest.TestCase):

    def test_train_model(self):
        dl.setenv('prod')
        adapter = DINOv2Adapter()
        model_entity = dl.models.get(model_id="")

        model_entity.dataset_id = ""

        model_entity.metadata['system'] = {}
        model_entity.metadata['system']['subsets'] = {'train': dl.Filters(field='metadata.system.tags.train', values=True).prepare(),
                                                    'validation': dl.Filters(field='metadata.system.tags.validation', values=True).prepare()}
        model_entity.status = 'pre-trained'
        model_entity.update(True)

        print('started the training')
        adapter.train_model(model_entity)


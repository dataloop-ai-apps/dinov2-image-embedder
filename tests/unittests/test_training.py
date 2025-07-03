import unittest
import os
import dtlpy as dl
from PIL import Image
import sys
import logging

# Configure logging to show progress
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from adapter import DINOv2Adapter


class TestModelAdapter(unittest.TestCase):

    def test_train_model(self):
        # Set up logger for this test
        logger = logging.getLogger("[TRAINING TEST]")
        logger.info("Starting DINOv2 training test...")
        
        dl.setenv('rc')
        # dl.login()
        adapter = DINOv2Adapter()
        model_entity = dl.models.get(model_id="")

        model_entity.dataset_id = ""
        
        # # Add test-friendly configuration
        # if not hasattr(model_entity, 'configuration') or model_entity.configuration is None:
        #     model_entity.configuration = {}
        
        # # Use smaller parameters for testing
        # model_entity.configuration.update({
        #     'num_epochs': 10,
        #     'batch_size': 4,
        #     'learning_rate': 1e-4,
        #     'save_interval': 10
        # })

        model_entity.metadata['system'] = {}
        model_entity.metadata['system']['subsets'] = {'train': dl.Filters(field='metadata.system.tags.train', values=True).prepare(),
                                                    'validation': dl.Filters(field='metadata.system.tags.validation', values=True).prepare()}
        model_entity.status = 'pre-trained'
        model_entity.update(True)

        logger.info('Starting the training process...')
        adapter.train_model(model_entity)
        logger.info('Training completed successfully!')

if __name__ == '__main__':
    test = TestModelAdapter()
    test.test_train_model()
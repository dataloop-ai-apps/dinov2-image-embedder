import unittest
import os

from PIL import Image
from adapter import DINOv2Adapter


class TestModelAdapter(unittest.TestCase):

    def test_embed_image(self):
        adapter = DINOv2Adapter()
        adapter.load(local_path="")
        image = Image.open(os.path.join(os.path.abspath("../../tests/assets/unittests/image.png")))
        features = adapter.embed_images([image])
        assert len(features[0]) == 384


if __name__ == "__main__":
    unittest.main()

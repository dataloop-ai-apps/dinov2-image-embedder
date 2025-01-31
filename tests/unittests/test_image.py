import unittest
import dtlpy as dl
import os
print(os.getcwd())

from adapter import DINOv2Adapter
from PIL import Image


class TestModelAdapter(unittest.TestCase):

    def test_embed_image(self):
        adapter = DINOv2Adapter()
        adapter.load(local_path="")
        image = Image.open(os.path.join("assets", "image.jpg"))
        features = adapter.embed_images([image])
        print(features)


if __name__ == "__main__":
    unittest.main()

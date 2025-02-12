import dtlpy as dl
import torchvision.transforms as transforms
import traceback
import logging
import torch
from PIL import Image, ImageFile
from transformers import CLIPProcessor, CLIPModel
from typing import List

logger = logging.getLogger("[DINOV2-ADAPTER]")
ImageFile.LOAD_TRUNCATED_IMAGES = True


class DINOv2Adapter(dl.BaseModelAdapter):
    @staticmethod
    def load_dinov2_model(model_name) -> torch.nn.Module:
        """
        Load a DINOv2 model from torch hub.

        Args:
            model_name: One of ['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14']
        Returns:
            Loaded model
        """
        try:
            model = torch.hub.load("facebookresearch/dinov2", model_name)
            model.eval()
            return model
        except Exception as e:
            raise Exception(f"Failed to load model {model_name}: {str(e)}")

    def load(self, local_path, **kwargs):
        if self._model_entity is not None:
            model_name = self.model_entity.configuration.get("model_name")
        else:
            model_name = "dinov2_vits14"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.load_dinov2_model(model_name)
        self.model.to(self.device)
        self.configuration["embeddings_size"] = self.configuration.get(
            "embeddings_size", 384
        )

    def preprocess_image(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Preprocess an image for DINOv2 inference.

        Args:
            images: List of images
        Returns:
            Preprocessed image tensor
        """
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        try:
            if not images.mode == "RGB":
                images = images.convert("RGB")
            image_tensor = transform(images).unsqueeze(0)
            image_tensor = image_tensor.to(self.device)
            return image_tensor
        except Exception as e:
            raise Exception(f"Failed to preprocess image: {traceback.format_exc()}")

    def embed_images(self, images: List[Image.Image]):
        # Preprocess for Meta clip
        batch_tensors = [self.preprocess_image(image) for image in images]
        batch = torch.cat(batch_tensors)

        with torch.no_grad():
            batch_embeddings = self.model(batch)

        return batch_embeddings.cpu().numpy().tolist()

    def embed(self, batch: List[dl.Item], **kwargs):
        features = []
        # TODO: batch processing
        for item in batch:
            if "image" in item.mimetype:
                orig_image = Image.fromarray(
                    item.download(save_locally=False, to_array=True)
                )
                image_features = self.embed_images([orig_image])
                features.extend(image_features)
            else:
                features.append(None)
        return features

    def prepare_item_func(self, item: dl.Item):
        return item


if __name__ == "__main__":
    model = dl.models.get(model_id="67a8a11fdefe6a0c7bf1990b")
    adapter = DINOv2Adapter(model_entity=model)
    filters = dl.Filters()
    filters.add(
        field="id",
        values=[
            "67a484cb6336b7145d8b967d",
            "67a22d8a7e61fdd99381f343",
            "67a21e4cd594d4bfd3925076",
            "676d4d2ae5e4f32de26f7a06",
            "676d4d2a07bcd5407839b9a3",
            "676d4d29e5e4f35e926f79ff",
        ],
        operator=dl.FiltersOperations.IN,
    )
    adapter.embed_dataset(
        dataset=dl.datasets.get(dataset_id="676d4cb7638f92c8253f7649"), filters=filters
    )

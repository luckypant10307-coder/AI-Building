"""Damp, mould and surface-defect classifier (component 7).

Unlike everything else in this package, this component is trained on real
photographs of real buildings (BD3), not on the simulator. It is a classifier
rather than a detector because BD3 and the Roboflow wall-damage set both carry
image-level labels only -- there are no bounding boxes to fine-tune on.

Its job in the system: the photograph confirms what the sensors predicted days
earlier. The sensors say a wall is wetting; the camera says whether a stain or
algae has actually appeared on it yet.
"""

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# BD3 ships one folder per class. Folder names vary in case, so compare lowered.
CLASSES = ["algae", "major crack", "minor crack", "normal",
           "peeling", "spalling", "stain"]

# The classes that mean water: biological growth and water marks. These are what
# the sensor side is predicting when it forecasts days-to-visible-damp.
DAMP_CLASSES = ("algae", "stain")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalise(name):
    return name.strip().lower()


def build_model(num_classes=len(CLASSES), pretrained=True):
    """ResNet18 with an ImageNet backbone and a fresh head.

    BD3's own paper benchmarks ResNet18 at 0.8301 F1 on the original split,
    which is the number this is trying to be comparable with. ViT scores higher
    (0.9342) but is far slower on a CPU, and this has to run in the Flask
    backend beside everything else.
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    try:
        model = models.resnet18(weights=weights)
    except Exception:
        # no network for the weight download: say so rather than silently
        # training from scratch, which would look like a much worse model
        raise RuntimeError(
            "could not fetch ImageNet weights; rerun with pretrained=False "
            "only if you intend to train from random initialisation")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_transforms(size=224):
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transforms(size=224):
    return transforms.Compose([
        transforms.Resize(int(size * 1.14)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class DefectClassifier:
    """A trained classifier, loaded for inference."""

    def __init__(self, model, classes, size=224, device="cpu"):
        self.model = model.eval().to(device)
        self.classes = classes
        self.device = device
        self.tf = eval_transforms(size)

    @classmethod
    def load(cls, checkpoint, device="cpu"):
        blob = torch.load(Path(checkpoint), map_location=device, weights_only=False)
        model = build_model(num_classes=len(blob["classes"]), pretrained=False)
        model.load_state_dict(blob["state_dict"])
        return cls(model, blob["classes"], blob.get("size", 224), device)

    def predict_array(self, bgr):
        """Same as predict, for an OpenCV BGR array already in memory."""
        return self._predict(Image.fromarray(bgr[:, :, ::-1]).convert("RGB"))

    def predict(self, image_path):
        """Class probabilities for one photograph, plus a damp summary."""
        return self._predict(Image.open(image_path).convert("RGB"))

    @torch.no_grad()
    def _predict(self, image):
        batch = self.tf(image).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.model(batch), dim=1)[0]
        by_class = {name: float(probs[i]) for i, name in enumerate(self.classes)}
        damp = sum(p for name, p in by_class.items()
                   if normalise(name) in DAMP_CLASSES)
        top = max(by_class, key=by_class.get)
        return {
            "top_class": top,
            "top_prob": by_class[top],
            "damp_probability": damp,
            "by_class": by_class,
        }


def confirms_damp(prediction, threshold=0.5):
    """Does this photograph corroborate a damp alert from the sensors?"""
    return prediction["damp_probability"] >= threshold

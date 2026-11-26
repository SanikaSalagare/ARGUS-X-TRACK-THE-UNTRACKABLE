"""Neural appearance embedder (optional). MobileNetV3-small ImageNet features, L2-normalised.
NOTE: general image features, not a person-ReID-trained network. Tune the similarity threshold on your own clip."""
import cv2
import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class NeuralEmbedder:
    def __init__(self, device):
        import torch
        import torchvision
        self.torch = torch
        self.dev = torch.device("cuda:0" if device == 0 and torch.cuda.is_available() else "cpu")
        m = torchvision.models.mobilenet_v3_small(weights="DEFAULT")
        self.net = torch.nn.Sequential(m.features, torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten()).eval().to(self.dev)

    def __call__(self, crop_bgr):
        img = cv2.cvtColor(cv2.resize(crop_bgr, (64, 128)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        t = self.torch.from_numpy(((img - MEAN) / STD).transpose(2, 0, 1)).unsqueeze(0).to(self.dev)
        with self.torch.no_grad():
            v = self.net(t)[0].cpu().numpy()
        n = np.linalg.norm(v)
        return v / n if n > 0 else None

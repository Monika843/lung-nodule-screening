"""
Grad-CAM implementation.

Highlights which regions of the input image most influenced the model's
prediction. This is NOT proof of disease -- it visualizes model attention,
which the dashboard states explicitly next to every heatmap.
"""
import torch
import torch.nn.functional as F
import numpy as np


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer

    def generate(self, input_tensor: torch.Tensor, class_idx: int = None):
        """
        Registers hooks, runs a forward+backward pass, then removes the
        hooks again before returning. This keeps the model "clean" for any
        later calls elsewhere (e.g. plain torch.no_grad() inference on
        another page of the app) -- a persistent hook would otherwise try
        to register a gradient hook on a tensor that has no gradient in a
        no_grad() context and crash.
        """
        activations = {}
        gradients = {}

        def forward_hook(module, inp, out):
            activations["value"] = out
            if out.requires_grad:
                out.register_hook(lambda grad: gradients.__setitem__("value", grad.detach()))

        handle = self.target_layer.register_forward_hook(forward_hook)

        try:
            self.model.eval()
            # Make sure this forward pass tracks gradients even if it happens
            # to be called from inside a torch.no_grad() block upstream.
            with torch.enable_grad():
                input_tensor = input_tensor.clone().requires_grad_(True)
                output = self.model(input_tensor)
                if class_idx is None:
                    class_idx = output.argmax(dim=1).item()

                self.model.zero_grad()
                score = output[0, class_idx]
                score.backward()

            grads = gradients["value"][0]           # (C, H, W)
            acts = activations["value"][0].detach()  # (C, H, W)
            weights = grads.mean(dim=(1, 2))          # (C,)

            cam = torch.zeros(acts.shape[1:], dtype=torch.float32)
            for i, w in enumerate(weights):
                cam += w * acts[i]

            cam = F.relu(cam)
            cam = cam / (cam.max() + 1e-8)
            cam = cam.cpu().numpy()

            input_h, input_w = input_tensor.shape[-2:]
            cam_resized = np.array(
                torch.nn.functional.interpolate(
                    torch.tensor(cam)[None, None, :, :],
                    size=(input_h, input_w), mode="bilinear", align_corners=False
                )[0, 0]
            )
            probs = F.softmax(output, dim=1)[0].detach().cpu().numpy()
            return cam_resized, class_idx, probs
        finally:
            handle.remove()

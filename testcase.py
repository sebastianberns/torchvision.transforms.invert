import torch
import torchvision.transforms as transforms
import unittest
import numpy as np
from PIL import Image
try:
    import accimage
except ImportError:
    accimage = None

from invert import Invert


class Tester(unittest.TestCase):

    def test_invert(self):
        mode = "L"
        size = (4, 4)
        invert = transforms.Compose([
            Invert(),
            transforms.ToTensor()
        ])
        convert = transforms.ToTensor()

        a = Image.fromarray(np.eye(size[0], size[1], dtype=np.uint8), mode=mode)

        # Invert L
        img = Image.fromarray(np.arange(0, 255, 16, dtype=np.uint8).reshape(size), mode=mode)
        inv = Image.fromarray(np.arange(255, 0, -16, dtype=np.uint8).reshape(size), mode=mode)
        assert torch.equal(invert(img), convert(inv))

        # Invert LA
        img.putalpha(a)
        inv.putalpha(a)
        assert torch.equal(invert(img), convert(inv))

        # Invert RGB
        r = Image.fromarray(np.arange(0, 255, 16, dtype=np.uint8).reshape(size), mode=mode)
        g = Image.fromarray(np.arange(255, 0, -16, dtype=np.uint8).reshape(size), mode=mode)
        b = Image.fromarray(np.arange(127, 0, -8, dtype=np.uint8).reshape(size), mode=mode)
        img = Image.merge('RGB', (r, g, b))
        r = Image.fromarray(np.arange(255, 0, -16, dtype=np.uint8).reshape(size), mode=mode)
        g = Image.fromarray(np.arange(0, 255, 16, dtype=np.uint8).reshape(size), mode=mode)
        b = Image.fromarray(np.arange(128, 255, 8, dtype=np.uint8).reshape(size), mode=mode)
        inv = Image.merge('RGB', (r, g, b))
        assert torch.equal(invert(img), convert(inv))

        # Invert RGBA
        img.putalpha(a)
        inv.putalpha(a)
        assert torch.equal(invert(img), convert(inv))

        # Checking if Invert can be printed as string
        Invert().__repr__()


if __name__ == '__main__':
    unittest.main()

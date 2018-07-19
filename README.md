# torchvision transform: Invert

Inverts the color channels of an PIL Image while leaving intact the alpha channel.

This transform is useful for data preprocessing in any case where the information of interest in a pixel image is placed on a white background. We want to invert the image (and a black background) since a convolution layer by default adds zeros all around as padding.

Can be easily integrated into a torchvision transform composition:

```python
import torchvision.transforms as transforms
from invert import Invert

invert = transforms.Compose([
    Invert(),
    transforms.ToTensor()
])
```

[Simple demo](demo.ipynb)
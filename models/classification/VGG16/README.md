# ImageLogViewer VGG16 model

In order for the classification feature of ImageLogViewer to work, there must be a file named `model.pt` in this directory. This file should contain the trained VGG16 model.

If you don't have a trained model, you can download a pre-trained VGG16 model from torchvision and adapt it for the classification task in ImageLogViewer. Follow these steps:

1. Run the following Python code in ImageLogViewer's root directory:

```python
import torch

from models.classification.VGG16.transform import VGG16

model = VGG16(classes_number=3, trainable_stages=0)

# Save the modified model to a file named "model.pt"
torch.save(model.state_dict(), "models/classification/VGG16/model.pt")
```

2. Additionally, there must be a `class_dict.json` file that maps class names to class indices. This file should be in the following format:

```json
{
    "0": "class0",
    "1": "class1",
    "2": "class2"
}
```

3. After these steps, you will have a modified VGG16 model saved as `model.pt` in the current directory, ana a `class_dict.json` file to map the output predictions to their corresponding semantic class names. This file will be automatically loaded by ImageLogViewer when you run the classification feature.

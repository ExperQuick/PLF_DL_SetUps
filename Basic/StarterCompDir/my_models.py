"""
Example: Model Component for PyLabFlow
======================================

📘 Purpose
----------
This file demonstrates how to define a **trainable model** as a PyLabFlow component.

The `Model1st` class below wraps a simple convolutional neural network for MNIST-like
datasets. It combines the flexibility of PyTorch's `nn.Module` with PyLabFlow’s
component system — making it dynamically loadable, configurable, and reproducible.

Following this pattern ensures your model integrates seamlessly into
the `WorkFlow` and `PipeLine` ecosystem.

Key benefits:
-------------
✅ Supports dynamic loading from configuration  
✅ Tracks arguments (`args`) for experiment reproducibility  
✅ Fully PyTorch-compatible for training and inference  
✅ Easily replaceable in configuration without code edits
"""

from plf.utils import Component


from torch import nn

# ----------------------------------------------------------------------
# EXAMPLE: PyTorch Model Component
# ----------------------------------------------------------------------
class Model1st(Component, nn.Module):
    """
    Model1st
    --------
    A simple configurable convolutional neural network.

    🔗 Integration
    --------------
    - Extends both `Component` (for PyLabFlow) and `nn.Module` (for PyTorch).
    - Instantiated dynamically by the workflow via `load_component()`.
    - Can be swapped or modified in the pipeline configuration file
      without changing any training code.

    ✅ Why users should follow this pattern:
        - Ensures every model is reproducible and registered in the experiment database.
        - Allows experiments to easily compare models with different depths or layers.
        - Keeps model construction logic isolated inside `_setup()` for clean reloads.
    """
    def __init__(self):
        """
        Initialize the model component.

        Users should define required setup arguments (`self.args`) here.
        These arguments must be present in the experiment configuration under:
            config["args"]["model"]["args"]

        Example:
        --------
        ```json
        "model": {
            "loc": "CompBase.models.Model1st",
            "args": {
                "conv_deep": 2,
                "dense_deep": 1
            }
        }
        ```
        """
        Component.__init__(self)
        nn.Module.__init__(self)

        # Expected keys required for _setup()
        self.args = {'conv_deep', 'dense_deep'}

    # ------------------------------------------------------------------
    # REQUIRED BY Component: _setup()
    # ------------------------------------------------------------------
    def _setup(self, args):
        """
        Construct the model architecture using the provided arguments.

        Parameters
        ----------
        args : dict
            Dictionary containing:
            - conv_deep : int, number of additional convolutional layers
            - dense_deep : int, number of additional fully connected layers

        Responsibilities
        ----------------
        - Define all model layers.
        - Build variable-depth networks dynamically based on config.
        - Keep all initialization inside this method for reproducibility.
        """

        # ----- Base convolutional stem -----
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # ----- Variable-depth convolutional stack -----
        convs = [ ]
        for i in range(args['conv_deep']):
            convs.append(nn.Conv2d(64, 64, kernel_size=3, padding=1))
            convs.append(nn.ReLU())

        self.conv3 = nn.Sequential(*convs)

        # ----- Pooling and fully connected layers -----
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 64)

        # Variable-depth dense layers stack
        denses = []
        for i in range(args['dense_deep']):
            denses.append(nn.Linear(64 , 64) )
            denses.append(nn.ReLU())
        self.fc2 = nn.Sequential(*denses)

        # Output layer for 10-class classification
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()


    # ------------------------------------------------------------------
    # FORWARD PASS
    # ------------------------------------------------------------------
    def forward(self, x):
        """
        Define the forward computation.

        This method is called automatically during training and inference.

        Parameters
        ----------
        x : torch.Tensor
            Input batch (shape: [B, 1, 28, 28] for MNIST)

        Returns
        -------
        torch.Tensor
            Output logits (shape: [B, 10])
        """
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        # Additional convolutional stack
        x = self.relu(self.conv3(x))

        # Flatten and pass through dense layers
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))

        # Final classification layer
        x = self.fc2(x)
        return x
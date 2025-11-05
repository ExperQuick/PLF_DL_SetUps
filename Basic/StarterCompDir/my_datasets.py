"""
Example: Dataset and Transform Components for PyLabFlow
=======================================================

📘 Purpose
----------
This file demonstrates how users should implement **dataset** and **transform**
components for workflows in PyLabFlow.

These classes show how to properly extend the base `Component` class so that
the pipeline (`PipeLine`) can dynamically load, validate, and manage them.

Two example components are provided:
    1. `MNISTDataset` – a dataset component wrapping torchvision’s MNIST
    2. `Augment` – a simple image transformation pipeline

Users can treat these as templates when creating their own datasets and preprocessing steps.
"""

from plf.utils import Component

from torch.utils.data import Dataset
from torchvision import datasets, transforms


# ----------------------------------------------------------------------
# EXAMPLE: Dataset Component
# ----------------------------------------------------------------------
class MNISTDataset(Component, Dataset):
    """
    MNISTDataset
    ------------
    Example dataset component that loads the MNIST dataset.

    🔗 Integration
    --------------
    - Inherits from both `Component` and `torch.utils.data.Dataset`.
    - Managed by the PyLabFlow `PipeLine` system through the `WorkFlow`.
    - Dynamically instantiated using `load_component(loc=..., args=...)`.

    ✅ Why users should follow this structure:
        - `Component` gives the dataset dynamic loading and argument validation.
        - `Dataset` gives PyTorch dataloader compatibility.
        - `_setup()` ensures reproducibility and clear parameterization.
    """
    def __init__(self):
        """
        Initialize the dataset component.
        """
        Dataset.__init__(self)
        Component.__init__(self)


        # Define required setup arguments
        # These keys must exist in the config under args['dataset'].
        self.args = {'root_dir', 'train', 'transform'}

    # ------------------------------------------------------------------
    # REQUIRED BY Component: _setup()
    # ------------------------------------------------------------------
    def _setup(self,args):
        """
        Prepare the dataset.

        This method is automatically called by `load_component()` when the
        component is created by the workflow or pipeline.

        Parameters
        ----------
        args : dict
            Contains the keys: 'root_dir', 'train', and 'transform'.

        Responsibilities
        ----------------
        - Download or load the dataset.
        - Dynamically load the transform component.
        - Keep all resource management within PyLabFlow context.
        """
        # Load MNIST dataset (PyTorch will download if not present)
        self.mnist = datasets.MNIST(root=args['root_dir'], train=args['train'], download=True)

        # Load the transform component dynamically
        # (e.g., args['transform'] = {'loc': 'CompBase.data.Augment', 'args': {}})
        self.transform = self.load_component(**args['transform'])

    # ------------------------------------------------------------------
    # STANDARD DATASET INTERFACE
    # ------------------------------------------------------------------
    def __len__(self):
        """Return total number of samples in the dataset."""
        return len(self.mnist)

    def __getitem__(self, idx):
        """
        Return a transformed sample and label.

        The pipeline never calls this directly — it’s used internally by
        PyTorch’s DataLoader during training.
        """
        img, label = self.mnist[idx]
        if self.transform:
            img = self.transform(img)
        return img, label




# ----------------------------------------------------------------------
# EXAMPLE: Transform Component
# ----------------------------------------------------------------------
class Augment(Component):
    """
    Augment
    -------
    Simple transformation component for MNIST images.

    🔗 Integration
    --------------
    - Dynamically loaded inside a dataset (e.g. MNISTDataset)
    - Must implement `_setup()` for initialization
    - May define `__call__()` for runtime data processing

    ✅ Why users should follow this structure:
        - Keeps transforms modular and reusable.
        - Enables composable pipelines where transforms can be replaced via config.
        - Maintains PyLabFlow’s reproducibility and version tracking.
    """
    def __init__(self):
        """
        Initialize the component.
        """
        super().__init__()

        # No required arguments — empty dict
        self.args = {}

    def _setup(self,args):
        """
        Initialize the transformation logic.

        Parameters
        ----------
        args : dict
            Reserved for future parameters (currently unused).

        Responsibilities
        ----------------
        - Create the transformation pipeline.
        - Ensure deterministic and consistent preprocessing steps.
        """
        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))  # normalize grayscale MNIST
            ])

    # ------------------------------------------------------------------
    # TRANSFORM INTERFACE
    # ------------------------------------------------------------------   
    def __call__(self, img):
        """
        Apply the transformation.

        This allows the component to be called like a function:
            transformed_img = transform(img)

        PyTorch Datasets use this convention, so all PyLabFlow transform
        components should implement `__call__`.
        """
        return self.transform(img)


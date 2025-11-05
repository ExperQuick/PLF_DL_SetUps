"""
Example: BasicWorkFlow — A Minimal Reference Workflow for PyLabFlow
===================================================================

📘 Purpose
----------
This example shows **how users should implement their own WorkFlow** classes
that integrate seamlessly with the PyLabFlow `PipeLine`.

PyLabFlow separates *experiment orchestration* (handled by `PipeLine`)
from *domain-specific logic* (implemented in a subclass of `WorkFlow`).

This BasicWorkFlow demonstrates:
  ✅ How a workflow interacts with the pipeline lifecycle (`new → prepare → run → status`)
  ✅ How to use `load_component()` to dynamically load model/dataset objects
  ✅ How to define file alias paths, logging, and resumable checkpoints
  ✅ How to remain consistent with the abstract interface in `WorkFlow`

Use this as a **template** for building your own workflows.
"""



from plf.utils import WorkFlow
from typing import Dict, Any
import json, os
from copy import deepcopy
from pathlib import Path
import pandas as pd

import torch
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.optim as optim



class BasicWorkFlow(WorkFlow):
    """
    BasicWorkFlow
    -------------
    A minimal example of a PyTorch training workflow that follows PyLabFlow conventions.

    This class inherits from `WorkFlow` (which itself extends `Component`), 
    making it compatible with `PipeLine`.

    The pipeline manages configuration files, IDs, and persistence,
    while the workflow defines how an experiment is actually executed (training, evaluation, etc.).

    🧩 Users should follow this structure because:
        - It satisfies all abstract requirements of `WorkFlow`.
        - It keeps your training loop reproducible and pluggable.
        - It allows your experiments to be tracked, resumed, and logged automatically.
    """

    # ----------------------------------------------------------------------
    # INITIALIZATION
    # ----------------------------------------------------------------------
    def __init__(self):
        """
        Define all the required parameters and metadata for this workflow.
        These declarations tell the pipeline what to expect.
        """
        super().__init__()

        # ---- REQUIRED: define workflow-specific arguments ----
        # The pipeline will validate these at creation time.
        self.args = {'num_epochs'} # Arguments expected while the workflow initilization

        # ---- REQUIRED: define the workflow argument template ----
        # Every workflow must define what arguments (components and values)
        # must exist under `args` in the pipeline config.
        # This ensures every experiment is reproducible and correctly defined.
        self.template = {
            'model', 'dataset', 'batch_size'
        }

        # ---- REQUIRED: list of file alias names (for path management) ----
        # Each alias here must have a matching case in get_path().
        # This gives the pipeline a unified way to locate workflow artifacts.
        self.paths = {"history.history", "quick", 'weights.last'}

        # ---- RECOMMENDED: define logging structure ----
        # Even if you track only loss, define a schema — it enforces consistency.
        self.logings = {  # here it is not that nessesary as  we are only keeping  loss per epoch   but  recomended
            "history.history": ['epoch', 'loss']
        }
    

    # ----------------------------------------------------------------------
    # REQUIRED: _setup() (called during component initialization)
    # ----------------------------------------------------------------------
    def _setup(self, args):
        """
        Called automatically by `load_component` when the workflow is created.

        The `_setup()` method initializes internal attributes 
        from the arguments provided in the pipeline configuration.
        """
        self.num_epochs= args['num_epochs']


    # ----------------------------------------------------------------------
    # REQUIRED: prepare()
    # ----------------------------------------------------------------------
    def prepare(self):
        """
        Prepare all resources needed before running.

        This method is called by `PipeLine.prepare()` after the pipeline has been created or loaded.
        It’s where the workflow converts pipeline config entries into live objects.

        Responsibilities:
          - Load model and dataset components dynamically.
          - Create DataLoaders.
          - Initialize loss and optimizer.
          - Resume from any previous checkpoint.
        """
        if not self.P.cnfg:
            print("not initiated")
            return
        

        # Deep-copy args from pipeline config to avoid side-effects.
        args = deepcopy(self.P.cnfg["args"])
        
        # --- Load model and dataset dynamically ---
        # This is the recommended way to integrate with PyLabFlow.
        # `load_component()` automatically sets .P and calls _setup() internally.
        self.model = self.load_component(**args["model"])
        ds = self.load_component(**args['dataset'])

        # Create DataLoader for training
        self.trainDataLoader = DataLoader(
            dataset = ds, batch_size = args['batch_size']
        )

        # Resume previous training state (if any)
        self.resume()
        print("Data loaders are successfully created")

        # Define loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        # here we fixed loss and  optimised for minimality  but yu can make them changable just like  model and dataset

        # Set device (GPU if available, otherwise CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        return True


    # ----------------------------------------------------------------------
    # CHECKPOINT MANAGEMENT
    # ----------------------------------------------------------------------
    def resume(self):
        """
        Resume from the last saved checkpoint.

        The pipeline ensures that quick logs (stored as JSON) exist for each experiment.
        This method simply reads that log and restores the last epoch count.

        ✅ Why this is required:
           - It allows workflows to pause/resume experiments transparently.
           - Makes workflows state-aware, enabling automation in long runs.
        """
        with open(self.P.get_path(of="quick"), encoding="utf-8") as quick:
            quick = json.load(quick)

        # Resume from last epoch if available, else start from 0
        self.current_epoch = quick['last']['epoch'] if quick['last']['epoch'] else 0
        
    # ----------------------------------------------------------------------
    # TRAINING LOOP
    # ---------------------------------------------------------------------- 
    def train_epoch(self, current_epoch):
        """
        Perform one epoch of training.

        This is the core computation logic — it defines how your workflow actually "runs".

        The pipeline will never interfere with this; it just tracks progress and paths.
        """
        total_loss = 0.0

        for x,y in self.trainDataLoader:
            # Move data to the selected device
            x,y = x.to(self.device), y.to(self.device)

            # Forward pass
            pred = self.model(x)

            # Compute loss
            loss = self.criterion(pred, y)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()
            total_loss += loss.item()

        # Update current epoch
        self.current_epoch = current_epoch+1

        data = {   # Prepare metrics to log
                "epoch": self.current_epoch ,

                "loss": total_loss/len(self.trainDataLoader)
            }

        # Log metrics, checkpoint, and model weights
        self.log(of="history.history",  data = data)

        quick = {
            "last": {
                'epoch':self.current_epoch
                }
            }
        self.log(of="quick", data = quick)

        self.log(of='weights.last')

        print(f"epoch: {self.current_epoch} | Loss: {data['loss']}")

    # ----------------------------------------------------------------------
    # REQUIRED: run()
    # ----------------------------------------------------------------------
    def run(self):
        """
        The main execution loop for the workflow.

        Called automatically by `PipeLine.run()` once everything is prepared.

        It checks the pipeline’s runtime control flags (e.g., `should_running`)
        to safely handle remote or scheduled stop commands.
        """
        for current_epoch in range(self.current_epoch, self.num_epochs):
            if not self.P.should_running:
                return

            self.train_epoch(current_epoch=current_epoch)

    # ----------------------------------------------------------------------
    # REQUIRED: new()
    # ----------------------------------------------------------------------
    def new(self, args: Dict[str, Any]) -> None:
        """
        Initialize new experiment logs.

        Called automatically when the pipeline is created via `PipeLine.new()`.
        Responsible for creating empty log files (CSV, JSON, etc.)

        ✅ Why required:
           - Guarantees each new experiment starts cleanly.
           - Prevents overlapping history between multiple runs.
        """
        # Ensure all required configuration keys are provided
        if not self.template.issubset(set(args.keys())):
            raise ValueError(f'the args should have {", ".join(self.template- set(list(args.keys())))}')
        
        # Create empty history CSVs
        for i in self.logings:  # here no ned for a loop as we are logging only 1 csv. But recomended
            record = pd.DataFrame([], columns=self.logings[i])
            record.to_csv(self.P.get_path(of=i), index=False)

        # Initialize quick checkpoint with epoch 0
        quick = {
            "last": {  # here  you can keep  more  info  like  best loss, best loss-epoch, best othermetric, best other metric-epoch  etc  to  get quick info of ppl
                'epoch':0
                }
            }
        self.log(of='quick', data=quick)

    # ----------------------------------------------------------------------
    # LOGGING AND SAVING
    # ----------------------------------------------------------------------        
    def log(self, of, data=None):
        """
        Unified logging handler.

        This method defines how each alias (‘of’) writes to disk.
        The pipeline doesn’t handle any of this logic — that’s up to your workflow.
        """
        if of=='quick':
            pth = self.P.get_path(of='quick')

            # Read existing quick file if available
            if os.path.exists(pth):
                with open(pth) as fl:
                    qck = json.load(fl)
                qck.update(data)
            else:
                qck = data

            # Write back to the same file
            with open(self.P.get_path(of='quick'), 'w') as fl:
                json.dump(qck, fl, indent=4)

        elif of == 'history.history':
            metrics = self.logings[of]
            record = pd.DataFrame([[data[i] for i in metrics]], columns=metrics)
            record.to_csv(
                self.P.get_path(of=of),
                mode="a", # append to existing log
                header=False,
                index=False,
            )
        elif of == 'weights.last': # here we are keepinmg only last epoch weights you also can keepfor example best epoch(according to loss or other metrics) 
            # but  add the alias in self.path in __init__ and  write logic in  get_path method
            # Save model weights
            torch.save(
                    self.model.state_dict(),
                    self.P.get_path(of='weights.last'),
                )

    # ----------------------------------------------------------------------
    # REQUIRED: get_path()
    # ----------------------------------------------------------------------        
    def get_path(self, of, pplid, args= None) -> str:
        """
        Return standardized, pipeline-relative paths for workflow artifacts.

        Each alias must map to a unique path under the pipeline’s base data directory.

        ✅ Why required:
           - The pipeline depends on this to locate and manage your artifacts.
           - It ensures every workflow is relocatable and self-contained.
        """
        if "history.history" == of:
            path = Path(*of.split(".")) / f"{pplid}.csv"
        
        elif of == "quick":
            path = Path("Quicks") / f"{pplid}.json"

        elif of == 'weights.last':
            path = Path("Weights") / "'last" / f"{pplid}.pt"

        else:
            raise ValueError(
                f"Invalid value for 'of': {of}. Supported values: "
                "'config', 'weight', 'gradient', 'history', 'quick'."
            )
        
        return path


    # ----------------------------------------------------------------------
    # OPTIONAL: clean() & status()
    # ----------------------------------------------------------------------
    def clean(self):
        """
        Optional cleanup — e.g., remove caches or temporary artifacts.
        Called when `PipeLine.clean()` is executed.
        """
        pass

    def status(self) -> str:
        """
        Return lightweight progress information for monitoring.

        The pipeline uses this when displaying status of running or past workflows.
        """
        with open(self.P.get_path(of='quick')) as fl:
            qck = json.load(fl)
        return qck['last']
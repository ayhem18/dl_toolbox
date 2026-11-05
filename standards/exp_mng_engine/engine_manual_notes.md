# Overview
This is a .md file, written by me for me. 

# Abstraction

* The entry point is [engine.py](../../../dl_toolbox/src/dl_toolbox/experiment_management/engine/core/engine.py)  

* The train() function is quite simple 

1. if this is a worker -> extract the existing run and start the fit function. This still feels like a patch 
2. otherwise: depending on the "force" parameter, either create the new run or overwrite it. 

The bulk of the logic is at one function: 

```
_assemble()
```

1. parsing the recipe. I think I should delegate most of the parsing logic to Pydantic, instead of wasting tokens on validation logic.
2. compose tool box: this should not be the case, the return type is unspecified ... this can very easily break. This is probably the sloppiest part of the engine
3. the object returned by compose_toolbox calls an apply_seed() method ... just terrible
4. after applying the seed, extract the elements necessary of the run identity: name, and hash configuration.
5. depending on the create parameter: self._create_or_overwrite_run() or self._require_existing_run()

6. The _create_or_overwrite_run() logic should probably be delegated an identity manager. The class is not abstract the way it is supposed to be: it imports the  HASH_TAKEN, BOTH_MATCH, BOTH_FREE; parameters from the a concrete implementation of the identity manager. This behavior should probably be configurable. 
 

1. Now, let's rethink the engine. The better option is to think of run types first

- TrainOnlyRun: overfit is a good example of this
- TrainValRun: what is already implemented in the fit
- LrFinderRun: train only but different optimization loop
- EvalRun: evaluate, no optimizer just evaluation
- HyperParamSearchRun: might be worth investigating later on
- InspectRun: 



## Checkpointing

The checkpointing component is tricky. 

One there is the write operation. Writing a checkpoint that can be later loaded and used. 

let's break down the current checkpointing behavior in the [fit.py](../../../dl_toolbox/src/dl_toolbox/dl_engine/training/fit.py) 

```python

    callbacks: list[L.Callback] = [checkpointer.as_callback()]
    # early stopping configuration 
    trainer = L.Trainer(
        max_epochs=config.trainer.max_epochs,
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        precision=config.trainer.precision,
        accumulate_grad_batches=config.accumulate_grad_batches,
        callbacks=callbacks,
        enable_progress_bar=config.trainer.enable_progress_bar,
        logger=False,
        enable_checkpointing=False,
    )
```

1. How do callbacks work ? According to the docs and the LLM answer: for each hook: `on_fit_start`, `on_validation_epoch_end` ... the trainer calls the same of hook for each callback and only then the hook of the module (if explicitly implemented )

2. Currently we have: 

```python
callbacks: list[L.Callback] = [checkpointer.as_callback()]
```

let's check the [checkpointer.py](../../../dl_toolbox/src/dl_toolbox/dl_engine/training/checkpointer/checkpointer.py) and the [checkpointer_manager.py](../../../dl_toolbox/src/dl_toolbox/dl_engine/training/checkpointer/manager.py)

```python

class _OwnedCheckpoint(L.Callback):
    def __init__(self, checkpointer: Checkpointer) -> None:
        super().__init__()
        self._checkpointer = checkpointer

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        del pl_module
        if trainer.sanity_checking:
            return
        self._checkpointer.save(trainer)
```

the checkpointer is an involved class. 

```python
class Checkpointer:
    def __init__(self, *, binding: CheckpointerBinding) -> None:
        self._binding = binding

    def as_callback(self) -> L.Callback:
        return _OwnedCheckpoint(self)

    def save(self, trainer: L.Trainer) -> None:
        # this is the most basic functionality & it is still loaded. 
        # The write logic. 
        # the scoring -> whether a checkpoint should be written to begin with
        # if it is inspectable -> we need a locking mechanism
        # we need to take into account DDP as well

        # this part should be clarified: DDP & inspectability & selection mechanism & the write operation 

        # currently this logic only holds for runs whether the lightning module is the core of the run logic. 

    def fetch(self, request: CheckpointRequest, *, dest: Path) -> CheckpointRecord:
        return self._binding.fetch(request, dest=dest)

    def fit_state(self) -> FitRunState:
        return self._binding.fit_state()

    def mark_finished(self) -> None:
        self._binding.mark_finished()
``` 

Let's for now dig deep into the existing solution and checkpointing logic. Googling says that, given a trainer, we can identify whether the training is multi process as follows: 

```python
is_multi_process = self.trainer.world_size > 1
```

that's important for us to make sure that a run class that was not implemented with multi-process in mind is not used in a multi-processing training 

There is
# Overview

just a quick document for me to gather my thoughts and understand the internals of pytorch lightning so that I can use them effectively in the future

## important components

### hooks

1. hooks, module and data module apparently. The Trainer orchestrates the loop. 

I am not sure how the data module interacts with the rest, but for now I don't think the interaction is important or significant

for each hook, the trainer calls the same hook for each callback and only then the hook of the module (if explicitly implemented )

### Multi-Process training

stands for distributed data parallel. I need to understand how this works. From my experience, it is a multi-process and not multi-threading implementation. 

One flag is 
```python     
    # from the Trainer implementation 
    @property
    def is_global_zero(self) -> bool:
        """Whether this process is the global zero in multi-node training.

        .. code-block:: python

            def training_step(self, batch, batch_idx):
                if self.trainer.is_global_zero:
                    print("in node 0, accelerator 0")

        """
        return self.strategy.is_global_zero
``` 

whether the process is a global zero, what does that mean ??? Can we tell from a trainer whether the training is DDP or single process ? 

```python
is_multi_process = self.trainer.world_size > 1
```

that's important for us to make sure that a run class that was not implemented with multi-process in mind is not used in a multi-processing training 


So here we are talking about 2 layers of parallelism 

1. multi-nodes: differnt physical machines
2. multi-processes: different processes within a given physical machine

within each process, we have the local_rank. 

the global_rank = local_rank + machine_rank * num_processes_in_machine 

it should be possible to determine whether training is multi-node or only on a single machine. 

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
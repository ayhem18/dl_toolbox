# Coding Standards

## Keyword-only parameters (`*`)

Use a bare `*` in function signatures to force **keyword-only** arguments for every parameter that follows it.

```python
def load_vocabulary(
    language: str,
    *,
    pull_n: int,
    min_zipf: float,
    wordlist: str = "best",
) -> list[tuple[str, float]]:
    ...
```

### When to require it

Apply `*` to optional and configuration parameters in **package APIs** (`blw`, `ruw`, scripts that wrap them), especially when:

- several parameters share the same type (`int`, `float`, `str`) and positional order is easy to swap by mistake;
- the parameter list is long enough that call sites should read as self-documenting (`pull_n=`, `min_zipf=`);
- a function has optional overrides (`start=`, `mkdir=`, `download=`) that must not be passed positionally.

### When it is optional

Small internal helpers with one or two obvious parameters may omit `*`, but prefer keyword-only for anything exported or called from more than one module.

### Examples

```python
# Good — config knobs are explicit at the call site
build_russian_vocab(config, db_path, *, start=None, morph=None)

# Good — all options keyword-only
get_wordnet31_dict(*, start=None, download=True, extract=True)

# Avoid in public APIs — positional ints are fragile
load_vocabulary("ru", 20000, 3.5)
```

### Defaults

Keyword-only does not remove the need to think about defaults. It only ensures callers **name** each argument. Document non-obvious defaults in the docstring.

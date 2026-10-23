1. When the number of elements in a list if known, use the following pattern

```
list = [null] * n

iterate then assign values to the list
```

DO NOT USE list.append() to add elements to a list.


2. Reduce the database IO overhead by using batched inserts / deletions (well operations in general)

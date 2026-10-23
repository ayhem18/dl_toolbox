# Testing Standards

## Goal

The main objective is not only to check whether code runs, but to build a testing culture that gives high confidence in:

- endpoint behavior
- database side effects
- object storage side effects
- duplicate-handling logic
- correctness across many different inputs and scenarios

## Core Principles

### 1. Tests should verify outcomes, not just code paths

The primary purpose of tests is to verify that the system produces the correct outputs and side effects.

This means tests should check:

- returned API responses
- database state
- object storage state
- duplicate handling behavior
- failure behavior under bad inputs or infrastructure errors

### 2. Prefer target-based testing whenever possible

Whenever practical, tests should be written in a target-based style.

Meaning:

- define the desired output or invariant
- determine the inputs that should produce that output
- run the code
- verify the output matches the expectation

This is preferred over reproducing internal implementation logic in the test.

Why:

- it avoids writing tests that simply duplicate the code under test
- it gives more confidence that the test is validating externally meaningful behavior
- it makes refactoring easier because tests stay tied to contract, not implementation details

Examples:

- if two uploads contain identical PDF bytes, the expected output is that they resolve to the same `document_id`
- if a non-PDF file is uploaded, the expected output is a client error and no database row
- if a new document is uploaded successfully, the expected output is one DB row and one stored object

### 3. Core Tenet: Elevate Looping and Randomization as First-Class Assertions

We treat looping and randomized trials not as optional extras, but as first-class core testing practices.

To prevent tests from being brittle or limited to a single hardcoded path, tests should use:

- **Deterministic random generators (with fixed seeds):** ensures reproducibility while expanding the testing space across multiple execution scenarios.
- **Repeated, randomized loop trials:** loops over multiple generated data combinations, verifying that correctness holds globally under any valid state structure.
- **Randomized state configurations:** generate arbitrary variations of database rows (such as page rendering states, processing statuses, or sorting properties) to ensure that the logic behaves predictably even with chaotic data configurations.

This is highly effective for:

- **Query filter combination coverage:** validating that filter and query parameters (such as pipeline versions, types, or status filters) work flawlessly when subjected to chaotic variations.
- **Sorting order invariants:** seeding objects with randomly generated sort order values, and validating that the API outputs sorted records strictly under any random permutation.
- **State transitions & calculations:** simulating various combinations of execution records and assuring that summaries (like total pages rendered/completed) match calculated invariants.

The practical standard:

- Use deterministic random seeds (e.g. `random.Random(42)`) when reproducibility is critical.
- Use repeated loop trials with meaningful volumes (e.g. 10 to 100 iterations per test) to fully exercise combinations.
- Keep the test suite deterministic, ensuring that a failing combination is easily replayable.

### 4. Long-running tests are acceptable when they buy confidence

Some tests should intentionally run longer if they meaningfully reduce risk.

Examples:

- repeated duplicate-upload checks
- randomized ingestion scenarios
- state transition tests over many iterations
- infrastructure-backed integration checks

The goal is not to make all tests long.

The goal is to allow slower, higher-confidence tests where they save time and uncertainty later.

### 5. Do not let tests depend on implementation internals unnecessarily

Tests should avoid:

- relying on private helper behavior unless the helper is itself the test target
- reconstructing internal algorithm steps line by line
- asserting brittle formatting or incidental details

Tests should prefer:

- public API behavior
- persisted state
- contract-level invariants
- externally visible side effects

## Testing Layers

The backend should use multiple testing layers.

### Unit tests

Purpose:

- validate small isolated logic quickly

Examples:

- file validation
- storage key generation
- content hash generation
- duplicate-detection decisions

Properties:

- fast
- isolated
- no real DB
- no real object storage

### Integration tests

Purpose:

- validate interactions with real infrastructure components

Examples:

- repository operations against PostgreSQL
- Alembic migration correctness
- object storage integration behavior
- document-ingestion workflow with DB and storage together

Properties:

- may use Dockerized Postgres
- may use local S3-compatible storage
- slower than unit tests

## Randomized Test Guidance

Randomized testing should be used deliberately and systematically.


Important rules:

- Randomization must be reproducible when debugging.
- Always use a fixed seed (`random.Random(seed_value)`) so that a test failure can be replayed exactly.
- Use clear iteration counts (e.g. 10 to 100 trials) so that the performance and token/time cost of the test is predictable.

Examples of acceptable patterns:

- Loop over 100 randomized filenames for the same payload.
- Loop over 100 randomized invalid uploads.
- Run many document-upload attempts and verify invariants hold.
- Execute 10 repeated query trials where database rows are seeded with random attributes and verify filtering logic dynamically.

## Preferred Testing Style

When possible, tests must adhere whenever possible to the following principles:

- contract-level assertions
- generated inputs
- repeated execution
- avoid duplicating production logic in the tests: WE do not test the logic by reimplementing the logic

## Folder-Level Standards

The tests files should be organized in the same way as the production file: a structure that mirrors the main python package is a very reasonable choice.


## Summary

The testing standard for this backend is intentionally strict:

- use target-based tests whenever possible
- use randomized and repeated inputs when they improve confidence
- assert infrastructure side effects directly
- keep test logic independent from production implementation details
- use multiple test layers instead of relying on only one style of test

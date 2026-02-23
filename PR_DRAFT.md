## Motivation

This change targets dLLM performance optimization on Ascend, with **decode throughput improvement** as the top priority.

- Improve output throughput on the same hardware (Ascend 910B, single card).
- Keep the accuracy impact limited and explicitly reported.

## Modifications

- Core changed file: `python/sglang/srt/dllm/algorithm/credit_decoding.py` (added).
  - Adds Credit Decoding logic to improve decode-stage efficiency on the Ascend dLLM path.
- Naming convention check:
  - File naming follows the `dllm/algorithm` module style (`snake_case`): `credit_decoding.py`.
  - Class naming follows project conventions (`PascalCase`): `CreditDecoding`.

## Accuracy Tests

- [x] This change can affect model outputs.
- [ ] This change does not affect model outputs.

Current result:

- Scenario: Ascend 910B (1 card), Confidence = 0.85
- Command: `sglang-dllm/test/srt/dllm/test_llada2_mini_ascend.py`
- Accuracy: `0.94 -> 0.925` (minor drop)

## Benchmarking and Profiling

- Hardware: Ascend 910B, 1 card
- Software:
  - CANN: 8.3.RC2
  - Torch: 2.7.1
- Workload: Confidence = 0.85
- Key metrics:
  - output_throughput: `99.328 -> 110.416`
  - Accuracy: `0.94 -> 0.925`

## Checklist

- [ ] Format your code according to the [Format code with pre-commit](https://docs.sglang.io/developer_guide/contribution_guide.html#format-code-with-pre-commit).
- [ ] Add unit tests according to the [Run and add unit tests](https://docs.sglang.io/developer_guide/contribution_guide.html#run-and-add-unit-tests).
- [ ] Update documentation according to [Write documentations](https://docs.sglang.io/developer_guide/contribution_guide.html#write-documentations).
- [x] Provide accuracy and speed benchmark results according to [Test the accuracy](https://docs.sglang.io/developer_guide/contribution_guide.html#test-the-accuracy) and [Benchmark the speed](https://docs.sglang.io/developer_guide/contribution_guide.html#benchmark-the-speed).
- [ ] Follow the SGLang code style [guidance](https://docs.sglang.io/developer_guide/contribution_guide.html#code-style-guidance).

## Pre-PR Checks

- [x] Confirm `credit_decoding.py` naming matches module conventions (`snake_case`).
- [x] Confirm no remaining `CreditDecoding.py` filename references in the repository.
- [ ] Run pre-commit hooks locally and attach results.
- [ ] Ensure required tests are attached in PR description/comments.
- [ ] Add/confirm `run-ci` label path (directly or via authorized maintainer).

## Review Process

- [ ] Ping Merge Oncall to start PR flow.
- [ ] Request approvals from CODEOWNERS/reviewers.
- [ ] Ensure `run-ci` label is present.
- [ ] Use CI commands as needed: `/tag-run-ci-label`, `/rerun-failed-ci`, `/tag-and-rerun-ci`.

## Notes for Reviewers

- This is an Ascend-focused performance change centered on `credit_decoding.py`.
- It improves decode throughput (`99.328 -> 110.416`) with a minor accuracy trade-off (`0.94 -> 0.925`).
- Please review as a performance/accuracy trade-off change.

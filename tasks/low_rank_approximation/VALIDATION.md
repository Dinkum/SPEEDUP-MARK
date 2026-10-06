# low_rank_approximation validation

The executable contract is [task_spec.py](task_spec.py); [README.md](README.md) defines the accepted output and numerical tolerances.

Run the repository regression suite with the optional numerical packages installed:

```console
python3 -B -m unittest discover -s tests -v
```

Check each declared `grading_cases` size with `python3 -m speedupmark low_rank_approximation --n SIZE --seed 0 --samples 3 --json`. Include every family-selection residue described in the workload distribution; the default three samples do not cover every task's longer family cycle.

A release result must record the tested task version, content revision, candidate hash, Python/platform, package versions, sizes, and seeds. Reusing an earlier timing table does not validate changed source. These checks establish correctness for the tested inputs and do not establish model performance or a stable speed ranking.

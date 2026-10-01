# Gzip Compression

Input is `{"plaintext": bytes}`. Return `{"compressed_data": bytes}` containing one complete gzip member, with no trailing data, which decompresses exactly to the plaintext. Its byte length must be **at most `ceil(1.001 * baseline_gzip_size)`**, where the baseline is `zlib.compressobj(level=9, method=DEFLATED, wbits=31)` plus finish flush on the same runtime. Verification enforces both size and bounded exact decompression. Defaults are 100,000 and 250,000 bytes. Seed families include incompressible bytes, skewed words, repeated binary motifs and mixed logs/noise. This bounded stdlib adaptation uses a 0.1% size tolerance rather than upstream's no-larger rule; uncompressed gzip is not a universal valid shortcut.

Optimize `candidate.py:solve(problem, reference_solve)`. Only Python standard-library dependencies are required. All output containers must be materialized built-in types; integers must be exact `int`, never `bool`. All construction and preprocessing belongs inside the timed solve call. Inputs are read-only.

Provenance: independently authored SPEEDUP-MARK adaptation of the task described in AlgoTune, `AlgoTuneTasks/gzip_compression/description.txt` (https://github.com/ScalingIntelligence/AlgoTune). No upstream implementation was copied. The bounded generator, Python data representations and exact verification contract below are SPEEDUP-MARK-specific; scores are not equivalent to upstream AlgoTune scores.

## Workload distribution

- **Size:** n is plaintext byte count.
- **Selection:** seed % 4 selects random bytes, skewed words, repeated motif, or mixed log/binary.
- **Randomized:** Uniform bytes; 100 lowercase words of lengths 3–15 sampled with cubic rank skew; a 4096-byte motif over values 0–15; or random 512-byte blocks.
- **Fixed structure:** Mixed mode alternates a fixed log line block with binary blocks. Small n may contain only the fixed prefix. Three samples do not cover all four modes.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers. See [sampling and coverage](../../GUIDE.md#sampling) for how the grader chooses and records seeds.

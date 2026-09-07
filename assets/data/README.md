# Datasets

Place uncompressed CH graph/range pairs here. Subdirectories are supported.

Example:

```text
assets/data/bw/bw.sch
assets/data/bw/bw.sch.ranges
```

The Dataset panel scans this directory recursively. A graph file is recognized when a file with the same path plus `.ranges` exists.

Compressed `.bz2` input is not loaded directly yet. Extract it before placing the files here.

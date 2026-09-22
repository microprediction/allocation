# research

Studies that are too large to run on a laptop.

`experiments/` holds the simulations behind the papers, and they are meant to
be run in place while the argument is being worked out. What lands here is
different: a question whose answer needs hours of compute, packaged so it can
be copied to a bigger machine, run unattended, and merged afterwards.

Anything in this directory follows four rules, and they exist because each one
has cost a day at some point.

**Self-contained.** No imports from `experiments/` or from `allocation/`
beyond the published package. A directory here can be `rsync`'d on its own
and will run.

**Sharded by an index, not by a seed.** Every unit of work seeds itself from
`(seed, global_index)`, so a result does not depend on how the work was
divided. Each study proves this with a check that runs the same draws as one
shard and as several and compares.

**Smoke first.** A `smoke.py` that runs in under a minute and asserts the
properties the study depends on. Run it on the new machine before starting
anything long. Every check in it corresponds to a defect that actually
happened.

**One BLAS thread per worker.** The work is parallel across draws already.
Letting each worker open its own thread pool oversubscribes the machine and,
on a laptop, runs it out of memory.

| study | question | status |
|---|---|---|
| `subportfolio/` | how should a sub-portfolio be formed from an optimal index? | running |

# ChampSim

ChampSim is a trace-based simulator for microarchitecture research.

Useful trace links:
- Traces (access requires an LDAP ID): https://drive.google.com/drive/folders/1zYchkn-M1auZp_l5wRkIzAcDYPkNtW3H?usp=sharing
- ChampSim wiki: https://champsim.github.io/ChampSim/master/index.html

## Clone the repository

```bash
git clone https://github.com/cs683-iitb-autumn-2026/pa2.git
```

## Compile

To compile ChampSim, specify three parameters: the L1D prefetcher, the L2C replacement policy, and the binary name.

For example, `./build_champsim.sh no lru baseline` builds a single-core processor with a hashed perceptron branch predictor, no L1D data prefetcher, and the LRU replacement policy for the L2C.

You can give the binary any name. Custom names help distinguish binaries that use the same configuration.

```bash
./build_champsim.sh ${L1D_PREFETCHER} ${L2C_REPLACEMENT} ${BINARY_NAME}

./build_champsim.sh no lru baseline
```

## Run simulation

```bash
./bin/[BINARY] -warmup_instructions [N_WARM] -simulation_instructions [N_SIM] -traces [TRACE_DIR]/[TRACE]
./bin/hashed_perceptron-no-no-no-no-no-no-no-lru-lru-lru-lru-lru-lru-lru-lru-1core-baseline -warmup_instructions 25000000 -simulation_instructions 25000000 -traces ../traces/trace1.champsimtrace.xz
```

Where:
- `${BINARY}`: ChampSim binary compiled by `build_champsim.sh` (for example, `hashed_perceptron-no-no-no-no-no-no-no-lru-lru-lru-lru-lru-lru-lru-lru-1core-baseline`)
- `${N_WARM}`: number of instructions for the warmup period (25 million)
- `${N_SIM}`: number of instructions for the detailed simulation (25 million)
- `${TRACE_DIR}`: directory containing the trace (for example, `../traces/`)
- `${TRACE}`: name of the trace (for example, `trace1.champsimtrace.xz`)

## Evaluate the simulation

ChampSim measures IPC (instructions per cycle) as its main performance metric. It also prints other useful metrics at the end of each simulation.

## Install GCC 7 on Ubuntu

```bash
sudo apt update
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
vim /etc/apt/sources.list
```

Update the last line with:

```bash
deb [arch=amd64] http://archive.ubuntu.com/ubuntu focal main universe
```

Then run:

```bash
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get install gcc-7
sudo apt-get install g++-7
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 0
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 0
```

If GCC and G++ are already present in `/usr/bin`, configure the alternatives using:

```bash
sudo update-alternatives --config g++
sudo update-alternatives --config gcc
```


## PA2 Tasks 2 and 3: what was added

| File | Purpose |
|------|---------|
| `replacement/irrip.l2c_repl` | Task 2A: instruction-conscious RRIP (I-RRIP) at the L2C |
| `replacement/dyn_irrip.l2c_repl` | Task 2B: set-dueling dynamic I-RRIP; Task 3: prefetch awareness (`DYN_PF_AWARE`) |
| `prefetcher/spp.l1d_pref` | Task 1 stride pattern prefetcher (SPP-Final, unchanged from Part 1), used in Task 3 |
| `inc/cache.h`, `src/cache.cc`, `src/ooo_cpu.cc`, `replacement/*` | `update_replacement_state(..., uint8_t isInstruction)`: every call site passes the packet's `instruction` flag |
| `src/page_table_walker.cc` | `handle_page_fault()` returned no value; with GCC >= 8 this makes the `-O3` binary crash, so `return 0;` was added (callers ignore the value) |
| `src/main.cc` | the single-core path now also calls `l2c_replacement_final_stats()` |
| `scripts/` | build/run/parse/report scripts (below) |
| `results/` | raw ChampSim output of every run and `summary.csv` |
| `Summary.pdf` | report for all tasks |

Every policy parameter is a compile-time macro with the chosen value as default.
`scripts/build.sh` builds a parameter variant without editing the policy file:

```bash
scripts/build.sh dyn_irrip_T40 no dyn_irrip DYN_THRESHOLD_PCT=40   # -> bin/dyn_irrip_T40
```

Reproduce all results (Linux / WSL; traces named `trace1.champsimtrace.gz` ... in `<trace_dir>`):

```bash
scripts/experiments.sh build
JOBS=16 scripts/experiments.sh run <trace_dir> results
python3 scripts/parse_results.py results results/summary.csv
python3 scripts/make_report.py results/summary.csv report
python scripts/build_pdf.py <task1_report.pdf> report/tasks23.html Summary.pdf   # Windows: Edge/Chrome + pypdf
```

`scripts/sync_wsl.sh` copies the repository into a Linux directory and strips CRLF line endings (needed when the
repository is checked out on Windows and built under WSL).

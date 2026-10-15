# ChampSim

ChampSim is a trace-based simulator for microarchitecture research.

Useful trace links:
- Traces (access requires an LDAP ID): https://drive.google.com/drive/folders/1zYchkn-M1auZp_l5wRkIzAcDYPkNtW3H?usp=sharing
- ChampSim wiki: https://champsim.github.io/ChampSim/master/index.html

## Clone the repository

```bash
git clone https://github.com/Shivraj1906/PA2-CS683-2026.git
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

## PA2 Tasks 2 and 3: Below are the set of changes we made for TASK 2 and 3.

| File | Purpose |
|------|---------|
| `replacement/irrip.l2c_repl` | Task 2A: instruction-conscious RRIP (I-RRIP) at the L2C |
| `replacement/dyn_irrip.l2c_repl` | Task 2B: set-dueling dynamic I-RRIP; Task 3: prefetch awareness (`DYN_PF_AWARE`) |
| `inc/cache.h`, `src/cache.cc`, `src/ooo_cpu.cc`, `replacement/*` | `update_replacement_state(..., uint8_t isInstruction)`: every call site passes the packet's `instruction` flag |
| `src/page_table_walker.cc` | This file gave error on GCC >=8, so we added a return value in `handle_page_fault()` which prevents the `-O3` binary from crashing, so `return 0;` was added |
| `src/main.cc` | the single-core path now also calls `l2c_replacement_final_stats()` |

Every policy parameter is a compile-time macro with the chosen value as default.
`scripts/build.sh` builds a parameter variant :

```bash
scripts/build.sh dyn_irrip_T40 no dyn_irrip DYN_THRESHOLD_PCT=40   # -> bin/dyn_irrip_T40
```
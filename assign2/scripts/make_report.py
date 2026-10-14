#!/usr/bin/env python3
"""Build the Task 2 / Task 3 report (HTML + figures) from results/summary.csv.

    python3 scripts/make_report.py results/summary.csv report

Writes report/tasks23.html and report/figures/*.png. scripts/build_pdf.ps1 turns
the HTML into a PDF and prepends the Task 1 report to make Summary.pdf.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plotlib import (TRACES, TRACE_LABELS, Results, bar_chart, hbar_chart, html_table,  # noqa: E402
                     line_chart, sweep_table)

csv_path, outdir = sys.argv[1], sys.argv[2]
figdir = os.path.join(outdir, "figures")
os.makedirs(figdir, exist_ok=True)
R = Results(csv_path)

FIG_NO = [0]
TAB_NO = [0]


def fig(name, caption, width="100%"):
    FIG_NO[0] += 1
    return ('<figure><img src="figures/%s.png" style="width:%s"><figcaption><b>Figure %d.</b> %s</figcaption></figure>'
            % (name, width, FIG_NO[0], caption))


def figrow(items):
    """items: list of (name, caption) rendered side by side, one figure number."""
    FIG_NO[0] += 1
    cells = "".join('<div><img src="figures/%s.png"><div class="sub">(%s) %s</div></div>' % (n, chr(97 + i), c)
                    for i, (n, c) in enumerate(items[:-1]))
    return '<figure class="row">%s</figure><figcaption class="rowcap"><b>Figure %d.</b> %s</figcaption>' % (
        cells, FIG_NO[0], items[-1])


def table(html, caption):
    TAB_NO[0] += 1
    return '<div class="tab">%s<div class="tabcap"><b>Table %d.</b> %s</div></div>' % (html, TAB_NO[0], caption)


def p(tag, metric="speedup", trace=None):
    return R.mean(tag, metric) if trace is None else R.get(tag, trace, metric)


def pct(x):
    return "%+.2f%%" % (100 * (x - 1))


def png(name):
    return os.path.join(figdir, name + ".png")


SP = lambda t: p(t)                       # geomean speedup over LRU  # noqa: E731

# --------------------------------------------------------------------------
# configurations
# --------------------------------------------------------------------------
LRU, SRRIP, IRRIP = "base_lru", "base_rrip", "irrip_def"
DYN = "d1_W2_L32"                          # final dynamic I-RRIP (file defaults)
SPP = "spp_lru"
SPP_DYN = "spp_dyn"                        # Task 3 final

ABL = [  # tag, instr ins, data ins, data hit, wb hit, eviction
    ("base_rrip",      "2", "2", "&rarr;0", "&rarr;0", "first at 3"),
    ("ia_I1D2",        "1", "2", "&rarr;0", "&rarr;0", "first at 3"),
    ("ia_I0D2",        "0", "2", "&rarr;0", "&rarr;0", "first at 3"),
    ("ia_I1D3",        "1", "3", "&rarr;0", "&rarr;0", "first at 3"),
    ("ia_I0D3",        "0", "3", "&rarr;0", "&rarr;0", "first at 3"),
    ("ia_I0D2_S1",     "0", "2", "&minus;1", "&rarr;0", "first at 3"),
    ("ia_I0D3_S1",     "0", "3", "&minus;1", "&rarr;0", "first at 3"),
    ("ia_I0D2_W0",     "0", "2", "&rarr;0", "none", "first at 3"),
    ("ia_I0D3_W0",     "0", "3", "&rarr;0", "none", "first at 3"),
    ("ia_I0D3_S1W0",   "0", "3", "&minus;1", "none", "first at 3"),
    ("ia_I0D2_E1",     "0", "2", "&rarr;0", "&rarr;0", "data first"),
    ("ia_I0D3_E1",     "0", "3", "&rarr;0", "&rarr;0", "data first"),
    ("ia_I0D2_W0E1",   "0", "2", "&rarr;0", "none", "data first"),
    ("irrip_def",      "0", "3", "&rarr;0", "none", "data first"),
    ("ia_I0D3_S1W0E1", "0", "3", "&minus;1", "none", "data first"),
]
ABL_NAME = {
    "base_rrip": "SRRIP", "ia_I1D2": "A1", "ia_I0D2": "A2", "ia_I1D3": "A3", "ia_I0D3": "A4",
    "ia_I0D2_S1": "P1", "ia_I0D3_S1": "P2", "ia_I0D2_W0": "P3", "ia_I0D3_W0": "P4", "ia_I0D3_S1W0": "P5",
    "ia_I0D2_E1": "E1", "ia_I0D3_E1": "E2", "ia_I0D2_W0E1": "E3", "irrip_def": "I-RRIP", "ia_I0D3_S1W0E1": "E4",
}

THRESH = [t for t in [5, 10, 15, 20, 25, 30, 40, 50, 60, 70] if R.has("d0_T%02d" % t)]
D0 = ["d0_T%02d" % t for t in THRESH]
def wtag(w, leaders=32):
    return ("d1_W3D2_L%d" % leaders) if w == 1.5 else ("d1_W%d_L%d" % (w, leaders))


WEIGHTS = [w for w in [1, 1.5, 2, 3, 4, 8] if R.has(wtag(w))]
D1W = [wtag(w) for w in WEIGHTS]

MPKI_COLS = [("ipc", "IPC", "{:.4f}"), ("speedup", "Speedup", "{:.4f}"),
             ("l2c_load_mpki", "L2C load MPKI", "{:.2f}"),
             ("l2c_instr_load_mpki", "L2 instr. load MPKI", "{:.2f}"),
             ("l2c_data_load_mpki", "L2 data load MPKI", "{:.2f}")]


def dyn_frac(tag):
    return " / ".join("%.2f" % R.get(tag, t, "dyn_irrip_fraction") for t in TRACES)


def dyn_ratio(tag):
    return " / ".join("%.2f" % R.get(tag, t, "dyn_mean_ratio") for t in TRACES)


def four_charts(prefix, tags, labels, extra=()):
    """speedup + the three L2 load MPKIs (+ extra metrics) as small grouped bar charts."""
    specs = [("speedup", "Speedup over LRU", True)] + [
        ("l2c_load_mpki", "L2C load MPKI", False),
        ("l2c_instr_load_mpki", "L2 instruction load MPKI", False),
        ("l2c_data_load_mpki", "L2 data load MPKI", False)] + list(extra)
    names = []
    for metric, ylabel, ref in specs:
        name = "%s_%s" % (prefix, metric)
        bar_chart(png(name), R, tags, labels, metric, ylabel, ref_line=1.0 if ref else None,
                  with_mean=True, legend_cols=min(len(tags), 4))
        names.append((name, ylabel))
    return names


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
abl_tags = [a[0] for a in ABL]
hbar_chart(png("t2a_ablation"), [SP(t) for t in abl_tags], [ABL_NAME[t] for t in abl_tags],
           "Geometric-mean speedup over LRU (3 traces)", ref=1.0, ref_label="LRU", size=(6.4, 3.6),
           highlight={"I-RRIP"})
t2a_figs = four_charts("t2a", [LRU, SRRIP, IRRIP], ["LRU", "SRRIP", "I-RRIP"])

# threshold sensitivity
line_chart(png("t2b_thresh_speedup"), THRESH, [[p(t, "speedup", tr) for t in D0] for tr in TRACES] +
           [[SP(t) for t in D0]], [TRACE_LABELS[t] for t in TRACES] + ["Geomean"],
           "Threshold T (%)", "Speedup over LRU", ref_line=1.0)
line_chart(png("t2b_thresh_frac"), THRESH, [[p(t, "dyn_irrip_fraction", tr) for t in D0] for tr in TRACES],
           [TRACE_LABELS[t] for t in TRACES], "Threshold T (%)", "Fraction of decisions = I-RRIP",
           ylim=(-0.05, 1.05))
line_chart(png("t2b_weight_speedup"), WEIGHTS, [[p(t, "speedup", tr) for t in D1W] for tr in TRACES] +
           [[SP(t) for t in D1W]], [TRACE_LABELS[t] for t in TRACES] + ["Geomean"],
           "Instruction-miss weight W", "Speedup over LRU", ref_line=1.0, categorical=True)
line_chart(png("t2b_weight_frac"), WEIGHTS, [[p(t, "dyn_irrip_fraction", tr) for t in D1W] for tr in TRACES],
           [TRACE_LABELS[t] for t in TRACES], "Instruction-miss weight W", "Fraction of decisions = I-RRIP",
           ylim=(-0.05, 1.05), categorical=True)
t2b_figs = four_charts("t2b", [LRU, SRRIP, IRRIP, DYN], ["LRU", "SRRIP", "I-RRIP", "Dyn-I-RRIP"])

HAVE_T3 = all(R.has(t) for t in ["spp_dyn", "spp_dyn_nopf", "spp_rrip", "spp_irrip"])
if HAVE_T3:
    T3_TAGS = [LRU, SPP, DYN, "spp_dyn_nopf", SPP_DYN]
    T3_LABELS = ["Baseline (no pf, LRU)", "SPP + LRU", "Dyn-I-RRIP (no pf)", "SPP + Dyn-I-RRIP", "SPP + Dyn-I-RRIP-PF"]
    t3_specs = [
        ("ipc", "IPC", None), ("speedup", "Speedup over baseline", 1.0),
        ("l1d_pf_accuracy", "L1D prefetch accuracy (%)", None), ("coverage", "L1D prefetch coverage (%)", None),
        ("l1d_load_mpki", "L1D load MPKI", None), ("l2c_load_mpki", "L2C load MPKI", None),
        ("l2c_instr_load_mpki", "L2 instruction load MPKI", None), ("l2c_data_load_mpki", "L2 data load MPKI", None),
    ]
    t3_figs = []
    for metric, ylabel, ref in t3_specs:
        tags, labels = T3_TAGS, T3_LABELS
        if metric in ("l1d_pf_accuracy", "coverage"):
            tags = [t for t in T3_TAGS if t.startswith("spp")]
            labels = [l for t, l in zip(T3_TAGS, T3_LABELS) if t.startswith("spp")]
        name = "t3_" + metric
        bar_chart(png(name), R, tags, labels, metric, ylabel, ref_line=ref, with_mean=True,
                  legend_cols=5 if len(tags) > 3 else 3, size=(6.8, 2.3))
        t3_figs.append((name, ylabel))

# --------------------------------------------------------------------------
# text
# --------------------------------------------------------------------------
H = []
H.append("""<div class="title">
<div class="course">CS683: Advanced Computer Architecture</div>
<h1>Programming Assignment II</h1>
<div class="subtitle">Task 2: Instruction-conscious L2 replacement (I-RRIP and Dynamic I-RRIP)<br>
Task 3: SPP with prefetch-aware Dynamic I-RRIP</div>
<table class="authors"><tr><td>Haresh Gupta</td><td>23b0931</td></tr><tr><td>Yash Singh</td><td>23b0934</td></tr>
<tr><td>Sai Divya Teja</td><td>24b1016</td></tr><tr><td>Shivam Singh Yadav</td><td>23b1007</td></tr></table>
</div>""")

H.append("""<h2>2 Setup and Metrics (Tasks 2 and 3)</h2>
<ul>
<li><b>System:</b> the ChampSim of the PA2 repository, single core, hashed-perceptron branch predictor,
L1I 32&nbsp;KB 8-way, L1D 48&nbsp;KB 12-way, <b>L2C 512&nbsp;KB 8-way (1024 sets)</b>, LLC 2&nbsp;MB 16-way, LRU everywhere
except the L2C. No prefetcher except where SPP is named. 25M warm-up and 25M simulated instructions on Trace1&ndash;Trace3.</li>
<li><b>Baseline</b> for every speedup: no prefetcher and LRU at the L2C (the repository default,
<code>./build_champsim.sh no lru</code>). SRRIP (<code>rrip.l2c_repl</code>) is shown as a second reference because I-RRIP is built on it.</li>
<li><b>Metrics:</b> IPC = instructions / cycles; speedup = IPC / IPC<sub>baseline</sub>; MPKI values are the ChampSim ROI statistics
<code>L2C LOAD MPKI</code>, <code>L2C INSTRUCTION LOAD MPKI</code> and <code>L2C DATA LOAD MPKI</code> (instruction + data = load);
L1D accuracy = useful / prefetches issued to the next level; L1D coverage = (M<sub>b</sub> &minus; M<sub>p</sub>) / M<sub>b</sub> over L1D load misses.
Averages over the traces: geometric mean for IPC and speedup, arithmetic mean otherwise.</li>
<li><b>Interface change (Task 2A requirement):</b> <code>update_replacement_state</code> now has a ninth parameter
<code>uint8_t isInstruction</code>. It was added to the function-pointer type and all declarations in <code>inc/cache.h</code>, to every
replacement file, and to all 11 call sites, which pass the <code>instruction</code> field of the packet that causes the update
(MSHR entry on a fill, RQ/WQ/PQ entry on a hit). The policy treats a line as an instruction line when
<code>isInstruction</code> is set and the request is a <code>LOAD</code> or <code>PREFETCH</code>; this is exactly the rule
<code>fill_cache()</code> uses for <code>BLOCK::instruction</code> and the rule behind the L2 instruction MPKI statistic.</li>
<li><b>Two harness fixes.</b> (1) <code>PAGE_TABLE_WALKER::handle_page_fault()</code> is declared <code>uint64_t</code> but had no
<code>return</code>. GCC&nbsp;&ge;&nbsp;8 treats the end of such a function as unreachable, and the unmodified repository built with
GCC&nbsp;11 crashes after a few thousand instructions; we added <code>return 0;</code> (all callers ignore the value).
(2) The single-core path of <code>main.cc</code> never called <code>l2c_replacement_final_stats()</code>; we added the call so the policy
can print its decision statistics. Neither change affects timing.</li>
<li><b>Reproducibility:</b> every configuration below is a line in <code>scripts/experiments.sh</code>. Policy parameters are
compile-time macros whose defaults are the chosen values; a sweep point is built by <code>scripts/build.sh</code> as a two-line wrapper that
<code>#define</code>s the macro and <code>#include</code>s the policy file.</li>
<li><b>SPP and compiler:</b> Task 3 uses the Task 1 prefetcher (<code>prefetcher/spp.l1d_pref</code>, SPP-Final) unchanged. Tasks 2 and 3 were
compiled with GCC&nbsp;11 under WSL; Task 1 used GCC&nbsp;7 in Docker. This shifts the absolute numbers slightly: the baseline IPC is
%s / %s / %s here and 0.2322 / 0.2511 / 0.3821 in Task 1, and the speedup of SPP is %s / %s / %s here and 1.0098 / 1.0133 / 1.0128 in
Task 1. Every comparison inside Tasks 2 and 3 uses the same build, so these comparisons are consistent.</li>
</ul>""" % (tuple("%.4f" % p(LRU, "ipc", t) for t in TRACES) + tuple("%.4f" % p(SPP, "speedup", t) for t in TRACES)))

# ---------------- Task 2A ----------------
srrip_i = p(SRRIP, "l2c_instr_load_mpki"); irrip_i = p(IRRIP, "l2c_instr_load_mpki")
srrip_d = p(SRRIP, "l2c_data_load_mpki"); irrip_d = p(IRRIP, "l2c_data_load_mpki")
H.append("""<h2>3 Task 2A: Instruction-Conscious L2 Replacement (I-RRIP)</h2>
<h3>3.1 Why instruction lines deserve priority at the L2</h3>
<p>An L1D miss that hits in the L2 costs about 10 extra cycles, and the out-of-order core can overlap much of that latency
with independent work (the ROB has 352 entries). An L1I miss stalls the fetch unit: no new instructions enter the pipeline until the
line arrives, so its latency adds almost directly to execution time. Our traces have large code footprints; even with SRRIP the L2
instruction load MPKI is %.2f, %.2f and %.2f, i.e. %.0f%% of all L2 load misses on Trace2. Keeping instruction lines in the L2 turns
expensive front-end stalls into short L2 hits, and the cost is extra data misses, which are partly hidden by the core.</p>

<h3>3.2 Design</h3>
<p>I-RRIP keeps the SRRIP hardware: a 2-bit re-reference prediction value (RRPV) per line, a victim is a line with RRPV&nbsp;3, and if
no line has RRPV&nbsp;3 all lines of the set age by one. It changes all three parts of the policy:</p>
<table class="data compact"><thead><tr><th>Part</th><th>SRRIP</th><th>I-RRIP</th></tr></thead><tbody>
<tr><td>Insertion</td><td>every line at RRPV 2</td><td>instruction line at RRPV <b>0</b>, data line at RRPV <b>3</b></td></tr>
<tr><td>Promotion</td><td>every hit &rarr; RRPV 0</td><td>instruction and data demand hits &rarr; RRPV 0; <b>a writeback hit does not promote</b></td></tr>
<tr><td>Eviction</td><td>first line (lowest way) with RRPV 3</td><td>among lines with RRPV 3, <b>a data line before an instruction line</b></td></tr>
</tbody></table>
<p>Hardware cost over SRRIP: one instruction bit per line (8192 bits for the 512&nbsp;KB L2; ChampSim already keeps it in
<code>BLOCK::instruction</code>). The other changes are only different constants and one extra comparison in the victim search.</p>

<h3>3.3 How each part was chosen: ablation</h3>
<p>We changed one part at a time, starting from SRRIP, and kept a change only if it raised the geometric-mean speedup by more than
0.05%% (smaller gaps are ties; with three traces we cannot separate them from noise, the same rule as in Task 1). Table %d lists
every variant, Figure %d shows the geometric-mean speedups.</p>""" % (
    p(SRRIP, "l2c_instr_load_mpki", "trace1"), p(SRRIP, "l2c_instr_load_mpki", "trace2"), p(SRRIP, "l2c_instr_load_mpki", "trace3"),
    100 * p(SRRIP, "l2c_instr_load_mpki", "trace2") / p(SRRIP, "l2c_load_mpki", "trace2"), TAB_NO[0] + 1, FIG_NO[0] + 1))

rows = []
for tag, ii, di, dh, wb, ev in ABL:
    cls = ' class="bold"' if tag == IRRIP else ""
    rows.append("<tr%s><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td>%s<td>%.4f</td><td>%.2f</td><td>%.2f</td></tr>" % (
        cls, ABL_NAME[tag], ii, di, dh, wb, ev,
        "".join("<td>%.4f</td>" % p(tag, "speedup", t) for t in TRACES), SP(tag),
        p(tag, "l2c_instr_load_mpki"), p(tag, "l2c_data_load_mpki")))
abl_html = ('<table class="data"><thead><tr><th>Name</th><th>Instr. insert</th><th>Data insert</th><th>Data hit</th><th>WB hit</th>'
            '<th>Victim</th><th>Speedup T1</th><th>Speedup T2</th><th>Speedup T3</th><th>Geomean</th><th>L2 instr. MPKI (mean)</th>'
            '<th>L2 data MPKI (mean)</th></tr></thead><tbody>%s</tbody></table>') % "".join(rows)
H.append(table(abl_html, "I-RRIP ablation. Insert columns are RRPVs; &ldquo;Data hit&rdquo; &rarr;0 means promotion to RRPV 0, "
               "&minus;1 means one step closer; &ldquo;WB hit&rdquo; is the promotion on a writeback hit. Speedup over LRU. "
               "Bold: the submitted I-RRIP."))
H.append(fig("t2a_ablation", "Geometric-mean speedup over LRU of every ablation variant (dashed line: LRU). "
                              "Orange: the submitted I-RRIP.", "86%"))

g = SP
H.append("""<p><b>Insertion.</b> Inserting instruction lines at RRPV 0 instead of 2 (A2: %s) gives them two extra aging rounds before they can be
evicted; RRPV 1 (A1: %s) protects them less and is worse. The larger step is inserting data at RRPV 3 (A4: %s, A3: %s): a new data line is
then the first victim of its set unless it is reused before the next miss. Most L2 data lines are not reused before eviction (the
L2 load hit rate of SRRIP is only 30%%, 49%% and 27%%), so distant insertion removes little useful data and gives the instruction
lines the room. Both changes are kept: instruction at 0, data at 3.</p>
<p><b>Promotion.</b> Two promotion changes were tested. (i) Promoting a data hit only one step (P1, P2) instead of to RRPV 0 helps a
little on its own (A4 &rarr; P2: %s &rarr; %s). (ii) Not promoting on a <i>writeback</i> hit is the most important single change
(A4 &rarr; P4: %s &rarr; %s). Writebacks are dirty lines evicted from the L1D; they are 11&ndash;25%% of all L2 accesses and over 92%% of them
hit in the L2 (the line is still there from its fill). Under SRRIP each such hit moves a data line to RRPV 0, i.e. the line the L1D has
just thrown away gets the highest priority in the L2, ahead of every instruction line. A writeback is not a reuse by the core, so it
should not change the prediction. Once writebacks no longer promote, the partial data promotion gives nothing more (P4 %s vs P5 %s, a tie), so
demand hits keep the SRRIP rule (&rarr; RRPV 0): a data line that is really reused deserves to stay, and the simpler rule is kept.</p>
<p><b>Eviction.</b> Choosing a data line before an instruction line among the RRPV-3 candidates matters when both kinds age to RRPV 3
together, which is common when data is inserted at 2 (A2 &rarr; E1: %s &rarr; %s). With data inserted at 3, most RRPV-3 lines are already
data lines, and the gain in speedup is a tie (P4 %s vs I-RRIP %s). We keep it because it is the only part that protects an
<i>old</i> instruction line, and it further lowers the L2 instruction MPKI, the stated goal of the task (Trace2: %.2f &rarr; %.2f),
at no extra cost beyond the instruction bit.</p>
<p><b>Why this combination is the best of the variants.</b> Every part is aimed at the same effect, keeping instruction lines and
dropping data that is not reused, and each part was checked against the variant without it. The submitted I-RRIP has the highest
geometric-mean speedup together with P4/P5/E4 (all within 0.01%%) and, with E4, the lowest mean L2 instruction MPKI among them; E4 adds the partial data promotion without any gain, so the simpler I-RRIP is submitted. The price is visible
on Trace1, the trace whose data is reused in the L2: its data MPKI rises from %.2f to %.2f and its speedup falls to %.4f. Task 2B fixes this.</p>""" % (
    pct(g("ia_I0D2")), pct(g("ia_I1D2")), pct(g("ia_I0D3")), pct(g("ia_I1D3")),
    pct(g("ia_I0D3")), pct(g("ia_I0D3_S1")), pct(g("ia_I0D3")), pct(g("ia_I0D3_W0")),
    pct(g("ia_I0D3_W0")), pct(g("ia_I0D3_S1W0")),
    pct(g("ia_I0D2")), pct(g("ia_I0D2_E1")), pct(g("ia_I0D3_W0")), pct(g(IRRIP)),
    p("ia_I0D3_W0", "l2c_instr_load_mpki", "trace2"), p(IRRIP, "l2c_instr_load_mpki", "trace2"),
    p(SRRIP, "l2c_data_load_mpki", "trace1"), p(IRRIP, "l2c_data_load_mpki", "trace1"), p(IRRIP, "speedup", "trace1")))

H.append("<h3>3.4 Results</h3>")
H.append(table(html_table(R, [LRU, SRRIP, IRRIP], ["LRU (baseline)", "SRRIP", "I-RRIP"], MPKI_COLS, bold={IRRIP}),
               "Task 2A: IPC, speedup over LRU and L2 load MPKIs (no prefetcher)."))
H.append(figrow(t2a_figs + ["Task 2A: speedup and L2C / L2 instruction / L2 data load MPKI of LRU, SRRIP and I-RRIP."]))
H.append("""<p>I-RRIP lowers the mean L2 instruction load MPKI from %.2f (SRRIP) to %.2f (%.0f%% fewer instruction misses) and raises the
data load MPKI from %.2f to %.2f. On Trace2, where instructions cause most L2 misses, this is a %s speedup over LRU and the total L2 load
MPKI also falls. On Trace3 the instruction misses almost disappear (%.2f &rarr; %.2f). On Trace1 the extra data misses outweigh the saved
instruction misses.</p>""" % (srrip_i, irrip_i, 100 * (1 - irrip_i / srrip_i), srrip_d, irrip_d, pct(p(IRRIP, "speedup", "trace2")),
                               p(SRRIP, "l2c_instr_load_mpki", "trace3"), p(IRRIP, "l2c_instr_load_mpki", "trace3")))

# ---------------- Task 2B ----------------
H.append("""<h2>4 Task 2B: Dynamic I-RRIP</h2>
<h3>4.1 Design</h3>
<p><code>replacement/dyn_irrip.l2c_repl</code> uses set dueling. A few <b>leader sets always run SRRIP</b>, and in the final design
an equal number of leader sets always run I-RRIP; all other sets (followers) use the policy that is currently winning.
Leader sets are chosen by complement-select: the 1024 sets are split into N constituencies of 1024/N sets, and in constituency
<i>k</i> the set with offset <i>k</i> is an SRRIP leader and the set with offset (1024/N&minus;1&minus;<i>k</i>) an I-RRIP leader.
This spreads the leaders over the whole index range, so they do not line up with power-of-two strides of the program.</p>
<p>In the leader sets the policy counts instruction and data (non-instruction) <b>demand</b> accesses and misses (writebacks are
excluded). Two decision rules were implemented (<code>DYN_MODE</code>):</p>
<ul>
<li><b>Mode 0: the formula of the hint.</b> In the SRRIP leaders, ratio = (instr. accesses + instr. misses) / (total accesses + total
misses). Every <code>DYN_EPOCH</code> = 1024 leader accesses the followers use I-RRIP if ratio &gt; T, then the four counters are halved so
that old program phases fade out. The ratio is measured in SRRIP sets on purpose: in I-RRIP sets the policy we are judging would already
have removed instruction misses and biased the measurement.</li>
<li><b>Mode 1: miss duel (submitted).</b> A 10-bit saturating counter PSEL starts at 512. A demand miss in an SRRIP leader adds its
weight to PSEL, a miss in an I-RRIP leader subtracts it. A data miss weighs 1, an instruction miss weighs <b>W = 2</b>. Every 1024 SRRIP-leader
accesses the followers use I-RRIP if PSEL &ge; 512, i.e. if the I-RRIP leaders have produced fewer weighted misses.</li>
</ul>
<h3>4.2 The formula of the hint: threshold sensitivity</h3>""")
H.append(table(sweep_table(R, D0, ["T = %d%%" % t for t in THRESH], "Threshold",
                           [(dyn_frac, "Fraction I-RRIP (T1/T2/T3)", None), (dyn_ratio, "Mean ratio (T1/T2/T3)", None)]),
               "Mode 0 (formula of the hint) with 10 thresholds; 32 SRRIP leader sets, epoch 1024. &ldquo;Fraction I-RRIP&rdquo; is the share "
               "of ROI decisions that chose I-RRIP, &ldquo;mean ratio&rdquo; the average measured ratio."))
H.append(figrow([("t2b_thresh_speedup", "speedup"), ("t2b_thresh_frac", "share of decisions that chose I-RRIP"),
                 "Mode 0: sensitivity to the threshold T."]))
best_t = max(D0, key=SP)
H.append("""<p>The measured ratio is stable per trace: about %.2f on Trace1, %.2f on Trace2 and %.2f on Trace3. Low thresholds
(T&nbsp;&le;&nbsp;15%%) choose I-RRIP almost always and match static I-RRIP; T&nbsp;=&nbsp;70%% never chooses it and equals SRRIP. In between, the
switch happens in the order Trace3, Trace1, Trace2, because that is the order of their ratios. But the traces need the opposite for Trace1 and
Trace3: <b>Trace3 gains from I-RRIP</b> (%s) although only about 31%% of its leader traffic is instruction traffic, and <b>Trace1 loses</b>
(%s) although 44%% is. A threshold between the two ratios (T = 30&ndash;40%%) gives Trace3 SRRIP and Trace1 I-RRIP, the worst assignment.
T = 50%% gets Trace1 right but still takes I-RRIP away from Trace3. The best threshold is the lowest (%s, %s), i.e. the formula is
best when it does nothing.</p>
<p>The reason: the ratio measures how much of the L2 traffic is instruction traffic, not whether protecting it pays off. That depends
on what the protection costs: on Trace1 the data has L2 reuse, and moving data insertion to RRPV 3 adds %.2f data MPKI while removing only
%.2f instruction MPKI. To decide correctly the policy has to measure the effect of I-RRIP itself, which is what set dueling with
I-RRIP leader sets does.</p>""" % (
    p("d0_T25", "dyn_mean_ratio", "trace1"), p("d0_T25", "dyn_mean_ratio", "trace2"), p("d0_T25", "dyn_mean_ratio", "trace3"),
    pct(p(IRRIP, "speedup", "trace3")), pct(p(IRRIP, "speedup", "trace1")), "T = %d%%" % int(best_t[4:]),
    pct(SP(best_t)),
    p(IRRIP, "l2c_data_load_mpki", "trace1") - p(SRRIP, "l2c_data_load_mpki", "trace1"),
    p(SRRIP, "l2c_instr_load_mpki", "trace1") - p(IRRIP, "l2c_instr_load_mpki", "trace1")))

H.append("<h3>4.3 Miss duel: sensitivity of weight, leader sets and epoch</h3>")
H.append(table(sweep_table(R, D1W, ["W = %g" % w for w in WEIGHTS], "Instr. miss weight",
                           [(dyn_frac, "Fraction I-RRIP (T1/T2/T3)", None)], bold={DYN}),
               "Mode 1 with six instruction-miss weights W (a data miss weighs 1; W = 1.5 is implemented as steps of 3 and 2), "
               "32 + 32 leader sets, epoch 1024. Bold: submitted."))
H.append(figrow([("t2b_weight_speedup", "speedup"), ("t2b_weight_frac", "share of decisions that chose I-RRIP"),
                 "Mode 1: sensitivity to the instruction-miss weight W."]))
GRID_W = [1, 1.5, 2]
GRID_L = [16, 32, 64]
grid_rows = []
for w in GRID_W:
    cells = []
    for L in GRID_L:
        t = wtag(w, L)
        if not R.has(t):
            cells.append("<td>&ndash;</td>")
            continue
        right = (R.get(t, "trace1", "dyn_irrip_fraction") < 0.5 and R.get(t, "trace2", "dyn_irrip_fraction") > 0.5
                 and R.get(t, "trace3", "dyn_irrip_fraction") > 0.5)
        cells.append("<td%s>%.4f<br><span class='small'>%s%s</span></td>" % (
            ' class="hl"' if t == DYN else "", SP(t), dyn_frac(t), " &#10003;" if right else ""))
    grid_rows.append("<tr><td>W = %g</td>%s</tr>" % (w, "".join(cells)))
grid_html = ('<table class="data grid"><thead><tr><th>Weight</th>%s</tr></thead><tbody>%s</tbody></table>'
             % ("".join("<th>%d + %d leader sets</th>" % (L, L) for L in GRID_L), "".join(grid_rows)))
H.append(table(grid_html, "Geometric-mean speedup for every combination of weight and number of leader sets, with the share of "
               "decisions that chose I-RRIP on Trace1 / Trace2 / Trace3 (&#10003;: SRRIP on Trace1 and I-RRIP on Trace2 and Trace3, "
               "the best static choice per trace). Highlighted: submitted."))
grid_tab = TAB_NO[0]
EP = [("d1_W2_L32_E256", "epoch 256"), (DYN, "epoch 1024 (submitted)"), ("d1_W2_L32_E4096", "epoch 4096"),
      ("d1_W1_L32_E256", "W = 1, epoch 256"), ("d1_W1_L32", "W = 1, epoch 1024"), ("d1_W1_L32_E4096", "W = 1, epoch 4096"),
      ("d0_T25_E256", "mode 0, T = 25%, epoch 256"), ("d0_T25", "mode 0, T = 25%, epoch 1024"),
      ("d0_T25_E4096", "mode 0, T = 25%, epoch 4096")]
EP = [(t, l) for t, l in EP if R.has(t)]
H.append(table(sweep_table(R, [t for t, _ in EP], [l for _, l in EP], "Variant (mode 1, W = 2, 32 + 32 leaders unless stated)",
                           [(dyn_frac, "Fraction I-RRIP (T1/T2/T3)", None)], bold={DYN}),
               "Epoch length: the followers re-read the decision every <i>epoch</i> SRRIP-leader accesses."))
ep_tab = TAB_NO[0]
H.append("""<p><b>Weight.</b> With W = 1 (a plain miss count) and 32 + 32 leaders the duel picks SRRIP for Trace1 and I-RRIP for Trace2,
but also SRRIP for Trace3, whose two policies differ by only a few misses. W = 2 picks I-RRIP for Trace3 as well and gives %s. W = 3 is a
tie (%s); W = 4 and 8 over-value instruction misses and choose I-RRIP for Trace1 again. W = 2 also has a physical meaning: an instruction
miss stalls fetch while the core overlaps part of a data miss with other work, so an instruction miss costs roughly twice as much.</p>
<p><b>Leader sets and weight together (Table %d).</b> The leaders are a sample of the cache, and on Trace1 and Trace3 the two policies are
close: on Trace1 I-RRIP adds about 2.5 data MPKI and removes about 1.2 instruction MPKI, so with W = 2 the weighted misses of the two
leader groups are almost equal. Which policy wins then depends on which sets are sampled, and neighbouring cells of the grid disagree on
Trace1 or Trace3. Three settings choose the best policy on every trace (marked &#10003;): 16 + 16 leaders with W = 1 or 1.5, and
32 + 32 leaders with W = 2. They are within 0.02%% of each other, a tie. We submit 32 + 32 leaders with W = 2: 32 leader sets per policy
(6.25%% of the 1024 sets together) is the size used by DIP for caches of this size, and it halves the sampling noise of 16 sets (the
standard error of a sampled miss rate falls with the square root of the number of sets). 64 + 64 leaders run 12.5%% of the cache with a fixed
policy, and in our runs they also led to I-RRIP on Trace1. A wrong choice is cheap on these traces: the worst cell of the grid is %s,
still better than SRRIP (%s) and close to static I-RRIP (%s), because the duel is only unsure where the two policies are close.</p>
<p><b>Epoch (Table %d).</b> PSEL integrates continuously; the epoch only sets how often the followers read it. Trace3 sends few requests to
the L2, so with 1024 it makes only 7 decisions in 25M instructions, which is still enough because its preference does not change. For
the duel all three epochs give the same decisions (within 0.02%%). For mode 0 the epoch changes which phases are sampled, but no epoch
makes the formula choose the right policies.</p>""" % (
    pct(SP(DYN)), pct(SP("d1_W3_L32")), grid_tab,
    pct(min(SP(wtag(w, L)) for w in GRID_W for L in GRID_L if R.has(wtag(w, L)))), pct(SP(SRRIP)), pct(SP(IRRIP)), ep_tab))

H.append("<h3>4.4 Results</h3>")
H.append(table(html_table(R, [LRU, SRRIP, IRRIP, DYN], ["LRU (baseline)", "SRRIP", "I-RRIP", "Dyn-I-RRIP"], MPKI_COLS, bold={DYN}),
               "Task 2B: IPC, speedup over LRU and L2 load MPKIs (no prefetcher). Dyn-I-RRIP = mode 1, W = 2, 32 + 32 leaders, epoch 1024."))
H.append(figrow(t2b_figs + ["Task 2B: speedup and L2C / L2 instruction / L2 data load MPKI."]))
H.append("""<p>Dynamic I-RRIP keeps the gains of I-RRIP where they exist (Trace2 %s, Trace3 %s) and follows SRRIP on Trace1 (%s instead of
%s with static I-RRIP), so it beats both SRRIP and I-RRIP in geometric mean (%s vs %s and %s). Trace1&rsquo;s data load MPKI goes back to the
SRRIP level (%.2f).</p>
<p><b>Storage.</b> On top of SRRIP (2-bit RRPV per line, 16384 bits): the instruction bit per line (8192 bits), a 10-bit PSEL, a 10-bit
epoch counter and the 1-bit follower mode, 8213 bits &asymp; 1.0 KiB in total. Leader sets are identified by comparing two 5-bit fields of the set
index, so no table is needed. (Mode 0 instead needs four 16-bit counters, 64 bits, instead of PSEL.)</p>""" % (
    pct(p(DYN, "speedup", "trace2")), pct(p(DYN, "speedup", "trace3")), pct(p(DYN, "speedup", "trace1")),
    pct(p(IRRIP, "speedup", "trace1")), pct(SP(DYN)), pct(SP(SRRIP)), pct(SP(IRRIP)), p(DYN, "l2c_data_load_mpki", "trace1")))

T3_SECTION = "@@T3@@"
H.append(T3_SECTION)

css = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "report.css")).read()
body = "\n".join(H)
if HAVE_T3:
    import t3_text
    body = body.replace(T3_SECTION, t3_text.section(R, table, fig, figrow, t3_figs, T3_TAGS, T3_LABELS, pct))
else:
    body = body.replace(T3_SECTION, "<h2>5 Task 3</h2><p>(results pending)</p>")
html = "<!doctype html><html><head><meta charset='utf-8'><title>CS683 PA2 Tasks 2 and 3</title><style>%s</style></head><body>%s</body></html>" % (css, body)
open(os.path.join(outdir, "tasks23.html"), "w", encoding="utf-8").write(html)
print("wrote", os.path.join(outdir, "tasks23.html"))

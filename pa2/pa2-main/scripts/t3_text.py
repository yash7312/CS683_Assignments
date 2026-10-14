"""Task 3 section of the report (called from make_report.py)."""
from plotlib import TRACES, html_table, sweep_table

VARIANTS = [
    ("spp_lru", "SPP + LRU"),
    ("spp_rrip", "SPP + SRRIP"),
    ("spp_irrip", "SPP + I-RRIP"),
    ("spp_dyn_nopf", "SPP + Dyn-I-RRIP (pf-unaware)"),
    ("spp_dyn_ins2", "PF-aware: insert pf at 2"),
    ("spp_dyn_use1", "PF-aware: first use to 1"),
    ("spp_dyn_use2", "PF-aware: first use to 2"),
    ("spp_dyn_noev", "PF-aware: no pf-first evict"),
    ("spp_dyn_stat", "PF-aware: pf in PSEL"),
    ("spp_dyn", "SPP + Dyn-I-RRIP-PF"),
]


def section(R, table, fig, figrow, t3_figs, T3_TAGS, T3_LABELS, pct):
    sp = lambda t, tr=None: R.mean(t, "speedup") if tr is None else R.get(t, tr, "speedup")  # noqa: E731
    have = [(t, l) for t, l in VARIANTS if R.has(t)]
    H = []
    H.append("""<h2>5 Task 3: SPP with Prefetch-Aware Dynamic I-RRIP</h2>
<h3>5.1 What the L2 sees from SPP</h3>
<p>SPP (Task 1) runs at the L1D. Each prefetch first looks up the L1D; on a miss it goes to the L2 as a request of type
<code>PREFETCH</code>, and if it misses there too, the line is filled into the L2 on its way to the L1D (SPP fills the L1D when less than
half of the L1D MSHRs are busy, otherwise only the L2). The replacement policy therefore sees three new events: a prefetch fill, a
prefetch request that hits a line already in the L2, and later perhaps a demand hit on a line that a prefetch brought in
(<code>BLOCK::prefetch</code> is still set; the cache clears it right after calling the policy).</p>
<p>The policy counts these events. On the three traces only %s of the prefetched lines filled into the L2 ever receive a demand hit at the
L2 (columns <i>pf fills</i> and <i>first use</i> below, counted over warm-up and simulation): the useful prefetches are consumed in the L1D, and the L2 copy is a duplicate that
the program will not ask the L2 for again unless the L1D loses it first.</p>
<h3>5.2 Prefetch-aware design</h3>
<p>The prefetch handling is part of <code>replacement/dyn_irrip.l2c_repl</code> (<code>DYN_PF_AWARE</code>, on by default). It applies to
data prefetches in all sets, in both follower modes:</p>
<ol>
<li><b>Insertion:</b> a prefetch fill is inserted at RRPV 3 (distant). It is a guess, not a demand, and its likely use is in the L1D.
In I-RRIP mode this is what data gets anyway; in SRRIP mode (Trace1) it keeps prefetched lines from pushing out demand data inserted at 2.</li>
<li><b>No promotion on a prefetch hit:</b> a prefetch request that hits in the L2 is about to copy the line into the L1D; it says nothing
about future L2 reuse. SRRIP and I-RRIP would move the line to RRPV 0.</li>
<li><b>First demand use:</b> the first demand hit on a prefetched line proves the prefetch right; the line is promoted to RRPV 0 like any
reused line.</li>
<li><b>Eviction:</b> among the RRPV-3 candidates, an unused prefetched data line is evicted first, then other data, then instructions.</li>
<li><b>Set dueling sees demand traffic only:</b> prefetch misses are not counted in PSEL (and not in the mode-0 ratio). Prefetch traffic
depends on the prefetcher, not on how well a replacement policy keeps instruction or data lines; counting it would add data misses to
both leader groups and dilute the instruction-miss signal the duel is based on.</li>
</ol>
<p>None of this needs new state: <code>BLOCK::prefetch</code> already exists in every cache line. With no prefetcher none of these paths runs,
so the Task 2B results are exactly the results of this file.</p>""" % _first_use_share(R))

    H.append("<h3>5.3 Which L2 policy to combine with SPP, and which prefetch rules matter</h3>")
    H.append(table(sweep_table(R, [t for t, _ in have], [l for _, l in have], "Configuration (SPP at L1D)",
                               [("l1d_pf_accuracy", "L1D acc. (%)", "{:.1f}"),
                                ("l2c_instr_load_mpki", "L2 instr. MPKI", "{:.2f}"),
                                ("l2c_data_load_mpki", "L2 data MPKI", "{:.2f}"),
                                (lambda t: _pf_counts(R, t), "pf fills / first use", None)],
                               bold={"spp_dyn"}),
                   "SPP combined with each L2 policy, and the prefetch-aware variants (one rule changed at a time). "
                   "Speedup over the baseline without prefetcher and with LRU; accuracy and MPKI are means over the traces; "
                   "pf fills / first use: prefetch fills into the L2 and first demand uses of such lines, summed over the traces. "
                   "Bold: submitted."))
    best_other = max([t for t, _ in have if t != "spp_dyn"], key=sp)
    H.append(_variant_text(R, sp, pct, have, best_other))

    H.append("<h3>5.4 Results</h3>")
    cols = [("ipc", "IPC", "{:.4f}"), ("speedup", "Speedup", "{:.4f}"), ("l1d_pf_accuracy", "L1D accuracy (%)", "{:.1f}"),
            ("coverage", "L1D coverage (%)", "{:.2f}"), ("l1d_load_mpki", "L1D load MPKI", "{:.2f}"),
            ("l2c_load_mpki", "L2C load MPKI", "{:.2f}"), ("l2c_instr_load_mpki", "L2 instr. MPKI", "{:.2f}"),
            ("l2c_data_load_mpki", "L2 data MPKI", "{:.2f}")]
    H.append(table(html_table(R, T3_TAGS, T3_LABELS, cols, bold={"spp_dyn"}),
                   "Task 3: IPC, speedup over the baseline (no prefetcher, LRU), L1D prefetch accuracy and coverage, and load MPKI of the "
                   "L1D, L2C, L2 instructions and L2 data. Accuracy and coverage are undefined without a prefetcher (&ndash;)."))
    for name, ylabel in t3_figs:
        H.append(fig(name, "Task 3: %s." % ylabel, "92%"))
    s_final, s_spp, s_dyn = sp("spp_dyn"), sp("spp_lru"), sp("d1_W2_L32")
    per = []
    for tr in TRACES:
        per.append("%s: %.4f vs %.4f (SPP) and %.4f (Dyn-I-RRIP)" % (tr.replace("trace", "Trace"), sp("spp_dyn", tr),
                                                                      sp("spp_lru", tr), sp("d1_W2_L32", tr)))
    H.append("""<p><b>Outcome.</b> SPP + prefetch-aware Dyn-I-RRIP reaches a geometric-mean speedup of %s, against %s for standalone SPP
(with LRU) and %s for Dyn-I-RRIP without prefetching, so it outperforms both. Per trace: %s. The two mechanisms attack different misses
and add up: SPP removes L1D data misses (coverage %.1f%%, L1D load MPKI %.2f &rarr; %.2f), and Dyn-I-RRIP removes L2 instruction misses
(mean %.2f &rarr; %.2f), which SPP cannot touch. On Trace1, where the duel keeps SRRIP, the result stays within %.2f%% of SPP + LRU; the L2
policy has almost no effect on that trace (LRU and SRRIP differ by 0.04%% there as well).</p>
<p><b>SPP accuracy and coverage</b> are the same with every L2 policy (accuracy %.1f%% vs %.1f%%, coverage %.2f%% vs %.2f%% for SPP + LRU vs
the submitted combination): the L2 policy does not change what SPP predicts, only how long the L2 keeps the lines.</p>""" % (
        pct(s_final), pct(s_spp), pct(s_dyn), "; ".join(per),
        R.mean("spp_dyn", "coverage"), R.mean("base_lru", "l1d_load_mpki"), R.mean("spp_dyn", "l1d_load_mpki"),
        R.mean("spp_lru", "l2c_instr_load_mpki"), R.mean("spp_dyn", "l2c_instr_load_mpki"),
        100 * abs(sp("spp_dyn", "trace1") / sp("spp_lru", "trace1") - 1),
        R.mean("spp_lru", "l1d_pf_accuracy"), R.mean("spp_dyn", "l1d_pf_accuracy"),
        R.mean("spp_lru", "coverage"), R.mean("spp_dyn", "coverage")))
    return "\n".join(H)


def _pf_counts(R, tag):
    f = u = 0
    for tr in TRACES:
        row = R.rows[(tag, tr)]
        f += int(row.get("dyn_pf_fills") or 0)
        u += int(row.get("dyn_pf_first_use") or 0)
    return "%d / %d" % (f, u) if f else "&ndash;"


def _first_use_share(R):
    parts = []
    for tr in TRACES:
        row = R.rows[("spp_dyn", tr)]
        f, u = int(row.get("dyn_pf_fills") or 0), int(row.get("dyn_pf_first_use") or 0)
        parts.append("%.1f%%" % (100.0 * u / f) if f else "&ndash;")
    return " / ".join(parts)


def _variant_text(R, sp, pct, have, best_other):
    g = {t: sp(t) for t, _ in have}
    d = lambda t: 100 * (g["spp_dyn"] / g[t] - 1)  # noqa: E731
    i3 = lambda t, tr: R.get(t, tr, "l2c_instr_load_mpki")  # noqa: E731
    fr = lambda t, tr: R.get(t, tr, "dyn_irrip_fraction")  # noqa: E731
    return """<p><b>Policy.</b> Combined with SPP, the L2 policies keep their Task 2 order: SRRIP %s, I-RRIP %s, prefetch-unaware
Dyn-I-RRIP %s (SPP + LRU: %s). The instruction-miss savings of I-RRIP add to the data-miss savings of SPP. Dyn-I-RRIP beats static I-RRIP
with SPP as well, again by leaving Trace1 on SRRIP (Trace1: %.4f vs %.4f).</p>
<p><b>Prefetch rules, one at a time.</b> The complete prefetch-aware policy (%s) and the prefetch-unaware one (%s) are a tie: since
I-RRIP already inserts every data line, prefetched or not, at RRPV 3, most prefetch rules repeat what the unaware policy does in I-RRIP
mode. Removing single rules from the aware policy shows which ones carry weight:</p>
<ul>
<li><b>Distant insertion matters.</b> Inserting prefetches at RRPV 2 instead of 3 costs %.2f%%. The prefetched lines then outlive the
other data lines and push out instruction lines: the L2 instruction MPKI rises on Trace2 (%.2f &rarr; %.2f) and Trace3 (%.2f &rarr; %.2f).</li>
<li><b>Keeping prefetches out of the duel matters.</b> Counting prefetch misses in PSEL costs %.2f%%. The extra misses are data misses that
both leader groups suffer alike; they add noise but no signal, and on Trace3, which has few L2 demand misses, they drag PSEL to SRRIP
(share of I-RRIP decisions %.2f instead of %.2f).</li>
<li><b>First-use promotion and prefetch-first eviction are ties</b> (promotion to RRPV 1: %s, to RRPV 2: %s, no prefetch-first eviction:
%s, vs %s). Only %s of the prefetched lines are ever used at the L2, so the rule for their first use rarely runs; and with data
inserted at RRPV 3 the RRPV-3 candidates are nearly always prefetched or other dead data lines anyway. We keep both because they are free and
follow the same reasoning, but the data does not show that they help.</li>
</ul>""" % (
        pct(g["spp_rrip"]), pct(g["spp_irrip"]), pct(g["spp_dyn_nopf"]), pct(g["spp_lru"]),
        sp("spp_dyn_nopf", "trace1"), sp("spp_irrip", "trace1"),
        pct(g["spp_dyn"]), pct(g["spp_dyn_nopf"]),
        d("spp_dyn_ins2"), i3("spp_dyn", "trace2"), i3("spp_dyn_ins2", "trace2"), i3("spp_dyn", "trace3"), i3("spp_dyn_ins2", "trace3"),
        d("spp_dyn_stat"), fr("spp_dyn_stat", "trace3"), fr("spp_dyn", "trace3"),
        pct(g["spp_dyn_use1"]), pct(g["spp_dyn_use2"]), pct(g["spp_dyn_noev"]), pct(g["spp_dyn"]), _first_use_share(R))

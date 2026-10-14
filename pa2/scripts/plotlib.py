"""Helpers for make_report.py: load the CSV, derive metrics, draw charts and
emit HTML tables."""
import csv
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TRACES = ["trace1", "trace2", "trace3"]
TRACE_LABELS = {"trace1": "Trace1", "trace2": "Trace2", "trace3": "Trace3"}

# categorical slots in fixed order (validated reference palette, light mode)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.edgecolor": INK_2,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 200,
})

METRIC_FMT = {
    "ipc": "{:.4f}", "speedup": "{:.4f}", "l1d_load_mpki": "{:.2f}", "l2c_load_mpki": "{:.2f}",
    "l2c_instr_load_mpki": "{:.2f}", "l2c_data_load_mpki": "{:.2f}", "l1d_pf_accuracy": "{:.1f}",
    "coverage": "{:.2f}", "dyn_irrip_fraction": "{:.2f}", "dyn_mean_ratio": "{:.3f}",
}


class Results:
    def __init__(self, path, baseline="base_lru"):
        self.rows = {}
        with open(path) as fh:
            for r in csv.DictReader(fh):
                self.rows[(r["tag"], r["trace"])] = r
        self.baseline = baseline

    def has(self, tag):
        return all((tag, t) in self.rows for t in TRACES)

    def raw(self, tag, trace, key):
        v = self.rows[(tag, trace)][key]
        return float(v) if v not in ("", None) else float("nan")

    def get(self, tag, trace, metric, baseline=None):
        base = baseline or self.baseline
        if metric == "speedup":
            return self.raw(tag, trace, "ipc") / self.raw(base, trace, "ipc")
        if metric == "coverage":
            mb = self.raw(base, trace, "l1d_load_miss")
            return 100.0 * (mb - self.raw(tag, trace, "l1d_load_miss")) / mb
        return self.raw(tag, trace, metric)

    def mean(self, tag, metric, baseline=None):
        vals = [self.get(tag, t, metric, baseline) for t in TRACES]
        if metric in ("speedup", "ipc"):
            return math.exp(sum(math.log(v) for v in vals) / len(vals))
        return sum(vals) / len(vals)


def _finish(fig, ax, path, n_series, legend=True, legend_cols=None):
    ax.yaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    if legend and n_series > 1:
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=legend_cols or min(n_series, 4),
                  frameon=False, fontsize=7.5, handlelength=1.1, columnspacing=1.0, borderaxespad=0.3)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def bar_chart(path, res, tags, labels, metric, ylabel, title=None, ref_line=None, with_mean=False,
              ylim=None, baseline=None, legend_cols=None, size=(3.4, 2.35)):
    """Grouped bars: one group per trace (+ mean), one bar per configuration."""
    groups = list(TRACES) + (["mean"] if with_mean else [])
    n = len(tags)
    fig, ax = plt.subplots(figsize=size)
    bw = 0.82 / n
    allv = []
    for i, (tag, lab) in enumerate(zip(tags, labels)):
        vals = [res.mean(tag, metric, baseline) if g == "mean" else res.get(tag, g, metric, baseline) for g in groups]
        allv += vals
        xs = [j + (i - (n - 1) / 2) * bw for j in range(len(groups))]
        ax.bar(xs, vals, width=bw, color=SERIES[i % len(SERIES)], edgecolor="white", linewidth=0.8,
               label=lab, zorder=3)
    if ref_line is not None:
        ax.axhline(ref_line, color=INK_2, linewidth=0.8, linestyle="--", zorder=4)
    ax.set_xticks(range(len(groups)))
    mean_name = "Geomean" if metric in ("speedup", "ipc") else "Mean"
    ax.set_xticklabels([TRACE_LABELS.get(g, mean_name) for g in groups])
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    elif metric == "speedup":
        lo = min(allv + [1.0]); hi = max(allv + [1.0]); span = max(hi - lo, 0.004)
        ax.set_ylim(lo - 0.12 * span, hi + 0.12 * span)
    if title:
        ax.set_title(title, fontsize=9, color=INK, loc="left")
    _finish(fig, ax, path, n, legend_cols=legend_cols)


def hbar_chart(path, values, labels, xlabel, ref=None, ref_label=None, fmt="{:.4f}", size=(6.6, 3.0), highlight=None):
    """One series of horizontal bars (e.g. geomean speedup per variant), value labelled."""
    fig, ax = plt.subplots(figsize=size)
    ys = list(range(len(values)))[::-1]
    colors = [SERIES[1] if (highlight and lab in highlight) else SERIES[0] for lab in labels]
    ax.barh(ys, values, height=0.62, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    lo = min(values + ([ref] if ref else [])); hi = max(values + ([ref] if ref else []))
    span = max(hi - lo, 0.002)
    ax.set_xlim(lo - 0.25 * span, hi + 0.22 * span)
    for y, v in zip(ys, values):
        ax.text(v + 0.01 * span, y, fmt.format(v), va="center", ha="left", fontsize=7, color=INK)
    if ref is not None:
        ax.axvline(ref, color=INK_2, linewidth=0.8, linestyle="--", zorder=4)
        if ref_label:
            ax.text(ref, len(values) - 0.35, " " + ref_label, fontsize=7, color=INK_2, va="bottom")
    ax.set_xlabel(xlabel)
    ax.xaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def line_chart(path, xs, series, labels, xlabel, ylabel, ref_line=None, size=(3.4, 2.35), ylim=None, categorical=False):
    """series: list of y-lists (one per line). categorical=True spaces the x values evenly."""
    fig, ax = plt.subplots(figsize=size)
    xticklabels = ["%g" % x for x in xs]
    if categorical:
        xs = list(range(len(xs)))
    for i, (ys, lab) in enumerate(zip(series, labels)):
        ax.plot(xs, ys, color=SERIES[i % len(SERIES)], linewidth=1.6, marker="o", markersize=3.6,
                markeredgecolor="white", markeredgewidth=0.8, label=lab, zorder=3)
    if ref_line is not None:
        ax.axhline(ref_line, color=INK_2, linewidth=0.8, linestyle="--", zorder=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(xs)
    ax.set_xticklabels(xticklabels)
    if ylim:
        ax.set_ylim(*ylim)
    _finish(fig, ax, path, len(series))


def html_table(res, tags, labels, cols, bold=None, per_trace=True, mean_row=True, baseline=None, extra=None):
    """cols: list of (metric, header, format). One block of rows per trace, plus an average block."""
    out = ['<table class="data">']
    out.append("<thead><tr><th>Configuration</th>" + "".join("<th>%s</th>" % h for _, h, _ in cols) + "</tr></thead><tbody>")
    blocks = (list(TRACES) if per_trace else []) + (["mean"] if mean_row else [])
    for b in blocks:
        name = TRACE_LABELS.get(b, "Average (geometric mean for IPC and speedup, arithmetic mean otherwise)")
        out.append('<tr class="group"><td colspan="%d">%s</td></tr>' % (len(cols) + 1, name))
        for tag, lab in zip(tags, labels):
            cls = ' class="bold"' if bold and tag in bold else ""
            cells = []
            for metric, _, f in cols:
                if metric in ("coverage", "l1d_pf_accuracy") and res.raw(tag, "trace1", "l1d_pf_issued") == 0:
                    cells.append("&ndash;")
                    continue
                try:
                    v = res.mean(tag, metric, baseline) if b == "mean" else res.get(tag, b, metric, baseline)
                    cells.append("&ndash;" if (isinstance(v, float) and math.isnan(v)) else f.format(v))
                except (KeyError, ZeroDivisionError, ValueError):
                    cells.append("&ndash;")
            out.append("<tr%s><td>%s</td>%s</tr>" % (cls, lab, "".join("<td>%s</td>" % c for c in cells)))
    out.append("</tbody></table>")
    return "\n".join(out)


def sweep_table(res, tags, labels, first_header, extra_cols=(), bold=None, baseline=None):
    """Compact one-row-per-variant table: speedup per trace, geomean, and extra mean columns."""
    hdr = "<th>%s</th>" % first_header + "".join("<th>Speedup %s</th>" % TRACE_LABELS[t] for t in TRACES) + "<th>Geomean speedup</th>"
    hdr += "".join("<th>%s</th>" % h for _, h, _ in extra_cols)
    out = ['<table class="data">', "<thead><tr>%s</tr></thead><tbody>" % hdr]
    for tag, lab in zip(tags, labels):
        cls = ' class="bold"' if bold and tag in bold else ""
        cells = ["%.4f" % res.get(tag, t, "speedup", baseline) for t in TRACES]
        cells.append("%.4f" % res.mean(tag, "speedup", baseline))
        for metric, _, f in extra_cols:
            if callable(metric):
                cells.append(metric(tag))
            else:
                cells.append(f.format(res.mean(tag, metric, baseline)))
        out.append("<tr%s><td>%s</td>%s</tr>" % (cls, lab, "".join("<td>%s</td>" % c for c in cells)))
    out.append("</tbody></table>")
    return "\n".join(out)

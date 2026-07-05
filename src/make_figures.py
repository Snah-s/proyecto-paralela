"""
make_figures.py — Genera las figuras vectorizadas del proyecto FINAL.
═══════════════════════════════════════════════════════════════════════════════
Lee src/results_mpi.csv y src/results_omp.csv, agrega por MEDIANA sobre las
repeticiones (robusto al thermal-throttling del laptop) y produce PDFs en
Proyecto_Final_Paralela/figures/.

Hardware de los experimentos: Intel i7-1165G7, 4 núcleos físicos / 8 hilos (HT).
  → la frontera p=4 (núcleos físicos) se marca en cada figura; p=8 está
    OVERSUSCRITO (8 procesos en 4 núcleos) y se sombrea como región HT.
"""
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy.optimize import curve_fit
import os

# Raíz del repo, robusta al cwd (script desde la raíz o notebook desde src/)
try:
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    ROOT = os.getcwd()
    if os.path.basename(ROOT) == "src":
        ROOT = os.path.dirname(ROOT)
FIG_DIR = os.path.join(ROOT, "Proyecto_Final_Paralela/figures")
os.makedirs(FIG_DIR, exist_ok=True)
N_PHYS  = 4          # núcleos físicos → frontera de oversubscription
plt.rcParams.update({
    "font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
    "figure.dpi": 120, "savefig.bbox": "tight", "legend.fontsize": 7.5,
    "axes.titlesize": 9.5,
})
COL = {1797:"#1b9e77", 5000:"#d95f02", 10000:"#7570b3", 15000:"#e7298a", 20000:"#66a61e"}
PS  = [1, 2, 4, 8]

mpi = pd.read_csv(os.path.join(ROOT, "src/results_mpi.csv"))
omp = pd.read_csv(os.path.join(ROOT, "src/results_omp.csv"))
df  = pd.concat([mpi, omp], ignore_index=True)
med = df.groupby(["paradigm", "stage", "n", "p"]).median(numeric_only=True).reset_index()

# TRAIN-parallel (estrategia rival): CSV propio con columna extra t_merge
_train_path = os.path.join(ROOT, "src/results_train.csv")
tr = pd.read_csv(_train_path) if os.path.isfile(_train_path) else None
tmed = tr.groupby(["n", "p"]).median(numeric_only=True).reset_index() if tr is not None else None

def get_train(n, col):
    s = tmed[tmed.n == n].sort_values("p")
    return s.p.values, s[col].values

def get(paradigm, stage, n, col):
    s = med[(med.paradigm==paradigm)&(med.stage==stage)&(med.n==n)].sort_values("p")
    return s.p.values, s[col].values

def mark_hardware(ax):
    ax.axvline(N_PHYS, color="grey", ls=":", lw=1)
    ax.axvspan(N_PHYS, 8, color="grey", alpha=0.08)

def save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path); plt.close(fig)
    print("  ✓", path)

# ── FIG 1: Speedup y Eficiencia (stage buf) ──────────────────────────────────
def fig_speedup():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for n in [1797, 5000, 10000, 15000, 20000]:
        p, t = get("mpi", "buf", n, "t_total")
        S = t[0] / t;  E = S / p
        a1.plot(p, S, "o-", color=COL[n], ms=4, label=f"n={n}")
        a2.plot(p, E, "o-", color=COL[n], ms=4, label=f"n={n}")
    a1.plot(PS, PS, "k--", lw=1, label="ideal S=p")
    for ax in (a1, a2): mark_hardware(ax); ax.set_xlabel("procesos p"); ax.set_xticks(PS)
    a1.set_ylabel("Speedup S(p)"); a1.set_title("Speedup (MPI, etapa buf)")
    a2.set_ylabel("Eficiencia E(p)"); a2.set_title("Eficiencia"); a2.axhline(1, color="k", ls="--", lw=0.8)
    a1.legend(); fig.tight_layout(); save(fig, "fig_speedup_eficiencia.pdf")

# ── FIG 2: GFLOP/s vs p ──────────────────────────────────────────────────────
def fig_flops():
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    for n in [1797, 10000, 20000]:
        p, f = get("mpi", "buf", n, "flops_per_sec")
        ax.plot(p, f/1e9, "o-", color=COL[n], ms=4, label=f"n={n}")
    p1, f1 = get("mpi", "buf", 20000, "flops_per_sec")
    ax.plot(PS, f1[0]/1e9*np.array(PS), "k--", lw=1, label="ideal ∝p")
    mark_hardware(ax); ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(PS); ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_xlabel("procesos p"); ax.set_ylabel("GFLOP/s"); ax.set_title("Rendimiento FLOP/s (MPI buf)")
    ax.legend(); fig.tight_layout(); save(fig, "fig_flops_vs_p.pdf")

# ── FIG 3: Desglose de comunicación (log p + oversubscription) ───────────────
def fig_comm():
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for col, lab, mk in [("t_bcast","bcast (X_train)","o"),
                         ("t_scatter","scatter (X_test)","s"),
                         ("t_gather","gather (ŷ)","^")]:
        p, t = get("mpi", "buf", 20000, col)
        ax.plot(p, t*1e3, mk+"-", ms=4, label=lab)
    mark_hardware(ax); ax.set_yscale("log"); ax.set_xticks(PS)
    ax.set_xlabel("procesos p"); ax.set_ylabel("tiempo (ms)")
    ax.set_title("Comunicación colectiva vs p (n=20000)")
    ax.annotate("p=8 oversuscrito\n(8 proc / 4 núcleos)", xy=(8, ax.get_ylim()[1]*0.3),
                xytext=(4.3, ax.get_ylim()[1]*0.5), fontsize=6.5,
                arrowprops=dict(arrowstyle="->", color="grey"))
    ax.legend(); fig.tight_layout(); save(fig, "fig_comm_breakdown.pdf")

# ── FIG 4: Punto óptimo p* ───────────────────────────────────────────────────
def fig_popt():
    p, t = get("mpi", "buf", 20000, "t_total")
    (a, b), _ = curve_fit(lambda p, a, b: a/p + b*p, p.astype(float), t, p0=[t[0], 0.1])
    pstar = np.sqrt(a/b)
    pp = np.linspace(1, 8, 200)
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.plot(p, t, "o", color="#66a61e", ms=6, label="medido (buf, n=20000)")
    ax.plot(pp, a/pp + b*pp, "-", color="#1b9e77", label=f"T=a/p+b·p\na={a:.1f}s, b={b:.3f}s")
    ax.plot(pp, a/pp, "--", color="grey", lw=1, label="a/p (cómputo)")
    ax.plot(pp, b*pp, ":", color="grey", lw=1, label="b·p (overhead)")
    ax.axvline(pstar, color="red", ls="-.", lw=1.2)
    ax.annotate(f"p*={pstar:.1f}", xy=(pstar, a/pstar+b*pstar), xytext=(pstar+0.3, t.max()*0.6),
                color="red", fontsize=8)
    mark_hardware(ax); ax.set_xticks(PS); ax.set_xlabel("procesos p"); ax.set_ylabel("T_total (s)")
    ax.set_title("Número óptimo de procesos"); ax.legend(fontsize=6.5)
    fig.tight_layout(); save(fig, "fig_popt.pdf")
    return a, b, pstar

# ── FIG 5: Escalabilidad — Brent (√p) vs Isoeficiencia (p·log p) ─────────────
def fig_escalabilidad():
    p = np.array([1, 2, 4, 8], dtype=float)
    brent  = np.sqrt(p)                       # sin comunicación (cota de Brent)
    isoeff = p * (1 + np.log2(p))             # con comunicación log p (normalizado, p=1→1)
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.plot(p, brent,  "s--", color="#1b9e77", label=r"Brent (sin comm): $n\propto\sqrt{p}$")
    ax.plot(p, isoeff, "o-",  color="#d95f02", label=r"Isoeficiencia (comm $\log p$): $n\propto p\log p$")
    ax.plot(p, p, "k:", lw=0.8, label=r"referencia $n\propto p$")
    mark_hardware(ax); ax.set_xticks(PS); ax.set_yscale("log")
    ax.set_xlabel("procesos p"); ax.set_ylabel(r"$n$ requerido / $n_0$")
    ax.set_title("Escalabilidad débil: crecimiento de n(p)\npara mantener E constante")
    ax.legend(fontsize=6.5); fig.tight_layout(); save(fig, "fig_escalabilidad.pdf")

# ── FIG 6: Desarrollo en 3 etapas (loop/vec/buf) ─────────────────────────────
def fig_stages():
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ns = [1797, 5000, 10000, 15000, 20000]
    for stage, mk, c in [("loop","o","#1b9e77"), ("vec","s","#d95f02"), ("buf","^","#7570b3")]:
        y = [med[(med.paradigm=="mpi")&(med.stage==stage)&(med.n==n)&(med.p==4)].t_total.values[0] for n in ns]
        ax.plot(ns, y, mk+"-", color=c, ms=4, label=f"etapa {stage}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("n (muestras)"); ax.set_ylabel("T_total (s), p=4")
    ax.set_title("Desarrollo incremental (3 etapas)"); ax.legend()
    fig.tight_layout(); save(fig, "fig_stages.pdf")

# ── FIG 7: MPI vs OMP (headline de paradigmas) ───────────────────────────────
def fig_mpi_vs_omp():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for n, c in [(1797,"#1b9e77"), (20000,"#66a61e")]:
        p, tm = get("mpi", "buf", n, "t_compute")
        _, to = get("omp", "numba", n, "t_compute")
        a1.plot(p, tm, "s--", color=c, ms=4, label=f"MPI buf n={n}")
        a1.plot(p, to, "o-",  color=c, ms=4, label=f"OMP n={n}")
        pf, fm = get("mpi", "buf", n, "flops_per_sec")
        _,  fo = get("omp", "numba", n, "flops_per_sec")
        a2.plot(pf, fm/1e9, "s--", color=c, ms=4, label=f"MPI buf n={n}")
        a2.plot(pf, fo/1e9, "o-",  color=c, ms=4, label=f"OMP n={n}")
    for ax in (a1, a2): mark_hardware(ax); ax.set_xticks(PS); ax.set_yscale("log"); ax.set_xlabel("p / hilos")
    a1.set_ylabel("T_compute (s)"); a1.set_title("Cómputo: MPI vs OMP")
    a2.set_ylabel("GFLOP/s"); a2.set_title("Rendimiento: MPI vs OMP")
    a1.legend(fontsize=6.5); fig.tight_layout(); save(fig, "fig_mpi_vs_omp.pdf")

# ── FIG 8: TEST-parallel vs TRAIN-parallel (prueba empírica) ─────────────────
def fig_test_vs_train():
    if tmed is None:
        print("  (results_train.csv ausente: se omite fig_test_vs_train)"); return None
    n = 20000
    p_te, comm_te = get("mpi", "buf", n, "t_comm")          # TEST: solo comm
    _,    tot_te  = get("mpi", "buf", n, "t_total")
    p_tr, comm_tr = get_train(n, "t_comm")
    _,    merg_tr = get_train(n, "t_merge")
    _,    tot_tr  = get_train(n, "t_total")
    over_te = comm_te                                        # TEST no fusiona
    over_tr = comm_tr + merg_tr                              # TRAIN: comm + fusión

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 3.0))
    # (a) overhead absoluto (ms), escala log
    a1.plot(p_te, over_te*1e3, "o-",  color="#1b9e77", ms=5, label="TEST (paraleliza test)")
    a1.plot(p_tr, over_tr*1e3, "s--", color="#d95f02", ms=5, label="TRAIN (paraleliza train)")
    a1.set_yscale("log"); a1.set_ylabel("overhead comm+fusión (ms)")
    a1.set_title("Overhead de la estrategia (n=20000)")
    # (b) overhead como % del tiempo total
    a2.plot(p_te, 100*over_te/tot_te, "o-",  color="#1b9e77", ms=5, label="TEST")
    a2.plot(p_tr, 100*over_tr/tot_tr, "s--", color="#d95f02", ms=5, label="TRAIN")
    a2.set_ylabel("overhead / T_total (%)"); a2.set_title("Peso del overhead")
    for ax in (a1, a2):
        mark_hardware(ax); ax.set_xticks(PS); ax.set_xlabel("procesos p")
    a1.legend(fontsize=6.5); fig.tight_layout(); save(fig, "fig_test_vs_train.pdf")
    return p_tr, over_te, over_tr, tot_te, tot_tr

if __name__ == "__main__":
    print("Generando figuras en", FIG_DIR)
    fig_speedup(); fig_flops(); fig_comm(); a,b,ps = fig_popt()
    fig_escalabilidad(); fig_stages(); fig_mpi_vs_omp(); tt = fig_test_vs_train()

    # ── Resumen numérico para las tablas del informe ─────────────────────────
    print("\n=== TABLA speedup/eficiencia (buf) ===")
    for n in [1797, 20000]:
        p, t = get("mpi","buf",n,"t_total"); _, tc = get("mpi","buf",n,"t_compute"); _, tk = get("mpi","buf",n,"t_comm")
        print(f"n={n}")
        for i,pp in enumerate(p):
            S=t[0]/t[i]; print(f"  p={int(pp)}: t_total={t[i]*1e3:8.1f}ms t_comm={tk[i]*1e3:7.2f}ms S={S:.2f} E={S/pp:.3f}")
    print(f"\np* (buf,n=20000): a={a:.2f}s b={b:.4f}s  p*={ps:.2f}")
    print("\n=== OMP vs MPI compute @p=4 ===")
    for n in [1797,5000,10000,15000,20000]:
        tm=med[(med.paradigm=='mpi')&(med.stage=='buf')&(med.n==n)&(med.p==4)].t_compute.values[0]
        to=med[(med.paradigm=='omp')&(med.n==n)&(med.p==4)].t_compute.values[0]
        print(f"  n={n:5d}: MPI/buf={tm:.3f}s OMP={to:.4f}s  ({tm/to:.1f}x)")
    print("\n=== accuracy(n) ===")
    print(med.groupby('n').accuracy.agg(['min','max']).to_string())
    print("\n=== GFLOP/s peak ===")
    print(f"  MPI buf n=20000 p=8: {get('mpi','buf',20000,'flops_per_sec')[1][-1]/1e9:.2f}")
    print(f"  OMP     n=20000 t=8: {get('omp','numba',20000,'flops_per_sec')[1][-1]/1e9:.2f}")

    if tmed is not None:
        print("\n=== TEST-parallel vs TRAIN-parallel (n=20000) ===")
        p_te, c_te = get("mpi","buf",20000,"t_comm");  _, t_te = get("mpi","buf",20000,"t_total")
        p_tr, c_tr = get_train(20000,"t_comm");        _, m_tr = get_train(20000,"t_merge")
        _,   t_tr  = get_train(20000,"t_total")
        print("  p | TEST over(ms) | TRAIN over(ms) | ratio | TEST S | TRAIN S")
        for i,pp in enumerate(p_te):
            ov_te = c_te[i]; ov_tr = c_tr[i]+m_tr[i]
            print(f"  {int(pp)} | {ov_te*1e3:11.2f} | {ov_tr*1e3:12.2f} | {ov_tr/ov_te:5.1f}x |"
                  f" {t_te[0]/t_te[i]:5.2f} | {t_tr[0]/t_tr[i]:5.2f}")

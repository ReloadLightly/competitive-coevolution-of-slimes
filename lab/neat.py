"""
lab/neat.py — NEAT (Stanley & Miikkulainen 2002) in the study's self-play
setting (WP9, part 2).

The fixed-topology families evolve the 273 weights of a 12-10-10-3 network.
NEAT evolves the network itself: genomes start minimal (every input wired to
every output) and grow hidden nodes and connections, which crossover aligns
by innovation number and speciation protects while they are tuned.
Recurrent connections are allowed (Ha's 2015 baseline is recurrent).

Comparability with the existing families (docs/paper/04-appendix.md, A.8):

  * the game and evaluation are the study's: the compiled physics of
    fastvolley, the 2015 baseline as external yardstick, the same sweep and
    held-out seeds;
  * the self-play evaluation is the generational GA's (algorithms.run_ga2015):
    100 genomes, 500 games per generation between random pairs, fitness the
    mean point margin, 1,000 generations = 500,000 games, the generation's
    fittest genome exported every 10 generations (5,000 games);
  * mutation perturbs weights with the study's sigma = 0.1 and initialises
    them with its scale 0.5; everything else is NEAT's published defaults.

Validation (test_repo.py): a NEAT genome that encodes a 12-10-10-3 network
computes that network's outputs bit for bit, scores identically against the
2015 baseline (same serves), and plays identical games step by step. So the
NEAT path is the study's game, not a reimplementation of it.

Reproduction (speciation, crossover, mutation) is plain Python on dict
genomes; the forward pass and the games are compiled.
"""

import heapq

import numpy as np
from numba import njit

from fastvolley import (BVX, BVY, BX, BY, LIFE_L, LIFE_R, LX, LY, OBS_SIZE, RX,
                        STATE_SIZE, T_LIMIT, _game_step, _reset_game,
                        baseline_forward)

N_IN, N_OUT = 12, 3
OUT_IDS = (12, 13, 14)          # node ids of the outputs; hidden ids start at 15
FIRST_HIDDEN = 15
MAX_VALS = 512                  # value slots per network during a game

KIND_NEAT, KIND_BASELINE = 0, 1

# NEAT's published defaults (Stanley & Miikkulainen 2002), except the weight
# scale, which follows the study (sigma 0.1, initial scale 0.5), and a
# compatibility threshold that adapts towards `target_species` (a common NEAT
# practice): with weights moved in steps of 0.1 the fixed threshold of 3.0
# was never reached and every genome stayed in one species (pilot, seed 7)
PARAMS = dict(
    pop=100, n_opponents=10, sigma=0.10, init_scale=0.5,
    p_weight_mutate=0.8, p_weight_perturb=0.9, p_add_node=0.03,
    p_add_conn=0.05, p_crossover=0.75, p_interspecies=0.001,
    p_disabled_inherit=0.75, c1=1.0, c2=1.0, c3=0.4, threshold=3.0,
    target_species=8, threshold_step=0.05,
    survival=0.2, elite_min_species=5, stagnation=15, keep_species=2)


# --------------------------------------------------------------------------
# genomes (Python)
# --------------------------------------------------------------------------
class Innovations:
    """Innovation numbers and split-node ids, shared by a run."""

    def __init__(self):
        self.conn = {}            # (src, dst) -> innovation number
        self.split = {}           # innovation of a split connection -> node id
        self.next_node = FIRST_HIDDEN

    def of(self, src, dst):
        if (src, dst) not in self.conn:
            self.conn[(src, dst)] = len(self.conn)
        return self.conn[(src, dst)]

    def split_node(self, innov):
        if innov not in self.split:
            self.split[innov] = self.next_node
            self.next_node += 1
        return self.split[innov]


def new_genome(rng, inno, scale):
    """Minimal genome: every input to every output, random weights and biases."""
    g = {"nodes": {o: rng.normal() * scale for o in OUT_IDS}, "conns": {}}
    for i in range(N_IN):
        for o in OUT_IDS:
            g["conns"][inno.of(i, o)] = [i, o, rng.normal() * scale, True, False]
    return g


def copy_genome(g):
    return {"nodes": dict(g["nodes"]),
            "conns": {k: list(v) for k, v in g["conns"].items()}}


def from_mlp(p, inno):
    """A NEAT genome computing the study's 12-10-10-3 network `p` exactly."""
    h1 = list(range(FIRST_HIDDEN, FIRST_HIDDEN + 10))
    h2 = list(range(FIRST_HIDDEN + 10, FIRST_HIDDEN + 20))
    inno.next_node = max(inno.next_node, FIRST_HIDDEN + 20)
    g = {"nodes": {}, "conns": {}}
    for j, n in enumerate(h1):
        g["nodes"][n] = p[120 + j]
    for j, n in enumerate(h2):
        g["nodes"][n] = p[230 + j]
    for j, n in enumerate(OUT_IDS):
        g["nodes"][n] = p[270 + j]
    for i in range(12):
        for j, n in enumerate(h1):
            g["conns"][inno.of(i, n)] = [i, n, p[i * 10 + j], True, False]
    for i, m in enumerate(h1):
        for j, n in enumerate(h2):
            g["conns"][inno.of(m, n)] = [m, n, p[130 + i * 10 + j], True, False]
    for i, m in enumerate(h2):
        for j, n in enumerate(OUT_IDS):
            g["conns"][inno.of(m, n)] = [m, n, p[240 + i * 3 + j], True, False]
    return g


def _reaches(g, start, target):
    """Is `target` reachable from `start` along feedforward connections?"""
    stack, seen = [start], {start}
    succ = {}
    for src, dst, _, _, rec in g["conns"].values():
        if not rec:
            succ.setdefault(src, []).append(dst)
    while stack:
        n = stack.pop()
        if n == target:
            return True
        for m in succ.get(n, ()):
            if m not in seen:
                seen.add(m)
                stack.append(m)
    return False


def mutate(g, rng, inno, P):
    if rng.random() < P["p_weight_mutate"]:
        for c in g["conns"].values():
            c[2] = (c[2] + rng.normal() * P["sigma"] if rng.random() < P["p_weight_perturb"]
                    else rng.normal() * P["init_scale"])
        for n in g["nodes"]:
            g["nodes"][n] = (g["nodes"][n] + rng.normal() * P["sigma"]
                             if rng.random() < P["p_weight_perturb"]
                             else rng.normal() * P["init_scale"])
    if rng.random() < P["p_add_node"]:
        enabled = [k for k, c in g["conns"].items() if c[3]]
        if enabled:
            k = enabled[rng.integers(len(enabled))]
            src, dst, w, _, rec = g["conns"][k]
            g["conns"][k][3] = False
            n = inno.split_node(k)
            if n not in g["nodes"]:
                g["nodes"][n] = 0.0
                g["conns"][inno.of(src, n)] = [src, n, 1.0, True, False]
                g["conns"][inno.of(n, dst)] = [n, dst, w, True, rec]
    if rng.random() < P["p_add_conn"]:
        sources = list(range(N_IN)) + sorted(g["nodes"])
        targets = sorted(g["nodes"])
        for _ in range(20):          # a few tries to find a new pair
            src = sources[rng.integers(len(sources))]
            dst = targets[rng.integers(len(targets))]
            k = inno.of(src, dst)
            if k in g["conns"]:
                continue
            rec = src == dst or _reaches(g, dst, src)
            g["conns"][k] = [src, dst, rng.normal() * P["init_scale"], True, rec]
            break
    return g


def crossover(fit, other, rng, P):
    """Child of the fitter parent `fit` and `other`, aligned on innovation."""
    child = {"nodes": {}, "conns": {}}
    for k, c in fit["conns"].items():
        if k in other["conns"]:
            o = other["conns"][k]
            gene = list(c if rng.random() < 0.5 else o)
            if not (c[3] and o[3]):
                gene[3] = rng.random() >= P["p_disabled_inherit"]
            # whether a connection closes a loop depends on the network it
            # sits in; the child has the fitter parent's structure, so it
            # takes that parent's flag (the other parent's could close a
            # feedforward cycle here)
            gene[4] = c[4]
        else:
            gene = list(c)
        child["conns"][k] = gene
    for n, bias in fit["nodes"].items():
        child["nodes"][n] = (other["nodes"][n] if n in other["nodes"]
                             and rng.random() < 0.5 else bias)
    return child


def distance(a, b, P):
    ka, kb = set(a["conns"]), set(b["conns"])
    match = ka & kb
    cut = min(max(ka), max(kb))
    nonmatch = (ka | kb) - match
    excess = sum(1 for k in nonmatch if k > cut)
    disjoint = len(nonmatch) - excess
    n = max(len(ka), len(kb))
    n = 1 if n < 20 else n
    w = (np.mean([abs(a["conns"][k][2] - b["conns"][k][2]) for k in match])
         if match else 0.0)
    return P["c1"] * excess / n + P["c2"] * disjoint / n + P["c3"] * w


# --------------------------------------------------------------------------
# phenotypes: a genome compiled to flat arrays
# --------------------------------------------------------------------------
def compile_genome(g):
    """(pos, bias, start, end, src, w, rec, out) for the compiled forward pass.

    Inputs hold value slots 0..11. The other nodes are evaluated in a
    topological order of the enabled feedforward connections (ties by node
    id) and take slots 12.. in that order. A node's incoming connections are
    summed in the order of their sources' slots, then its bias is added and
    tanh applied -- the order mlp_forward uses, which is what makes an
    encoded 12-10-10-3 network bit-identical. Recurrent connections read the
    source's value from the previous step.
    """
    nodes = sorted(g["nodes"])
    conns = [c for c in g["conns"].values() if c[3]]
    indeg = {n: 0 for n in nodes}
    succ = {n: [] for n in list(range(N_IN)) + nodes}
    for src, dst, _, _, rec in conns:
        if not rec and src >= N_IN:
            indeg[dst] += 1
            succ[src].append(dst)
    ready = [n for n in nodes if indeg[n] == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        n = heapq.heappop(ready)
        order.append(n)
        for m in succ[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                heapq.heappush(ready, m)
    if len(order) != len(nodes):
        raise ValueError("feedforward connections contain a cycle")
    slot = {i: i for i in range(N_IN)}
    for k, n in enumerate(order):
        slot[n] = N_IN + k
    if N_IN + len(order) > MAX_VALS:
        raise ValueError("network too large for MAX_VALS")
    incoming = {n: [] for n in nodes}
    for src, dst, w, _, rec in conns:
        incoming[dst].append((slot[src], w, rec))
    pos, bias, start, end, src, wts, rec = [], [], [], [], [], [], []
    for n in order:
        pos.append(slot[n])
        bias.append(g["nodes"][n])
        start.append(len(src))
        for s, w, r in sorted(incoming[n], key=lambda t: t[0]):
            src.append(s)
            wts.append(w)
            rec.append(r)
        end.append(len(src))
    return (np.array(pos, np.int64), np.array(bias, np.float64),
            np.array(start, np.int64), np.array(end, np.int64),
            np.array(src, np.int64), np.array(wts, np.float64),
            np.array(rec, np.bool_), np.array([slot[o] for o in OUT_IDS], np.int64))


def pack(phenotypes):
    """Concatenate phenotypes; per-network offsets into the node/conn arrays."""
    n_off, c_off = [0], [0]
    for ph in phenotypes:
        n_off.append(n_off[-1] + len(ph[0]))
        c_off.append(c_off[-1] + len(ph[4]))
    cat = [np.concatenate([ph[i] for ph in phenotypes]) for i in range(7)]
    outs = np.array([ph[7] for ph in phenotypes], np.int64)
    return (*cat, outs, np.array(n_off, np.int64), np.array(c_off, np.int64))


# --------------------------------------------------------------------------
# compiled forward pass and games
# --------------------------------------------------------------------------
@njit(cache=True, inline="always")
def forward(P, g, obs, vals, prev, out):
    """One step of network g of pack P on `obs`; prev holds last step's values."""
    pos, bias, start, end, src, w, rec, outs, n_off, c_off = P
    for i in range(12):
        vals[i] = obs[i]
    c0 = c_off[g]
    for k in range(n_off[g], n_off[g + 1]):
        s = 0.0
        for c in range(c0 + start[k], c0 + end[k]):
            v = prev[src[c]] if rec[c] else vals[src[c]]
            s += v * w[c]
        vals[pos[k]] = np.tanh(s + bias[k])
    for o in range(3):
        out[o] = vals[outs[g, o]]
    for k in range(n_off[g], n_off[g + 1]):
        prev[pos[k]] = vals[pos[k]]


@njit(cache=True, inline="always")
def _act(kind, P, g, w, b, rnn, obs, vals, prev, out):
    if kind == KIND_NEAT:
        forward(P, g, obs, vals, prev, out)
    else:
        baseline_forward(w, b, rnn, obs, out)


@njit(cache=True)
def play(P, g_r, kind_r, g_l, kind_l, w, b, rnn_r, rnn_l, bvx_buf, bvy_buf,
         use_buf, trace):
    """fastvolley.play_game with NEAT networks: same serves, same physics.

    Returns (score from the right player's view, length). With use_buf the
    serves come from the buffers and the state is written to `trace`."""
    s = np.zeros(STATE_SIZE)
    obs_l = np.zeros(OBS_SIZE)
    obs_r = np.zeros(OBS_SIZE)
    a_l = np.zeros(3)
    a_r = np.zeros(3)
    vr = np.zeros(MAX_VALS)
    pr = np.zeros(MAX_VALS)
    vl = np.zeros(MAX_VALS)
    pl = np.zeros(MAX_VALS)
    k = 0
    if use_buf:
        bvx = bvx_buf[k]
        bvy = bvy_buf[k]
    else:
        bvx = np.random.uniform(-20.0, 20.0)
        bvy = np.random.uniform(10.0, 25.0)
    k += 1
    _reset_game(s, obs_l, obs_r, bvx, bvy)
    obs_l_view = obs_r.copy()
    total = 0
    t = 0
    while True:
        _act(kind_r, P, g_r, w, b, rnn_r, obs_r, vr, pr, a_r)
        _act(kind_l, P, g_l, w, b, rnn_l, obs_l_view, vl, pl, a_l)
        if use_buf:
            nbvx = bvx_buf[k]
            nbvy = bvy_buf[k]
        else:
            nbvx = 0.0
            nbvy = 0.0
        result = _game_step(s, obs_l, obs_r, a_l, a_r, nbvx, nbvy)
        if result != 0:
            if not use_buf:
                s[BVX] = np.random.uniform(-20.0, 20.0)
                s[BVY] = np.random.uniform(10.0, 25.0)
            k += 1
        for i in range(OBS_SIZE):
            obs_l_view[i] = obs_l[i]
        if use_buf:
            trace[t, 0] = s[BX]
            trace[t, 1] = s[BY]
            trace[t, 2] = s[BVX]
            trace[t, 3] = s[BVY]
            trace[t, 4] = s[LX]
            trace[t, 5] = s[LY]
            trace[t, 6] = s[RX]
        total += result
        t += 1
        if t >= T_LIMIT:
            break
        if s[LIFE_L] <= 0.0 or s[LIFE_R] <= 0.0:
            break
    return total, t


@njit(cache=True)
def eval_vs_baseline(P, g, episodes, seed, w, b):
    """fastvolley.eval_vs_baseline for network g: same seed, same serves."""
    np.random.seed(seed)
    scores = np.empty(episodes, dtype=np.int64)
    lengths = np.empty(episodes, dtype=np.int64)
    rnn_r = np.zeros(7)
    rnn_l = np.zeros(7)
    empty = np.zeros(1)
    tr = np.zeros((1, 7))
    for e in range(episodes):
        sc, ln = play(P, g, KIND_NEAT, 0, KIND_BASELINE, w, b, rnn_r, rnn_l,
                      empty, empty, False, tr)
        scores[e] = sc
        lengths[e] = ln
    return scores, lengths


@njit(cache=True)
def play_pairs(P, pairs, w, b):
    """One game per row (i on the right, j on the left); scores and lengths."""
    n = pairs.shape[0]
    scores = np.empty(n, dtype=np.int64)
    lengths = np.empty(n, dtype=np.int64)
    rnn_r = np.zeros(7)
    rnn_l = np.zeros(7)
    empty = np.zeros(1)
    tr = np.zeros((1, 7))
    for r in range(n):
        sc, ln = play(P, pairs[r, 0], KIND_NEAT, pairs[r, 1], KIND_NEAT, w, b,
                      rnn_r, rnn_l, empty, empty, False, tr)
        scores[r] = sc
        lengths[r] = ln
    return scores, lengths


@njit(cache=True)
def seed_games(seed):
    """Seed the compiled games' random serves."""
    np.random.seed(seed)


# --------------------------------------------------------------------------
# the generational loop
# --------------------------------------------------------------------------
def _speciate(pop, species, rng, P, threshold):
    """Assign genomes to species by distance to last generation's
    representatives; new species for the rest. Returns the updated list."""
    for s in species:
        s["members"] = []
    for i, g in enumerate(pop):
        for s in species:
            if distance(g, s["rep"], P) < threshold:
                s["members"].append(i)
                break
        else:
            species.append({"rep": g, "members": [i], "best": -np.inf, "since": 0})
    species = [s for s in species if s["members"]]
    for s in species:
        s["rep"] = pop[s["members"][rng.integers(len(s["members"]))]]
    return species


def _reproduce(pop, fit, species, rng, inno, P):
    n = len(pop)
    for s in species:
        best = max(fit[i] for i in s["members"])
        if best > s["best"]:
            s["best"], s["since"] = best, 0
        else:
            s["since"] += 1
    ranked = sorted(range(len(species)), key=lambda k: -species[k]["best"])
    alive = [k for k in range(len(species))
             if species[k]["since"] < P["stagnation"] or k in ranked[:P["keep_species"]]]
    # fitness sharing: a species' share is its members' mean fitness (shifted
    # to be positive), so large species do not crowd out small ones
    lo = min(fit)
    adj = {k: np.mean([fit[i] for i in species[k]["members"]]) - lo + 1e-6 for k in alive}
    total = sum(adj.values())
    quota = {k: int(np.floor(n * adj[k] / total)) for k in alive}
    rest = sorted(alive, key=lambda k: -(n * adj[k] / total - quota[k]))
    for k in rest[: n - sum(quota.values())]:
        quota[k] += 1
    new = []
    all_members = [i for k in alive for i in species[k]["members"]]
    for k in alive:
        members = sorted(species[k]["members"], key=lambda i: -fit[i])
        q = quota[k]
        if q == 0:
            continue
        if len(members) > P["elite_min_species"]:
            new.append(copy_genome(pop[members[0]]))
            q -= 1
        parents = members[:max(1, int(np.ceil(P["survival"] * len(members))))]
        for _ in range(q):
            a = parents[rng.integers(len(parents))]
            if rng.random() < P["p_crossover"]:
                if rng.random() < P["p_interspecies"]:
                    b_ = all_members[rng.integers(len(all_members))]
                else:
                    b_ = parents[rng.integers(len(parents))]
                fa, fb = (a, b_) if fit[a] >= fit[b_] else (b_, a)
                child = crossover(pop[fa], pop[fb], rng, P)
            else:
                child = copy_genome(pop[a])
            new.append(mutate(child, rng, inno, P))
    return new, [species[k] for k in alive]


def genome_arrays(g):
    """A genome as two arrays: nodes (id, bias), connections
    (innovation, src, dst, weight, enabled, recurrent)."""
    nodes = np.array([[n, b] for n, b in sorted(g["nodes"].items())], np.float64)
    conns = np.array([[k, *c] for k, c in sorted(g["conns"].items())], np.float64)
    return nodes, conns.reshape(-1, 6)


def genome_from_arrays(nodes, conns):
    return {"nodes": {int(n): float(b) for n, b in nodes},
            "conns": {int(r[0]): [int(r[1]), int(r[2]), float(r[3]), bool(r[4]), bool(r[5])]
                      for r in conns}}


def run(seed, n_games, save_every, w, b, P=PARAMS, log=None):
    """NEAT under the generational GA's self-play evaluation.

    Returns a dict: the exported champion of every checkpoint (genomes),
    its fitness, the mean training game length, and the number of species,
    hidden nodes and enabled connections of the champion."""
    rng = np.random.default_rng(seed)
    seed_games(seed)
    inno = Innovations()
    pop = [new_genome(rng, inno, P["init_scale"]) for _ in range(P["pop"])]
    n = P["pop"]
    games_per_gen = n * P["n_opponents"] // 2
    n_gen = n_games // games_per_gen
    gens_per_ckpt = save_every // games_per_gen
    species = []
    threshold = P["threshold"]
    out = {"champs": [], "champ_fit": [], "meanlen": [], "n_species": [],
           "hidden": [], "connections": [], "threshold": []}
    len_acc, len_n = 0.0, 0
    for gen in range(1, n_gen + 1):
        Pk = pack([compile_genome(g) for g in pop])
        i = rng.integers(0, n, size=games_per_gen)
        j = rng.integers(0, n - 1, size=games_per_gen)
        j = j + (j >= i)                      # a distinct opponent, uniformly
        sc, ln = play_pairs(Pk, np.stack([i, j], axis=1).astype(np.int64), w, b)
        fit = np.zeros(n)
        played = np.zeros(n)
        np.add.at(fit, i, sc)
        np.add.at(fit, j, -sc)
        np.add.at(played, i, 1)
        np.add.at(played, j, 1)
        fit = np.where(played > 0, fit / np.maximum(played, 1), 0.0)
        len_acc += float(ln.sum())
        len_n += len(ln)
        species = _speciate(pop, species, rng, P, threshold)
        # steer the number of species towards the target
        if len(species) < P["target_species"]:
            threshold = max(P["threshold_step"], threshold - P["threshold_step"])
        elif len(species) > P["target_species"]:
            threshold += P["threshold_step"]
        if gen % gens_per_ckpt == 0:
            best = int(np.argmax(fit))
            g = pop[best]
            out["champs"].append(copy_genome(g))
            out["champ_fit"].append(float(fit[best]))
            out["meanlen"].append(len_acc / len_n)
            out["n_species"].append(len(species))
            out["hidden"].append(len(g["nodes"]) - N_OUT)
            out["connections"].append(sum(1 for c in g["conns"].values() if c[3]))
            out["threshold"].append(threshold)
            len_acc, len_n = 0.0, 0
            if log:
                log(gen, out)
        pop, species = _reproduce(pop, list(fit), species, rng, inno, P)
    return out

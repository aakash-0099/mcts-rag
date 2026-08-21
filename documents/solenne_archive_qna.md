# QnA — Solenne Archive Network Corpus (for MCTS RAG stress test)

## Question

> Station Kessom's hydrophone buoy was replaced in 2019 using a mooring
> clamp drawn from another station's surplus lot, per the Directorate's
> 2015 logistics memo. Tracing who requested that replacement, who they
> reported to, where that supervisor was previously posted, what
> equipment was calibrated there, which foundry supplied that
> equipment's hardware, and what product line and standard year that
> foundry's clamps came from — what tensile rating (in kilonewtons) was
> stamped on the clamp used in Kessom's May 2019 buoy replacement?

## Gold Answer

**14.2 kN**

## Required Reasoning Path (10 hops)

1. **§12 (Kessom log, May 2019)** — replacement buoy used a clamp from
   "Halvorsen-2015-surplus, Torvek-C."
2. **§12 (Kessom log, March–April 2019)** — the Verrin-9 failure was
   logged by **Technician Oyelaran Dutt**; the replacement was
   authorized by **Dr. Farrah Nkemelu**.
3. **§3 / §7 §5** — Dutt reported to Dr. Nkemelu for instrumentation
   matters at Kessom.
4. **§7 §5** — Before Kessom, Dr. Nkemelu was part of **Station
   Halvorsen's founding technical staff (2009)**, where she calibrated
   the **Ansitrom seismometer array** in 2010.
5. **§5** — The Ansitrom array's mounting/tensioning hardware was
   sourced from the **Draye Foundry**.
6. **§8** — Draye Foundry converted to cold-forged steel in **2009**
   and launched its flagship **Torvek-C** clamp line that same year.
7. **§2 / §8** — 2009 is also **Station Halvorsen's founding year**,
   confirming the Torvek-C line is the one tied to Halvorsen's original
   equipment/surplus stock.
8. **§9 (spec table)** — Torvek-C, standard revision year 2009,
   cold-forged steel → **rated tensile strength: 14.2 kN**.
9. **§10** — Halvorsen's surplus Torvek-C stock (left over after the
   2014 Torvek-C2 transition) was flagged for network redistribution
   in the 2015 inventory audit.
10. **§12** — This confirms the clamp drawn from "Halvorsen-2015-surplus,
    Torvek-C" for Kessom's May 2019 replacement is indeed a 2009-standard
    Torvek-C clamp → **14.2 kN**.

## Why this stresses a graph-based RAG

- No single paragraph, page, or even section contains more than 2 of
  the 10 facts needed.
- The critical entity (Dr. Nkemelu) is introduced in a personnel
  section (§3) with a forward-reference to a *different* personnel
  section (§7 §5), which itself is far from the equipment (§5),
  foundry history (§8), spec table (§9), and maintenance log (§12)
  sections.
- Several **decoys** are seeded to penalize shallow/greedy retrieval:
  - Dr. Renne Achterberg founded *two* stations (Kessom and Oskelund),
    inviting a wrong turn if a shallow retriever conflates "Kessom's
    founding director" with "Kessom's current instrumentation
    supervisor" (Nkemelu ≠ Achterberg).
  - Oskelund's hardware is explicitly *not* Draye-sourced (Ostley
    instead) — a trap for retrievers that jump straight from
    "Nkemelu was at Oskelund" (false; she was never posted there) or
    confuse Oskelund/Halvorsen provenance.
  - Torvek-C2 (14.5 kN, 2014) and Torvek-D (16.0 kN, 2021) are
    near-identical-sounding product lines with different ratings,
    testing whether the retriever locks onto the *correct* standard
    revision year (2009, matching Halvorsen's founding and surplus
    lot) rather than a more "recent-sounding" one.
  - Station Ferrant's October 2019 log entry uses *Torvek-C2* sourced
    *directly* from Draye (not surplus) — a near-miss pattern that
    could mislead a model pattern-matching on "Draye Foundry + clamp"
    without tracking the surplus-lot / year distinction.
  - Vantrelle's hardware is explicitly non-Draye, ruling out a
    plausible-looking dead end if the retriever fixates on Nkemelu's
    *final* posting rather than her *prior* one (the question requires
    prior posting, i.e., Halvorsen).

## Suggested evaluation use

- **Shallow/single-hop baseline** (e.g., dense retrieval + single LLM
  call) should be expected to fail or guess incorrectly, likely
  landing on 14.5 kN (Torvek-C2) or citing Oskelund/Ostley hardware.
- **MCTS RAG** with sufficient `num_simulations` and search depth
  should be able to construct the correct 10-hop chain above; use the
  hop list as a rubric for partial-credit scoring of the reasoning
  trace, not just final-answer accuracy.
- If your graph builder segments by section headers (`##`), you should
  get ~18 nodes, forcing genuine multi-hop traversal (min. 6–8 distinct
  nodes touched) rather than single-node lookup.

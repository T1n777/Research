# 📘 Engineering Chemistry — Complete Short Notes (PES University)

---

# UNIT 1 — Molecular Spectroscopy, Phase Equilibria & Computational Chemistry

---

## 1. EM Radiation & Spectroscopic Basics

**Quick Review:** Molecular spectroscopy = study of interaction of EM radiation with molecules. Pattern of absorption/emission reveals molecular structure and energy levels.

**Key Quantities & Formulas:**
- E = hν = hcν̃ = hc/λ
- ν = c/λ
- ν̃ = 1/λ = ν/c (units: cm⁻¹)
- ν̃(cm⁻¹) = 10⁷ / λ(nm)
- 1 cm⁻¹ = 1.986 × 10⁻²³ J
- c = 3 × 10⁸ m/s = 3 × 10¹⁰ cm/s; h = 6.626 × 10⁻³⁴ J·s

**EM Spectrum — Regions & Transitions:**

| Region | Wavelength | Technique | Transition |
|--------|-----------|-----------|------------|
| Radiofrequency | 10 m – 1 cm | NMR/ESR | Nuclear/electron spin flip |
| Microwave | 1 cm – 100 µm | Rotational spectroscopy | Rotational levels |
| Infrared | 100 µm – 1 µm | IR spectroscopy | Vibrational levels |
| Visible/UV | 1 µm – 10 nm | UV-Vis | Electronic (valence e⁻) |
| X-ray | 10 nm – 100 pm | X-ray spectroscopy | Inner electron |
| γ-ray | 100 pm – 1 pm | γ-ray spectroscopy | Nuclear rearrangement |

**Absorption vs Emission:** ΔE = E₂ − E₁ = hν
- Absorption: lower → higher level; dark lines on bright background
- Emission: higher → lower level; bright lines on dark background

**Atomic vs Molecular Spectra:** Atomic = sharp lines (only electronic); Molecular = broad bands (electronic + vibrational + rotational sub-levels merge)

⚠️ **Exam trap:** When ν̃ is in cm⁻¹, use c = 3 × 10¹⁰ cm/s to get ν in Hz directly. Using c = 3 × 10⁸ m/s without conversion gives answer 100× too small.

---

## 2. Born–Oppenheimer Approximation

**Quick Review:** Total molecular energy has 4 contributions; Etrans is negligible so it's dropped.

**Formula:**
- Full: E_total = E_trans + E_rot + E_vib + E_elec
- **BO Approximation: E = E_rot + E_vib + E_elec**
- Ordering: **E_rot ≪ E_vib ≪ E_elec**

**Energy Scales:**
- Rotational: 0.1 – 10 cm⁻¹ → Microwave
- Vibrational: 500 – 4000 cm⁻¹ → IR
- Electronic: 10,000 – 50,000 cm⁻¹ → UV-Vis

Key insight: Each electronic level has many vibrational sub-levels → each vibrational sub-level has many rotational sub-levels.

---

## 3. Raman Spectroscopy

**Quick Review:** When monochromatic radiation passes through a transparent medium, most scattering is Rayleigh (same λ); a small fraction is Raman (different λ).

- **Rayleigh:** Scattered λ = incident λ (majority)
- **Stokes lines:** Scattered λ > incident λ (molecule gains energy)
- **Anti-Stokes lines:** Scattered λ < incident λ (molecule loses energy)

**Activity condition:** Raman active = change in **polarisability** during vibration (not dipole moment).

**Key rule:** Homonuclear diatomics (H₂, N₂, O₂, Cl₂) — microwave inactive, IR inactive → **Raman active**.

---

## 4. Microwave (Rotational) Spectroscopy

**Quick Review:** Studies rotational energy levels of molecules. Uses microwave region.

**Activity condition:** Molecule must have a **permanent dipole moment**.
- Active: HCl, CO, HBr, NO
- Inactive: H₂, N₂, O₂, Cl₂, CO₂ (symmetric)

**Formulas:**
- μ = m₁m₂ / (m₁ + m₂) → reduced mass
- I = μr₀² → moment of inertia
- ε_J = B·J(J+1) cm⁻¹ → rotational energy levels (J = 0, 1, 2, …)
- B = h / 8π²Ic (cm⁻¹) → rotational constant
- **Δε = 2B(J+1)** → transition energy; ΔJ = ±1

**Spectrum:** Equally spaced lines, spacing = **2B** cm⁻¹

| J | ε_J | Transition | Energy absorbed |
|---|-----|-----------|----------------|
| 0 | 0 | 0→1 | 2B |
| 1 | 2B | 1→2 | 4B |
| 2 | 6B | 2→3 | 6B |
| 3 | 12B | 3→4 | 8B |

**From spectrum → r₀:** Spacing = 2B → get B → I = h/8π²Bc → r₀ = √(I/μ)

---

## 5. IR (Vibrational) Spectroscopy

**Quick Review:** Studies vibrational transitions. Uses IR region.

**Activity condition:** Vibration must produce a **change in dipole moment**.
- IR Active: HCl, CO, H₂O; CO₂ (asymmetric stretch + bending mode)
- IR Inactive: H₂, N₂, Cl₂; CO₂ symmetric stretch

### Simple Harmonic Oscillator (SHO)
**Formulas:**
- ε_v = ν̃_osc(v + ½) cm⁻¹ ; v = 0, 1, 2, …
- ν̃_osc = (1/2πc)√(k/μ) cm⁻¹
- **Zero Point Energy (ZPE) = ½ν̃_osc** (v = 0, molecule still vibrates at absolute zero)
- k = 4π²ν̃²c²μ → force constant (bond strength)

**Selection rule:** Δv = ±1 only → **only ONE absorption line** at ν̃_osc

**Energy levels:** Equally spaced by ν̃_osc cm⁻¹

### Anharmonic Oscillator (Morse Potential)
- ε_v = ν̃_e(v + ½) − ν̃_e·xₑ(v + ½)² cm⁻¹
- x_e = anharmonicity constant (small, positive)
- Energy levels **not equally spaced** — spacing decreases as v increases
- **Selection rule:** Δv = ±1, ±2, ±3, … (overtones allowed)

| Transition | Name | Energy (cm⁻¹) |
|-----------|------|--------------|
| v = 0→1 | Fundamental | ν̃_e(1 − 2x_e) |
| v = 0→2 | 1st overtone | 2ν̃_e(1 − 3x_e) |
| v = 0→3 | 2nd overtone | 3ν̃_e(1 − 4x_e) |

**Hot bands:** At higher T, v=1→2 etc. appear to the left of fundamental. Energy = ν̃_e(1 − 4x_e).

---

## 6. Electronic Spectroscopy & Franck–Condon Principle

**Quick Review:** Electronic transitions (HOMO→LUMO) fall in UV-Visible region.

**Transition types:**
- σ → σ* : Far UV (<200 nm)
- π → π* : UV (~200 nm)
- n → π* : Near UV/Visible (~300 nm)

**No selection rules for v** during electronic transitions → all Δv allowed → **progression** of lines.

Spectrum labeled (v', v"): v' = excited state, v" = ground state. Transitions from v"=0: (0,0), (1,0), (2,0) …

**Franck–Condon Principle:** Electronic transition occurs so rapidly that **internuclear distance does not change** during transition → always **vertical transitions**.

| Case | Condition | Result |
|------|-----------|--------|
| 1 | r_e" < r_e' | Max intensity at (v',0) where v' > 0; spread progression |
| 2 | r_e" = r_e' | Max intensity at (0,0); drops off rapidly |
| 3 | r_e" ≪ r_e' | Continuum (molecule dissociates; kinetic energy not quantised) |

---

## 7. Phase Equilibria — Definitions

**Quick Review:** Phases, components, and degrees of freedom govern phase equilibria.

**Phase (P):** Homogeneous, physically distinct, mechanically separable part.
- Gases: always P = 1
- Liquids: P = number of immiscible layers
- Solids: each solid = separate phase

**Component (C):** Minimum number of independently varying chemical constituents to express composition of every phase.

| System | C |
|--------|---|
| Pure water | 1 |
| CaCO₃ ⇌ CaO + CO₂ (closed) | 2 (3 species − 1 constraint) |
| NH₄Cl ⇌ NH₃ + HCl (closed) | 1 (fixed 1:1 ratio) |
| Same + extra HCl | 2 |

**Degrees of Freedom (F):** Minimum number of intensive variables (T, P, concentration) needed to fully define the system.

---

## 8. Gibbs Phase Rule

**Statement (Gibbs, 1874):** For a heterogeneous system not influenced by external forces:

**F = C − P + 2** (Gibbs Phase Rule)

**F = C − P + 1** (Condensed/Reduced Phase Rule — constant pressure, no gas phase)

**Derivation summary:** Variables = P(C−1) + 2; Constraints = C(P−1); F = Variables − Constraints = C − P + 2

---

## 9. Water System (1-Component)

**For C = 1: F = 3 − P**

| P (phases) | F | Region on diagram |
|------------|---|-----------------|
| 1 | 2 (bivariant) | Area |
| 2 | 1 (univariant) | Line (curve) |
| 3 | 0 (invariant) | Point |

**Curves:**
- **OC (Fusion/Melting):** Ice ⇌ Water; slope = **negative** (unique to water — melting point decreases with pressure because water contracts on melting, ΔV < 0)
- **OA (Vaporisation):** Water ⇌ Vapour; F = 1, positive slope
- **OB (Sublimation):** Ice ⇌ Vapour; F = 1, positive slope
- **O (Triple point):** T = 0.0098°C, P = 4.58 mmHg; F = 0 (all 3 phases coexist)
- **A (Critical point):** 374°C, 220 atm; liquid–vapour interface vanishes
- **OA' (Metastable):** Supercooled water ⇌ Vapour; vapour pressure > that of ice at same T

---

## 10. Pb–Ag System (2-Component)

**Condensed Phase Rule: F = C − P + 1 = 4 − P** (at constant pressure)

**Eutectic Point E:** 303°C; 97.4% Pb + 2.6% Ag; F = 0 (3 phases: solid Pb + solid Ag + liquid)

| Region/Curve | Phases | F |
|-------------|--------|---|
| Above MEN | Liquid melt only | 2 |
| Curve ME | Solid Pb + Liquid | 1 |
| Curve NE | Solid Ag + Liquid | 1 |
| Below FEG | Solid Pb + Solid Ag | 1 |
| Eutectic point E | Solid Pb + Solid Ag + Liquid | 0 |

**Eutectic mixture:** Lowest freezing point of all compositions; definite composition; sharp melting point.

**Pattinson's Process:** Desilverisation of argentiferous lead.
1. Heat above Pb m.p. → complete melt
2. Cool → solid Pb separates (on curve ME), ladled out
3. Remaining liquid enriched in Ag
4. Continue until eutectic (303°C, 2.6% Ag)
5. Eutectic solid processed for silver recovery

---

## 11. Computational Chemistry

**Quick Review:** Uses computer programs based on theoretical chemistry to calculate/predict molecular properties instead of wet experiments.

**Nobel Prizes:** 1998 (Walter Kohn — DFT; John Pople — computational methods); 2013 (multi-scale computational modelling)

**Methods:**

| Method | Based on | Speed | Accuracy | Best for |
|--------|----------|-------|----------|----------|
| Molecular Mechanics (MM) | Classical (Newton) | Very fast | Low (no electrons) | Large molecule geometry |
| Molecular Dynamics (MD) | Classical (Newton) | Fast | Moderate | Conformational analysis, motion |
| Semi-empirical | Schrödinger (approx.) | Moderate | Moderate | Transition states, excited states |
| Ab initio | Schrödinger (exact) | Slow | High | Small molecules, precise energies |
| DFT | Electron density | Moderate | High | Electronic structure, medium systems |

**Advantages over experiment:** Less cost, safer (no hazardous chemicals), faster.

**Applications:** Bond lengths/angles, HOMO/LUMO, UV-Vis/IR/NMR prediction, drug–protein docking, transition states, material design.

**Note:** Complementary to experiments, not a replacement.

---

## Unit 1 — All Formulas at a Glance

- E = hν = hcν̃ = hc/λ; ν = c/λ; ν̃ = ν/c; ν̃(cm⁻¹) = 10⁷/λ(nm)
- μ = m₁m₂/(m₁+m₂); I = μr₀²
- ε_J = BJ(J+1); B = h/8π²Ic; Δε = 2B(J+1); ΔJ = ±1
- ε_v = ν̃_osc(v+½); ν̃_osc = (1/2πc)√(k/μ); ZPE = ½ν̃_osc; Δv = ±1 (SHO)
- ε_v = ν̃_e(v+½) − ν̃_exₑ(v+½)²; Fundamental = ν̃_e(1−2xₑ); 1st overtone = 2ν̃_e(1−3xₑ)
- F = C − P + 2 (Gibbs); F = C − P + 1 (condensed)

---

# UNIT 2 — Electrochemistry & Corrosion

---

## 1. Electrochemical Basics

**Quick Review:** Electrochemistry = interconversion of chemical and electrical energy. Two types of cells: galvanic (chemical → electrical, spontaneous) and electrolytic (electrical → chemical, non-spontaneous).

**Electrode Potential:** When metal M is dipped in its own ion solution Mⁿ⁺, potential at metal–solution interface = electrode potential E.
- Case I: Ionisation faster → metal goes –ve → Helmholtz double layer (–ve metal / +ve ion layer)
- Case II: Deposition faster → metal goes +ve → Helmholtz double layer (+ve metal / –ve anion layer)

**Standard Electrode Potential (E°):** Potential at metal/1M solution interface at 298 K (1 atm for gases).

**Cell Notation (Daniel cell):** Zn(s) / Zn²⁺(1M) // Cu²⁺(1M) / Cu(s)
- Single slash (/) = phase boundary; Double slash (//) = salt bridge

**Formulas:**
- E_cell = E_cathode − E_anode
- ΔG = −nFE_cell; ΔG° = −nFE°_cell
- Spontaneous: ΔG < 0 ↔ E_cell > 0
- F = 96500 C mol⁻¹

**Electrochemical Series (selected E° at 298 K):**

| Half-Reaction | E° (V) |
|--------------|--------|
| Li⁺ + e⁻ → Li | −3.04 |
| Mg²⁺ + 2e⁻ → Mg | −2.37 |
| Al³⁺ + 3e⁻ → Al | −1.66 |
| Zn²⁺ + 2e⁻ → Zn | −0.76 |
| Fe²⁺ + 2e⁻ → Fe | −0.44 |
| Ni²⁺ + 2e⁻ → Ni | −0.25 |
| Pb²⁺ + 2e⁻ → Pb | −0.13 |
| 2H⁺ + 2e⁻ → H₂ | 0.000 (SHE reference) |
| Cu²⁺ + 2e⁻ → Cu | +0.34 |
| Ag⁺ + e⁻ → Ag | +0.80 |
| Au³⁺ + 3e⁻ → Au | +1.52 |
| F₂ + 2e⁻ → 2F⁻ | +2.87 |

Lower E° → more reactive → tends to oxidise (anode). Higher E° → nobler → tends to reduce (cathode).

---

## 2. Nernst Equation

**Quick Review:** Quantitative relationship between electrode potential and ion concentration.

**Derivation sketch:** W_max = nFE; ΔG = −nFE; ΔG = ΔG° + RT ln Q → −nFE = −nFE° + RT ln Q

**Formulas:**
- E = E° − (RT/nF) ln Q [general]
- **E = E° − (0.0591/n) log Q** [at 298 K] (2.303RT/F = 0.0591 V at 298 K)
- For metal–ion electrode: **E = E° + (0.0591/n) log[Mⁿ⁺]**
- For full cell: **E_cell = E°_cell − (0.0591/n) log Q**
- Q = [products] / [reactants]

**Constants:** R = 8.314 J K⁻¹ mol⁻¹; F = 96500 C mol⁻¹; 2.303RT/F at 298 K = **0.0591 V**

---

## 3. Types of Electrodes

| Type | Description | Nernst Equation |
|------|-------------|----------------|
| Metal–Metal Ion | Metal in its own ion solution | E = E° − (0.0591/n) log(1/[Mⁿ⁺]) |
| Metal–Insoluble Salt | Metal/sparingly soluble salt/soluble salt with same anion | E.g., Ag/AgCl: E = E° − 0.0591 log[Cl⁻] |
| Gas Electrode | Inert Pt + gas bubbled around it | H₂: E = E° − (0.0591/2) log(p_H₂/[H⁺]²) |
| Amalgam | Metal dissolved in Hg, in own ion solution | E = E° − (0.0591/2) log([Pb-Hg]/[Pb²⁺]) |
| Redox (Pt) | Inert Pt in solution with both oxidised + reduced forms | Sn⁴⁺/Sn²⁺: E = E° − (0.0591/2) log([Sn²⁺]/[Sn⁴⁺]) |
| Ion-Selective (Membrane) | Membrane selective to one ion | Glass electrode: E_G = E°_G − 0.0591 pH |

**Quinhydrone electrode:** Q + 2H⁺ + 2e⁻ ⇌ QH₂; E = E° − (0.0591/2) log([QH₂]/[Q][H⁺]²)

---

## 4. Reference Electrodes

### Standard Hydrogen Electrode (SHE)
- Notation: Pt/H₂/H⁺; Reaction: 2H⁺ + 2e⁻ ⇌ H₂; **E° = 0.000 V**
- **Disadvantages:** Maintaining exact [H⁺]=1M and p_H₂=1 atm is difficult; Pt gets poisoned by impurities; cannot use in oxidising/reducing environments

### Calomel Electrode
- **Notation:** Hg / Hg₂Cl₂(s) / Cl⁻
- **Construction:** Glass tube → mercury layer → Hg₂Cl₂ paste (calomel + Hg) → KCl solution of known concentration → Pt wire for electrical contact
- **Reactions:**
  - As Anode: 2Hg + 2Cl⁻ ⇌ Hg₂Cl₂ + 2e⁻
  - As Cathode: Hg₂Cl₂ + 2e⁻ ⇌ 2Hg + 2Cl⁻
- **Nernst:** E = E° − 0.0591 log[Cl⁻] (reversible to Cl⁻)

| [KCl] | Name | E at 298 K |
|--------|------|-----------|
| 0.1 M | Decinormal calomel | 0.3358 V |
| 1 M | Normal calomel | 0.2824 V |
| Saturated | SCE | 0.2422 V |

---

## 5. Concentration Cells

**Quick Review:** Identical electrodes; same species but different concentrations. Driving force = concentration gradient.

**Oxidation at anode (lower conc. c₁); reduction at cathode (higher conc. c₂)**

**Electrolyte Concentration Cell:**
- E.g., Cu/Cu²⁺(c₁) // Cu²⁺(c₂)/Cu
- **E_cell = (0.0591/n) log(c₂/c₁)** at 298 K; positive only when c₂ > c₁

**Electrode Concentration Cell (amalgam):**
- E.g., Na–Hg(c₁) / Na⁺ / Na–Hg(c₂)
- E_cell = (2.303RT/nF) log(c₁/c₂)

**Gas Electrode Concentration Cells:**
- H₂ cell: E = (0.0591/2) log(p₁/p₂); positive when p₁ > p₂
- Cl₂ cell: E = (0.0591/2) log(p₂/p₁); positive when p₂ > p₁

---

## 6. Glass Electrode & pH Measurement

**Quick Review:** Ion-Selective Electrode responding specifically to H⁺. Used to measure pH.

**Notation:** Ag/AgCl / 0.1N HCl / Glass membrane / Analyte

**Construction:**
- Thin-walled bulb of **Corning 015 glass** (exchanges H⁺)
- Inside: **0.1M HCl** (known pH = C₂, constant)
- Internal reference: **Ag/AgCl electrode** dipped in internal HCl
- Immersed in **analyte solution** (unknown [H⁺] = C₁)

**Working:** Ion exchange at glass membrane: H⁺(solution) + Na⁺Gl⁻(glass) ⇌ Na⁺(solution) + H⁺Gl⁻(glass). Boundary potential E_b develops ∝ H⁺ concentration difference.

**Formulas:**
- E_b = L′ + 0.0591 log[H⁺] = L′ − 0.0591 pH
- **E_G = E°_G − 0.0591 pH** (total glass electrode potential)
- **E_cell = E°_G − 0.0591 pH − E_SCE** (cell with SCE)
- **pH = (E°_G − E_SCE − E_cell) / 0.0591**
- **E°_G = E_cell(buffer) + 0.0591 × pH(buffer) + E_SCE** (calibration step)

**Cell representation:** Hg/Hg₂Cl₂/Cl⁻ // Analyte / Glass / 0.1N HCl / AgCl / Ag

**Advantages:** Works in oxidising/reducing environments; not poisoned; usable for small volumes; works pH 1–9 (ordinary glass), 1–14 (special glass).

**Disadvantages:** High electrical resistance (needs electronic pH meter); very delicate membrane; **Alkaline Error at pH > 9** — high [Na⁺] causes Na⁺ to replace H⁺ in gel layer → electrode reads Na⁺ as H⁺ → **falsely lower pH reading**.

---

## 7. Corrosion

**Quick Review:** Destruction/deterioration of metal by direct chemical or electrochemical attack. Spontaneous process (ΔG < 0).

**Why metals corrode:** Metals are extracted endothermically → put at high energy state → thermodynamic tendency to revert to their lower-energy ore form.

**Rusting of iron:** Fe₂O₃·xH₂O (reddish-brown)
**Copper patina:** CuCO₃ + Cu(OH)₂ (green — Statue of Liberty)

### Electrochemical Theory of Corrosion
Tiny galvanic cells form on metal surface (due to impurities, grain boundaries, stress):
- **Anode:** M → Mⁿ⁺ + ne⁻ (metal corrodes)
- **Cathode:** Reduction occurs (metal unaffected)

**Cathodic reactions:**

| Condition | H₂ liberation | O₂ absorption |
|-----------|--------------|--------------|
| Acidic | 2H⁺ + 2e⁻ → H₂↑ | 4H⁺ + O₂ + 4e⁻ → 2H₂O |
| Neutral/Alkaline | 2H₂O + 2e⁻ → 2OH⁻ + H₂↑ | 2H₂O + O₂ + 4e⁻ → 4OH⁻ |

**Corrosion of Iron — Full Mechanism:**
1. Anode: Fe → Fe²⁺ + 2e⁻
2. Cathode: 2H₂O + O₂ + 4e⁻ → 4OH⁻
3. 2Fe²⁺ + 4OH⁻ → 2Fe(OH)₂ (white/green)
4. Yellow rust (excess O₂): 4Fe(OH)₂ + O₂ + 2H₂O → 2[Fe₂O₃·3H₂O]
5. Black rust (limited O₂): 3Fe(OH)₂ + ½O₂ → Fe₃O₄·3H₂O

---

## 8. Types of Electrochemical Corrosion

### 1. Differential Metal Corrosion (Galvanic Corrosion)
Two dissimilar metals in contact + corrosive environment. Metal with **lower E° = anode (corrodes)**; higher E° = cathode (protected).

| Pair | Anode (corrodes) | Cathode (protected) |
|------|-----------------|-------------------|
| Fe + Cu | Fe (−0.44 V) | Cu (+0.34 V) |
| Fe + Zn | Zn (−0.76 V) | Fe (−0.44 V) |

### 2. Differential Aeration Corrosion
Different O₂ concentrations at different parts of same metal:
- **Low O₂ → Anode → corrodes**
- **High O₂ → Cathode → unaffected**
- Examples: Nail in wall (buried end = anode), wire mesh (inner strands = anode), paper pin (buried end = anode)

### 3. Waterline Corrosion
Special case of differential aeration — metal half-immersed in water:
- **Part in water (low O₂) = Anode → corrodes**
- **Part in air (high O₂) = Cathode → unaffected**
- Brown rust line just below waterline
- Examples: Ships, steel tanks partially filled, iron rods in NaCl

### 4. Pitting Corrosion
Localised, accelerated corrosion → deep pits. Small anodic area + large cathodic area → very high current density at anode → rapid intense dissolution.

- **Case 1 (Dust/Oil deposit):** Metal under deposit = low O₂ = anode; surrounding metal = large cathode → deep pit
- **Case 2 (Broken coating):** Scratched Sn on Fe → small exposed Fe = anode; large Sn coating = cathode → intense pitting at scratch

Reactions: Anode: M → Mⁿ⁺ + ne⁻; Cathode: 2H₂O + O₂ + 4e⁻ → 4OH⁻

### 5. Stress Corrosion Cracking (SCC)
Metal under **mechanical stress + specific corrosive environment** simultaneously:
- Stressed part = anode (corrodes); stress-free = cathode
- Crack deepens → structural failure (air crashes, bridge collapses, boiler explosions)

| Metal | Specific corrosive environment |
|-------|------------------------------|
| Brass | Ammoniacal solution / NH₃ vapours |
| Steel | NaOH solution and/or Cl⁻ ions |

---

## 9. Factors Affecting Corrosion Rate

| Factor | Effect |
|--------|--------|
| Nature of metal (E°) | Lower E° → more reactive → faster corrosion. Exception: Cr, Al, Ti form protective oxide layers (Cr₂O₃, Al₂O₃, TiO₂) |
| Difference in E° (ΔE) | Greater ΔE → greater EMF → faster corrosion |
| Nature of corrosion product | Insoluble/adherent/non-porous film → stops corrosion (Cr, Al). Porous/soluble → continues (Fe rust) |
| Anodic:Cathodic area ratio | Small anode + large cathode → intense pitting |
| Hydrogen overvoltage | Low overvoltage → H₂ easily released → faster cathodic reaction → faster corrosion |
| Polarisation | Slows corrosion. Anodic: Mⁿ⁺ accumulation → reduces oxidation tendency. Cathodic: O₂/H⁺ depletion → slows cathodic reaction |
| pH | Lower pH → faster corrosion (more H⁺ for cathodic reaction). Exception: Al corrodes fast in highly alkaline medium too |
| Temperature | Higher T → faster corrosion generally. Exception: if driven by dissolved gases (O₂, CO₂), higher T reduces their solubility → slower corrosion |

---

## 10. Corrosion Control

### Metallic Coatings

| Type | Coating metal | Behavior if damaged |
|------|--------------|-------------------|
| Anodic coating | More active (lower E°) than base metal, e.g., Zn/Mg on Fe | Still protects — coating sacrifices itself |
| Cathodic coating | Less active (higher E°) than base metal, e.g., Sn/Ni on Fe | Intense pitting at scratch (small anode, large cathode) — must be complete |

**Galvanising (Zn coating on Fe):**
1. Degreasing (organic solvent or caustic solution)
2. Pickling (dilute H₂SO₄ — removes rust/scale)
3. Flux treatment (ZnCl₂ + NH₄Cl — prevents re-oxidation)
4. Dip in molten Zn at **425–430°C**
5. Pass through hot rollers for uniform thickness

**Advantage:** Even if Zn peels, Fe still protected (Zn = sacrificial anode).
**Disadvantage:** Not for food storage — Zn dissolves in acids → toxic compounds.

### Inorganic Coating — Anodising (Aluminium)
Chemical conversion coating; protective Al₂O₃ layer formed — inherent part of metal.
1. Article = **anode**; electrolyte = chromic/sulfuric/phosphoric/oxalic acid at ~40°C; cathode = Pb or steel
2. Apply voltage **>40V** → 2Al + 3H₂O → Al₂O₃ + 3H₂
3. **Sealing:** Boil in water → Al₂O₃ + H₂O → Al₂O₃·H₂O (expands, seals pores)
4. Colouring: organic dyes or metal deposition (Ni, Co) in pores before sealing

### Corrosion Inhibitors
Chemicals added in small quantities to corrosive environment; only in **closed/confined systems**.

**Anodic inhibitors:** Large anions (CrO₄²⁻, WO₄²⁻) combine with Mⁿ⁺ → insoluble precipitate covers anode. Salts: Na₂CrO₄, Na₂WO₄.
⚠️ Insufficient dose → partial coverage → small anode + large cathode → **worse pitting than no inhibitor**.

**Cathodic inhibitors:**
- Block H⁺ diffusion: organic N/S compounds (amines, thiourea, mercaptans) adsorb on cathode
- Increase H₂ overvoltage: As₂O₃, Sb₂O₃, NaAsO₂ deposit on cathode
- Remove O₂ (scavengers): N₂H₄ + O₂ → N₂ + 2H₂O; 2Na₂SO₃ + O₂ → 2Na₂SO₄
- Decrease O₂ diffusion: ZnSO₄, MgSO₄, NiSO₄ → Zn²⁺ + 2OH⁻ → Zn(OH)₂ (deposits on cathode)

### Cathodic Protection

**Method 1 — Sacrificial Anode:**
- Connect structure to more active metal (Mg or Zn) → that metal corrodes, structure = cathode
- Examples: Mg/Zn bars on ship hulls, blocks on buried pipelines
- Adv: Simple, no external power; Disadv: Anodes must be replaced periodically

**Method 2 — Impressed Current:**
- Structure connected to **–ve terminal (cathode)** of external DC supply; inert electrode (graphite/Pt/Si) = anode
- Backfill of coke + bentonite around anode → reduces resistivity
- Adv: Protects large areas; Disadv: Expensive, non-uniform current → localised corrosion, hydrogen embrittlement

---

## Unit 2 — All Formulas at a Glance

- E_cell = E_cathode − E_anode; ΔG = −nFE; ΔG° = −nFE°
- E = E° − (0.0591/n) log Q [Nernst at 298 K]
- E = E° + (0.0591/n) log[Mⁿ⁺] [metal-ion]; E = E° − 0.0591 log[Cl⁻] [calomel]
- E_cell = (0.0591/n) log(c₂/c₁) [electrolyte conc. cell]
- E_G = E°_G − 0.0591 pH; pH = (E°_G − E_SCE − E_cell) / 0.0591
- E°_G = E_cell(buffer) + 0.0591 × pH(buffer) + E_SCE

---

# UNIT 3 — Energy Storage Devices, Hydrogen & Sensors

---

## 1. Batteries — Overview & Components

**Quick Review:** Battery = stores chemical energy → converts to electrical energy on demand.

| Component | Process | Role | Examples |
|-----------|---------|------|---------|
| Anode | Oxidation | Metals that oxidise easily | Zn, Pb, Li |
| Cathode | Reduction | Compounds that reduce easily | PbO₂, MnO₂, O₂ |
| Electrolyte | Ion transport | Good ionic conductivity | H₂SO₄, KOH, Nafion |
| Separator | Insulation | Prevents short circuit; allows ion transport | Polypropylene, Cellophane |

**Discharge (Galvanic):** Anode: M → Mⁿ⁺ + ne⁻; Cathode: A + ne⁻ → Aⁿ⁻; Overall: M + A → MA

**Charging (Electrolytic):** Reverse of discharge reactions.

---

## 2. Classification of Batteries

| Type | Regeneration | Example | Notes |
|------|-------------|---------|-------|
| Primary | Cannot regenerate | Dry cell, Li-MnO₂ | Galvanic only; discard after use |
| Secondary | Can regenerate | Li-ion, Pb-acid, Ni-Cd | Rechargeable; galvanic + electrolytic |
| Reserve | Electrolyte kept isolated until activation | Mg-AgCl (water), Zn-Ag₂O (KOH) | Unlimited shelf life; no self-discharge; emergency use |

**Reserve battery salient features:** Quick activation, high power for short duration, no self-discharge, unlimited shelf life, high reliability.

---

## 3. Battery Characteristics & Formulas

| Characteristic | Formula | Notes |
|---------------|---------|-------|
| Capacity | C = I × t; C = WnF/M | Ah; W = mass, n = electrons, M = molar mass |
| Mass from capacity | W = (C × M) / (n × F) | — |
| ESD | ESD = Capacity / Weight_battery | As/g or Ah/kg; weight = ALL components |
| Energy Density | ED = (i × E_cell × t) / W | Wh/kg |
| Power Density | PD = (i × E_cell) / W = ED/time | W/kg |
| Energy Efficiency | η% = (E_discharge / E_charge) × 100 | — |

**Voltage factors:** Higher E°_cell → higher voltage; T↑ → voltage↓; Q increases → marginal change.

**Cycle life limited by:** Corrosion at contact points, shedding of active material, irregular crystal growth → shorting.

**ESD comparison:** 7 g Li gives 1F of charge vs 104 g Pb for same charge.

---

## 4. Zinc-Air Battery

- **Type:** Metal-air; **EMF:** 1.4 V; **Electrolyte:** 30% KOH

**Construction:**
- Anode: Zn granules + gelling agent + small KOH
- Cathode: Graphite + MnO₂ catalyst on Ni wire mesh + air-permeable **Teflon layer** + air access holes
- Separator: Polypropylene soaked in electrolyte

**Electrode Reactions:**
- Anode: Zn + 2OH⁻ → ZnO + H₂O + 2e⁻
- Cathode: ½O₂ + H₂O + 2e⁻ → 2OH⁻
- **Overall: Zn + ½O₂ → ZnO; EMF = 1.4 V**

**Advantages:** High energy density (air from atmosphere — no mass contribution); long shelf life; miniaturisable; low cost; no ecological issues.

**Disadvantages:** Limited power output; CO₂ + 2KOH → K₂CO₃ + H₂O (reduces KOH electrolyte efficiency).

**Applications:** Hearing aids, medical devices, voice transmitters, railroad signaling.

---

## 5. Lithium Batteries

**Why Li is preferred:**
- Lightest metal → 7 g gives 1F vs 104 g for Pb → high ESD
- E°(Li⁺/Li) = −3.05 V → very high voltage (~4 V) when coupled
- Aqueous electrolytes cannot be used (Li reacts violently with water) → organic/inorganic solvents used

**Components:** Anode: Li metal; Cathode: MnO₂ or SO₂Cl₂; Electrolyte: Li salt in organic solvents (acetonitrile, propylene carbonate) or inorganic solvents (SOCl₂)

**Advantages:** High voltage (~4 V); high energy density; high tolerance (−40°C to 70°C); high ESD; flat discharge.

**Disadvantages:** Safety concerns (high reactivity of Li metal); poor cycle life due to **dendrite formation**; transport limitations.

---

## 6. Lithium-Ion Battery (LiCoO₂)

**Principle:** Li⁺ ions **intercalate/de-intercalate** between graphite and LiCoO₂. Li metal never forms → no dendrites.

**Construction:**
- Anode: Lithiated graphite (LiC₆) on **copper** current collector
- Cathode: LiCoO₂ on **aluminium** current collector
- Electrolyte: Organic carbonates (EC + DEC) with LiPF₆ or LiClO₄
- Separator: Micro-perforated polypropylene

**Cell Representation:** LiC₆ | Organic Carbonate (LiPF₆) | LiCoO₂

**Electrode Reactions:**

| | Charging | Discharging |
|-|---------|------------|
| Anode (LiCoO₂ during charging / Graphite during discharge) | LiCoO₂ → Li₍₁₋ₓ₎CoO₂ + xLi⁺ + xe⁻ | xLiC₆ → xLi⁺ + xe⁻ + xC₆ |
| Cathode (Graphite during charging / LiCoO₂ during discharge) | xLi⁺ + xe⁻ + xC₆ → xLiC₆ | Li₍₁₋ₓ₎CoO₂ + xLi⁺ + xe⁻ → LiCoO₂ |
| Overall | LiCoO₂ + xC₆ → Li₍₁₋ₓ₎CoO₂ + xLiC₆ | Li₍₁₋ₓ₎CoO₂ + xLiC₆ → LiCoO₂ + xC₆ |

**Open circuit voltage: 3.7 V**

**Advantages:** Lighter than other rechargeables; 3.7 V OCV; low self-discharge; no memory effect; good cycle life (no dendrites).

**Disadvantages:** Rising internal resistance with age; safety concerns if overheated/overcharged.

---

## 7. Fuel Cells — Principles & Types

**Quick Review:** Galvanic cell converting chemical energy of fuel–oxidant system **directly** to electrical energy. Does NOT store energy; fuel and oxidant must be continuously supplied.

**Traditional:** Chemical → Thermal → Mechanical → Electrical energy
**Fuel cell:** Chemical → Electrical energy (directly → 50–80% efficiency)

**General reactions:**
- Anode: Fuel → Oxidation product + ne⁻
- Cathode: Oxidant + ne⁻ → Reduction products
- Common fuels: H₂, CO, CH₃OH, C₂H₅OH, N₂H₄; Common oxidants: O₂, H₂O₂, halogens

**Advantages:** High efficiency (50–80%), eco-friendly, silent operation.
**Disadvantages:** High cost (expensive electrodes/catalysts), moderate power output, gases need high-pressure storage.
**Applications:** Space vehicles, vehicle traction (cars, buses), large-scale power generation.

### Types of Fuel Cells

| Type | Electrolyte | Temperature | Catalyst | Fuel | Key Notes |
|------|-----------|------------|---------|------|----------|
| AFC | Aqueous KOH | ~100°C | Non-noble feasible | H₂ only | No carbon fuels (CO₂ + 2KOH → K₂CO₃) |
| PAFC | Conc. H₃PO₄ | 160–220°C | Pt | Pure H₂ only | S compounds + CO poison Pt |
| MCFC | LiAlO₂+K₂CO₃+Li₂CO₃ (molten) | 600–650°C | Not needed | H₂ or CO | High temp eliminates catalyst need |
| PEMFC | Nafion/SPEEK polymer | 60–90°C | Pt | H₂/CH₃OH | Membrane must stay hydrated; CO poisons Pt |
| SOFC | ZrO₂ + Y₂O₃ (ceramic) | 650–1000°C | Not needed | H₂ or CO | Slow startup; very high temp |

---

## 8. Fuel Cell Efficiency

**Formula: η% = (ΔG/ΔH) × 100 = (−nFE/ΔH) × 100**

- n = electrons per mole of fuel; F = 96500 C/mol; E = EMF (V); ΔH in Joules (convert kJ → J)
- ΔG = work output; ΔH = heat input

---

## 9. H₂–O₂ Alkaline Fuel Cell

**Cell representation: H₂ | Pt | KOH | Ag | O₂**

**Construction:**
- Anode: Porous carbon + **Pt** catalyst
- Cathode: Porous carbon + **Ag** catalyst
- Electrolyte: 30–45% warm KOH

**Electrode Reactions:**
- Anode: H₂ + 2OH⁻ → 2H₂O + 2e⁻
- Cathode: ½O₂ + H₂O + 2e⁻ → 2OH⁻
- **Overall: H₂ + ½O₂ → H₂O; EMF = 1.23 V**

**Working:** H₂ diffuses → porous anode → adsorbs → reacts with OH⁻. O₂ diffuses → porous cathode → reduced to OH⁻. Water produced dilutes KOH; cell at 100°C → water escapes as steam. (Water used by Apollo astronauts.)

**Advantages:** Low temperature; alkali medium → non-noble catalyst feasible → lower cost.
**Disadvantages:** Reactants must be CO₂-free (CO₂ + 2KOH → K₂CO₃ → reduced efficiency); liquid electrolyte handling issues.

---

## 10. Direct Methanol Fuel Cell (DMFC)

**EMF: 1.21 V | Temperature: 60–90°C | Catalyst: Pt**

**Cell representation:** CH₃OH(aq) | Pt | Nafion/SPEEK | Pt | O₂

**Construction:**
- Anode + Cathode: Porous carbon + Pt
- Fuel: Aqueous CH₃OH; Oxidant: O₂
- Electrolyte: **Nafion** (fluorocarbon backbone −CF₂−CF₂− with −SO₃H groups) or **SPEEK**

**Electrode Reactions:**
- Anode: CH₃OH + H₂O → 6H⁺ + 6e⁻ + CO₂
- Cathode: 3/2 O₂ + 6H⁺ + 6e⁻ → 3H₂O
- **Overall: CH₃OH + 3/2 O₂ → 2H₂O + CO₂; EMF = 1.21 V**

**Methanol crossover:** Nafion has high CH₃OH permeability → methanol diffuses to cathode without oxidising → reduced performance. **Solution: Use SPEEK** (lower methanol permeability).

**Water management:**
- High T (>90°C) → dehydrates polymer → H⁺ conductivity lost → cracks → short circuit
- Low T (<60°C) → cell flooding → reduced efficiency → higher catalyst loading needed

**Advantages:** High energy density; long life (low T); no corrosive liquid.
**Disadvantages:** Pt required (expensive); CO poisons Pt; CO₂ + water management issues.

---

## 11. Supercapacitors (EDLC)

**Quick Review:** Very high capacitance (thousands of F vs µF for ordinary capacitors). Charge/discharge very quickly. **No redox reactions — only adsorption/desorption of ions.**

**Construction:**
- Electrodes: High surface area — porous carbon, graphene, CNTs, carbon aerogel
- Electrolyte: KOH, H₂SO₄, Na₂SO₄
- Separator: Ion-permeable porous polypropylene

**Working:** Potential applied → electrical double layer forms at electrode/electrolyte interface (nm scale charge separation). C ∝ A/d. Very small d (nm) + very large A (porous) → enormous capacitance.

**Formulas:**
- Q = C × V
- C = ε₀εᵣ × (A/d)
- **1/C_T = 1/C₊ + 1/C₋** (C₊ = cathode/electrolyte; C₋ = anode/electrolyte)

**Advantages:** Rapid charging (seconds); high power density; millions of cycles; safe (low internal resistance).
**Disadvantages:** Low energy density; high self-discharge; linear discharge voltage; high cost; power for short duration only.
**Applications:** Memory backup, hybrid car start-stop, flash photography, FM radios, emergency kits.

---

## 12. Ragone Plot

**Definition:** Double-logarithmic graph of **Energy Density (Wh/kg) [Y-axis]** vs **Power Density (W/kg) [X-axis]** to compare energy storage devices.

| Device | Position | Reason |
|--------|---------|--------|
| Fuel Cells | Top-left (high ED, low PD) | Continuously supplied fuel → high energy; slow redox kinetics → low power |
| Batteries | Middle (moderate ED, moderate PD) | Finite stored energy; moderate discharge rate |
| Supercapacitors | Bottom-right (low ED, high PD) | Very fast discharge (no redox); poor energy storage (charge doesn't hold long) |

---

## 13. Hydrogen Energy & Production

**Hydrogen economy:** Infrastructure based on H₂ as carbon-free energy carrier.

**Advantages:** Abundant; compatible with fuel cells; efficiency 65% (vs diesel 45%, gasoline 22%).
**Disadvantages:** High cost; highly flammable; mostly still from fossil fuels.

### Production Methods

| Feedstock | Process | Colour Code |
|-----------|---------|------------|
| Natural gas | Steam Reforming | Grey H₂ |
| Coal | Gasification | Black H₂ |
| Biomass | Thermolysis/Pyrolysis | Green H₂ |
| Water | Electrolysis/Photolysis | Green H₂ |

### Steam Methane Reforming (Grey H₂) — 4 Steps

**Step 1 — Desulphurisation:** Remove sulphur (poisons catalysts).

**Step 2 — Steam Reforming:**
- CH₄ + H₂O → CO + 3H₂
- T = 850–900°C; Catalyst = **Ni**; ΔH° = +206 kJ/mol (endothermic)
- Favoured at **high T, low P**

**Step 3 — Water Gas Shift:**
- CO + H₂O → CO₂ + H₂
- Catalyst = Fe₃O₄/Cr₂O₃; ΔH° = −41.0 kJ/mol (exothermic); T ~ 350°C

**Step 4 — Purification:**
- CO₂ removed; traces of CO removed by methanation (350–450°C): CO + 3H₂ → CH₄ + H₂O

### Alkaline Electrolysis of Water
- Anode: Ni alloys; Cathode: Cu, Pt; Electrolyte: KOH or NaOH (20–40 wt%); Catalyst: Ni-Co-Au, RuO₂; T: 343–363 K; P: up to 3 MPa
- Anode: 2OH⁻ → ½O₂ + H₂O + 2e⁻
- Cathode: 2H₂O + 2e⁻ → H₂ + 2OH⁻
- **Overall: 2H₂O + Energy → 2H₂ + O₂**

---

## 14. Hydrogen Storage

| Form | Conditions | Advantages | Disadvantages |
|------|-----------|-----------|--------------|
| Liquid | 20 K (−252.8°C) in cryogenic tanks | Low volume | Insulated tanks needed; ~30% energy loss for cooling |
| Gas | 350–700 bar high-pressure tanks | Less infrastructure | Large space; leak-proof tanks required |
| Solid | Adsorption/absorption on solid | Released on demand; safe | Under R&D |

**Solid-state storage materials:**
1. **MOF (Metal-Organic Frameworks):** Porous; high-density storage. Zn-MOF (MOF-5).
2. **LOHC (Liquid Organic H Carriers):** Absorb/release H₂ via chemical reaction; safe for large-scale transport. E.g., Methyl cyclopentane, Dibenzyl toluene.
3. **Interstitial Hydride:** More H₂ than same volume of liquid H₂. E.g., LaNi₅H₆.
4. **Complex Hydride:** Metal cations + H-containing anions; decomposes on heating. E.g., 3NaAlH₄ → Na₃AlH₆ + 2Al + 3H₂ (releases 7.4% H₂ at 200°C).
5. **Chemical H Storage:** H₂ released via chemical reaction. Ammonia borane (NH₃-BH₃) — hydrolysis releases H₂ with catalyst.

---

## 15. Electrochemical Sensors

### Potentiometric Sensors
- Measure **potential difference** (equilibrium of redox at electrode/electrolyte interface)
- E_cell = E° + (RT/nF) log(C_O/C_R) ; at 298K: E = (0.0591/n) log(C_O/C_R)
- Examples: ISE sensors (glass electrodes — pH, Na⁺, K⁺); Gas sensors (CO₂, NH₃, O₂ zirconia sensor)
- **Adv:** Easy to construct; accurate; selective; economical; detect various analytes
- **Limitations:** Needs calibration; impurities affect reading; temperature dependent

### Amperometric Sensors
- Measure **current at fixed applied potential**; current ∝ analyte concentration
- 3-electrode system: working electrode, auxiliary/counter electrode (inert Pt/graphite), reference electrode (Ag/AgCl or calomel)
- Examples: Glucose biosensor
- **Adv:** Can estimate non-reducible analytes (Mg²⁺, PO₄³⁻); more accurate
- **Limitations:** Potential applied for limited time only; cannot go more negative than −2V (H₂ evolution occurs)

---

## 16. Oxygen Sensor (Automotive Lambda Sensor)

**Quick Review:** Measures O₂ proportion in exhaust gas to maintain ideal air/fuel ratio in IC engines.

**Important points:**
- Stoichiometric air/fuel ratio for gasoline: **14.7:1**
- Located in exhaust stream; operates at minimum **360°C**
- Based on solid-state electrochemical fuel cell
- Compares O₂ in exhaust vs O₂ in atmosphere → generates voltage → feedback to ECU → ECU adjusts ratio

**Construction:**
- Anode: **Platinum (Pt)**
- Cathode: **Platinum (Pt)**
- Electrolyte: **ZrO₂ doped with Y₂O₃** (conducts O²⁻ ions at high temperature)

**Electrode Reactions:**
- Anode (exhaust side — low O₂): 2O²⁻ → O₂ + 4e⁻
- Cathode (reference air side — high O₂): O₂ + 4e⁻ → 2O²⁻
- **Cell voltage: E = (RT/4F) × ln(P₁/P₂)**; P₁ = O₂ in reference air; P₂ = O₂ in exhaust

**Output Voltages:**
- **0.2 V → lean mixture** (excess air) → more NOx emissions
- **0.8 V → rich mixture** (excess fuel) → more CO, unburnt fuel, carbon particles
- **~0.45 V → ideal set point**

⚠️ Must use **unleaded gasoline** — lead poisons Pt catalyst.
**Failure symptoms:** Increased tailpipe emissions, increased fuel consumption, hesitation on acceleration.

---

## 17. Glucose Biosensor

**Quick Review:** Amperometric biosensor determining blood glucose concentration in diabetics.

**Components:**
- Analyte: Glucose in blood
- Bioreceptor: Enzyme **Glucose Oxidase (GOx)** with FAD cofactor
- Transducer: **Platinum electrode**
- Display: Glucometer

**Working Mechanism (Updike and Hicks) — 3 steps:**

**Step 1 — Enzymatic Oxidation:**
- Glucose + GOx–FAD → Glucolactone + GOx–FADH₂
- FAD reduced to FADH₂ (initial electron acceptor)

**Step 2 — Cofactor Regeneration:**
- GOx–FADH₂ + O₂ → GOx–FAD + **H₂O₂**
- FAD regenerated; H₂O₂ produced as by-product

**Step 3 — Electrochemical Detection at Pt:**
- H₂O₂ → 2H⁺ + O₂ + 2e⁻ (oxidation at Pt)
- Electron transfers ∝ glucose molecules → current measured → concentration determined

**Limitations:** Haematocrit values/medication may falsify readings; limited selectivity — interference from **ascorbic acid and uric acid**.

---

## Unit 3 — All Formulas at a Glance

- C = I × t; C = WnF/M; W = (C × M)/(n × F)
- ESD = Capacity / Weight_battery (As/g or Ah/kg)
- ED = (i × E_cell × t)/W (Wh/kg); PD = (i × E_cell)/W (W/kg) = ED/time
- η%(fuel cell) = (ΔG/ΔH) × 100 = (−nFE/ΔH) × 100
- E_cell(conc. cell) = (0.0591/n) log(c₂/c₁)
- E_cell(O₂ sensor) = (RT/4F) ln(P₁/P₂)
- 1/C_T = 1/C₊ + 1/C₋ (supercapacitor); Q = CV

**Key Standard Values:**
- H₂-O₂ alkaline fuel cell: **1.23 V**
- Zn-Air battery: **1.4 V**
- Li-Ion battery OCV: **3.7 V**
- DMFC: **1.21 V**
- E°(Li⁺/Li) = **−3.05 V**
- O₂ sensor: 0.2V (lean), 0.45V (ideal), 0.8V (rich); min temp = **360°C**
- Stoichiometric air/fuel = **14.7:1**
- Steam reforming: 850–900°C; ΔH° = +206 kJ/mol
- Water gas shift: ~350°C; ΔH° = −41.0 kJ/mol
- H₂ liquefaction: −252.8°C (20 K); gas storage: 350–700 bar

---

# UNIT 4 — Nanomaterials, Functional Materials, OLEDs & Green Chemistry

---

## 1. Nanomaterials — Introduction & Definitions

**Quick Review:** Materials with at least one dimension in 1–100 nm range; properties are distinctly different from bulk.

- 1 nm = 10⁻⁹ m ≈ 10 hydrogen atoms ≈ 5 silicon atoms in a line
- Bulk: properties independent of size; Nano: properties change with size

**Key Definitions:**
- **Nanomaterials:** Materials with at least one external dimension between 1 nm and 100 nm.
- **Nanoscience:** Study of phenomena and synthesis, characterisation, exploration and exploitation of nanostructured materials.
- **Nanotechnologies:** Design, characterisation, production and application of structures by controlling shape/size at nanometer scale.

**Examples of different properties at nano scale:**
- Gold: Bulk = yellowish; Au nanoparticles = **red**
- Aluminium: Bulk = stable; Al nanoparticles = **combustible** (used as solid fuel in rockets)
- Melting point: **Lower** in nano range

**Why nano properties differ from bulk — 4 reasons:**
1. **Large fraction of surface atoms per unit volume** → surface atoms dominate properties
2. **Large surface energy** → surface atoms have fewer bonds → highly reactive
3. **Spatial (Quantum) Confinement** → bulk: continuous bands; nano (10–1000 atoms): **discrete energy levels; HOMO-LUMO gap increases as size decreases**
4. **Reduced imperfections** → smaller lattice → fewer dislocations/kinks/vacancies → enhanced mechanical strength

---

## 2. Classification of Nanomaterials

Based on how many dimensions are **outside** the nanorange (>100 nm):

| Class | Dimensions outside nano range | Examples |
|-------|------------------------------|---------|
| 0D (Zero Dimensional) | None — all dimensions in nano range | Quantum dots, Nanoclusters, Fullerene (C₆₀) |
| 1D (One Dimensional) | 1 (length can be large) | Nanotubes, Nanowires |
| 2D (Two Dimensional) | 2 (only thickness in nano range) | Nanofilms, Nanocoatings |
| 3D (Three Dimensional) | All 3, but composed of nano building blocks | Dispersions of nanoparticles, Bundles of nanowires |

---

## 3. Properties of Nanomaterials

### Surface Area Dependent Properties
- Surface area enormously increases at nanoscale → significant fraction of atoms at surface
- Catalytic activity, gas adsorption, chemical reactivity all depend on surface area
- Gold: bulk = catalytically inactive; Au nanoparticles = catalytically very active for selective redox reactions

### Thermal Properties
- **Lower melting point** — surface atoms bound by fewer bonds; fewer bonds to break per atom during melting
- **Decreased thermal conductivity** — nanoparticle size < mean free path of phonons → phonon scattering

### Optical Properties
Two reasons for unique optical behaviour:

**1. Surface Plasmon Resonance (SPR):**
- Positive lattice points + sea of electrons in metals
- Radiation falls → surface electrons polarise → collective oscillations = **plasmons**
- When plasmon frequency = radiation frequency → **resonance → radiation absorbed → material appears coloured**
- Plasmon frequency depends on **size, shape, nature** of metal → colour changes with size

**2. Increase in Energy Gap (Blue Shift):**
- As nanoparticle size ↓ → energy gap (VB to CB) ↑ → absorbed wavelength shifts to shorter values = **blue shift**

| Au sphere size | λ absorbed |
|--------------|-----------|
| >50 nm | 575 nm |
| 10–20 nm | 524 nm |
| 2–5 nm | 517 nm |

### Electrical Properties
**Conductivity decreases** for nanoparticles compared to bulk:
- **Quantum confinement:** Continuous electronic bands become discrete; band gap increases → good conductors become semiconductors/insulators at nanoscale
- **Surface scattering:** If nanoparticle dimension < mean free path of electrons → **inelastic scattering** → electrons lose velocity → conductivity decreases (elastic scattering does NOT affect conductivity)

### Mechanical Properties
**Strength is greater than bulk:**
- Mechanical properties can reach theoretical strength (1–2 orders of magnitude higher)
- Reason: **Reduced probability of defects** (dislocations, kinks, vacancies) in smaller lattice
- Example: Cu particles <50 nm = super hard; bulk Cu bends readily

### Magnetic Properties
- Ferromagnetic materials (Fe, Co, Ni): **ferromagnetism → superparamagnetism** at nanoscale (high surface energy)
- Magnetic domains can flip direction; in magnetic field → magnetised with high susceptibility
- Bulk Au and Pt = non-magnetic; at nanosize → **act as magnetic particles**
- **Superparamagnetism:** Single magnetic domain nanoparticles; thermal energy randomly flips magnetisation → behave like paramagnets but with very high magnetic susceptibility

---

## 4. Graphene — Structure & Synthesis

**Discovery:** First isolated by **Andre Geim and Konstantin Novoselov**, University of Manchester, 2004. **2010 Nobel Prize in Physics.**

**Definition:** Single layer of graphite — 2D crystalline allotrope of carbon with hexagonal (chicken wire) pattern.

**Carbon allotropes:**

| Form | Dimensionality |
|------|--------------|
| Graphite | 3D |
| Diamond | 3D |
| Buckminsterfullerene (C₆₀) | 0D |
| Carbon nanotube | 1D |
| Graphene | 2D |

**Structure:**
- Each C atom: 3 σ bonds (in-plane) + 1 π bond (out of plane)
- Bond length: **1.42 Å**
- Can **self-repair** holes using carbon-containing molecules

### Synthesis Methods

| Method | Quality | Yield | Cost | Notes |
|--------|---------|-------|------|-------|
| Epitaxial growth on SiC | High | High | High | SiC heated → Si sublimes → C forms graphene |
| Unzipping CNTs | High | Low | High | CNTs cut open along length |
| CVD (Chemical Vapor Deposition) | High | — | High | Carbon gas decomposed on Cu/Ni substrate |
| Micromechanical Exfoliation (Scotch tape) | Very high | Very low | High | Original method by Geim; adhesive tape peels layers |
| Liquid Phase Exfoliation | Moderate | Low | Low | Graphite in solvent + ultrasonication |
| Hummers Method | — | — | Moderate | Most common chemical method → graphite oxide → rGO |

**Hummers Method Steps:**
1. **Mixing:** Graphite + NaNO₃ in cold conc. H₂SO₄ (ice bath, T <20°C)
2. **Oxidation:** KMnO₄ added slowly (controlled — risk of explosion); mixture turns dark green/purple; 35–40°C, 1–2 hours
3. **Exfoliation & termination:** Water added slowly → heats up → turns brown (graphite oxide forming); H₂O₂ added → turns bright yellow (Mn VII → Mn II)
4. **Washing & drying:** Wash with distilled water → dilute HCl → ethanol; filter → dry → **Graphite Oxide**
5. **Ultrasonication:** In water → exfoliates into **Graphene Oxide (GO)**
6. **Thermal annealing at 1000–2000°C:** Converts rGO closer to graphene (most effective reduction method)

**Route:** Graphite → [KMnO₄, H₂SO₄, NaNO₃] → Graphite Oxide → [Ultrasonication] → GO → [Thermal annealing] → rGO ≈ Graphene

### Properties of Graphene
- ~200 times stronger than steel
- Excellent electrical conductor (very high charge carrier mobility)
- Excellent thermal conductor (best known)
- Transparent (absorbs only ~2.3% of light)
- Flexible; impermeable (even He cannot pass through); self-repairing

### Applications of Graphene
- **Integrated Circuits:** High mobility → replace silicon; higher frequency; faster circuits
- **Gas Sensors:** Can detect single molecule — one foreign atom changes electrical properties
- **Ultracapacitors:** High surface-to-mass ratio → may replace batteries; high recharge rate
- **Protective Coating:** Impermeable to acids/alkalis; containers, car body paint, steel structures
- **Bionic devices:** Non-corrosive, biodegradable, high conductivity, non-toxic
- **Flexible Displays:** Flexible LED screens; solar cells

---

## 5. Applications of Nanomaterials

| Area | Applications |
|------|------------|
| Medicine | Targeted drug delivery; injectable nanobots (disease detectors); **Ag nanoparticles as antibacterial agents** |
| Energy Storage | Electrodes in fuel cells (high surface area → faster electrochemical reactions) |
| Catalysis | Au, Ag nanoparticles; O₂ reduction; CO oxidation |
| Consumer Electronics | Nanophosphors for HDTV resolution (ZnSe, CdS, ZnS, PbTe); white LEDs |
| Environment | Catalytic converters (react with NOx and CO); water purification by CNTs |
| Cosmetics | **ZnO and TiO₂** in sunblock/sunscreens — UV protection at nanoscale |

---

## 6. Polymers — Types, Structure & Properties

**Definitions:**
- **Polymer:** Macromolecule formed by repeated joining of simple molecules. Can be linear, branched, or cross-linked.
- **Monomer:** Simple molecule with 2+ binding sites that can link to form polymer.
- **Polymerisation:** Process by which monomers → polymers.
- **Functionality:** Number of functional groups/bonding sites in a monomer.
- **Degree of Polymerisation (DP) = MW of polymer / MW of monomer**

### A. Addition Polymerisation
- Self-addition of **unsaturated monomers** without by-products
- Rapid; linear or branched; MW = integral multiple of monomer MW
- Mechanisms: Free radical (benzoyl peroxide), Cationic (AlCl₃), Anionic (BuLi), Coordination (Ziegler-Natta: Al(C₂H₅)₃ + TiCl₃)
- Examples: PVC, Polystyrene, Teflon, Polyethylene

**Types of copolymers:**
- Alternating (ABABAB…); Random (AABBABBA…); Block (AAAA-BBBB-AAAA…); Graft (branches of B on backbone of A)

### B. Condensation Polymerisation
- Intermolecular condensation between functional groups with **continuous elimination of by-products** (H₂O, NH₃, HCl)
- Stepwise; catalysed by acids/alkali; linear/branched/cross-linked; MW NOT integral multiple of monomer
- Monomer must have **two active functional groups**
- Examples: Nylons, Polyesters (Terylene), Phenol-formaldehyde resin

| Feature | Addition | Condensation |
|---------|---------|-------------|
| By-products | None | H₂O, NH₃, HCl etc. |
| Initiator/catalyst | Initiator needed | Catalyst needed |
| Polymer type | Linear or branched | Linear, branched, or cross-linked |
| MW | Integral multiple of monomer | Not integral multiple |
| Monomer | Unsaturated | Two active functional groups |

### Glass Transition Temperature (Tg)
**Definition:** Temperature at which amorphous polymer transitions from **glassy (stiff, hard, brittle)** to **rubbery (soft, flexible)**. Above Tg = rubbery; below Tg = glassy.

- Tg = measure of **flexibility** of polymer; predicts response to mechanical stress
- Above Tm = viscous liquid (molecular mobility); Tg < Tm

### Structure–Property Relationships

**A. Crystallinity:**
- Linear chains without bulky pendant groups → more crystalline (e.g., PE)
- Polar groups → strong secondary forces (dipole-dipole, H-bonding) → denser packing → more crystalline (PVC > PE)
- Stereoregularity: **Isotactic > syndiotactic > atactic** (crystallinity)
- High crystallinity → sharper melting point, greater rigidity, strength, density

**B. Tensile Strength:**
- Low MW → soft/gummy; High MW → tough/heat resistant (strength increases then plateaus)
- Polar groups → chains resist slipping → higher tensile strength (PVC, Nylon)
- Cross-linking → increases tensile strength

**C. Elasticity:**
- Long coiled/entangled chains straighten on stretching → return on release
- Polar groups → stiff (less elastic); Non-polar groups → weak van der Waals → more elastic
- Bulky/aromatic/cyclic groups → decreases elasticity
- Natural rubber: elasticity improved by **cross-linking (vulcanisation)**
- Non-elastic materials: **plasticisers** (e.g., Ph₃PO₄) added between chains → more flexible

**D. Chemical Resistance:**
- "Like dissolves like": polar groups → attacked by polar solvents; non-polar groups → attacked by non-polar solvents
- Residual unsaturation (double bonds) → oxidative degradation in air/UV (e.g., natural rubber)
- Dense packing → solvent can't penetrate → higher resistance (e.g., Teflon)
- Higher MW and greater cross-linking → lower solubility

**E. Plastic Deformation (Thermoplastic vs Thermosetting):**

| Feature | Thermoplastic | Thermosetting |
|---------|--------------|--------------|
| Behaviour | Softens on heating, hardens on cooling | Permanently hardens on heating |
| Remoulding | Can be remoulded | Cannot — degrades/chars |
| Bonding | Weak forces (vdW, dipole, H-bond) | Strong covalent cross-links |
| Examples | PE, PP, Polystyrene | Bakelite, Urea-formaldehyde |

---

## 7. Commercial Polymers

### PMMA (Plexiglass/Perspex)
- **Type:** Addition polymer (radical polymerisation)
- **Monomer:** Methyl methacrylate (MMA)
- **Synthesis:** Acetone + HCN → acetone cyanohydrin → [CH₃OH, conc. H₂SO₄] → MMA → [H₂O₂, 60–70°C, emulsion polymerisation] → PMMA
- **Tg = 105°C**; amorphous (bulky groups prevent crystallisation); 100% transparent; resistant to many chemicals but soluble in organic solvents (acetone, chlorinated hydrocarbons, esters)
- **Applications:** Aircraft windows, signal boards, artificial teeth, contact lenses, paints, transparent bottles — good substitute for glass but **poor scratch resistance**

### Epoxy Resin (Araldite/Epon)
- Contains **epoxy group** (3-membered ring: O bonded to 2 C atoms)
- **Synthesis:** Condensation of **Bisphenol A + Epichlorohydrin** (excess epichlorohydrin → epoxy end groups)
- **Curing agents:** Diamines, dicarboxylic acids, or acid anhydrides → 3D cross-linked thermoset
- Bisphenol A + Epichlorohydrin → [Condensation] → Epoxy Resin + Curing agent → Cross-linked thermoset
- **Properties:** Excellent adhesion; resistant to water, acid, alkali; high mechanical strength; absorbs less moisture; good insulation
- **Applications:** Aerospace/defence; bonds glass, metal, leather, wood, ceramic; antiskid floor coating; laminating; crease-resistant fabrics

### Kevlar (Aramid Fibre)
- **Type:** Aromatic polyamide (condensation polymer)
- **Chemical name:** Poly-paraphenylene terephthalamide
- **Synthesis:** Condensation of **1,4-phenylenediamine + 1,4-benzenedicarbonyl chloride (terephthaloyl chloride)**
  - n H₂N-C₆H₄-NH₂ + n ClOC-C₆H₄-COCl → [-NH-C₆H₄-NH-CO-C₆H₄-CO-]ₙ + 2n HCl
- **Why ~5× stronger than steel?** Many inter-chain **H-bonds** (C=O···H-N) + **aromatic π-π stacking** between adjacent strands

**Properties:** High tensile strength; high strength-to-weight ratio; high chemical inertness; low thermal expansion; flame resistant; high impact/cut resistance.

**Applications:** Bullet-proof vests, combat helmets, ballistic face masks, gloves, bicycle tire linings, boat hulls, aircraft panels, F1 cars, rotor blades, tennis rackets, archery strings.

**Disadvantages:** Needs special cutting tools; **hygroscopic** (absorbs moisture).

**Kevlar composite:** Kevlar fibre + epoxy resin → F1 car bodies, helicopter rotor blades, kayaks.

### Carbon Fibre & CFRP
- Fibres ~5–10 µm diameter; composed mostly of carbon atoms; thousands twisted to form yarn
- **Precursor:** Polyacrylonitrile (PAN) — from polymerisation of acrylonitrile

**Synthesis steps:**
1. Polymerisation: Acrylonitrile → PAN
2. Cyclisation: PAN heated at low T → cyclic (ladder) structure
3. Oxidative treatment: High T in air → introduce O, stabilise structure
4. **Graphitisation: 2000°C in inert atmosphere** → nearly pure carbon with graphite-like structure

**Structure:** Sheets of carbon like graphite but sheets interlock. Turbostratic carbon fibre: sheets haphazardly folded/crumpled.

**Properties:** High strength-to-weight ratio; corrosion resistant; good tensile strength; non-poisonous, biologically inert; low thermal expansion.

**Applications:** Aerospace, automotive (hoods), sports goods, medical (prostheses, implants, tendon repair), EMI/RF shielding, robot arms.

**Disadvantages:** Expensive production (high energy); volatile by-products including **HCN**.

**CFRP:** Carbon fibre + epoxy resin matrix; lightweight, high strength-to-weight ratio, low thermal expansion, high electrical conductivity.

---

## 8. Conducting Polymers

**Quick Review:** Polymers are generally insulators (no free electrons). Conducting polymers have **highly delocalised π-electron systems** — also called **synthetic metals**.

**Development:**
- **1977:** Heeger, MacDiarmid & Shirakawa — conductivity of polyacetylene increased **13×** by doping with electron acceptors/donors
- **2000: Nobel Prize in Chemistry** (Heeger, MacDiarmid, Shirakawa)

**Requirements:** Linear structure + extensive conjugation in backbone (alternating single/double bonds)

**Examples:** Polyacetylene, Polyphenylene, Polyaniline, Polypyrrole, Polythiophene

**Types of Doping:**

| Type | Process | Current carriers | Agents |
|------|---------|-----------------|--------|
| Oxidative (p-doping) | π-backbone partially oxidised | Positive sites on backbone | I₂ in CCl₄, HBF₄ |
| Reductive (n-doping) | π-backbone partially reduced | Negative sites on backbone | Sodium naphthalide in THF |
| Protonic acid doping | Protonation of backbone atoms | +ve and −ve species | Protonic acid; e.g., HCl on polyaniline |

**Polyaniline doping:** Leucoemeraldine → [partial oxidation with (NH₄)₂S₂O₈] → Emeraldine Base → [protonation with 1M HCl] → **Emeraldine Salt (conducting form)**

**Applications:** Electrode material in rechargeable batteries; conductive tracks on PCBs; sensors (humidity, gas, radiation, biosensor); electrochromic display windows; LEDs.

---

## 9. Biodegradable Polymers

**Definition:** Polymers decomposed by microorganisms into H₂O, CO₂, CH₄ and biomass.

**Two-step biodegradation:**
1. **Fragmentation:** Macromolecule broken into smaller chains by hydrolysis or oxidation
2. **Mineralisation:** Microorganisms convert to CO₂, CH₄, H₂O, biomass

**Susceptibility depends on:** MW, branching, crystallinity, solubility; environmental conditions (pH, O₂, light, T, humidity, microorganisms). Polyesters, polyamides, polyurethanes are biodegradable.

**Examples:**

| Abbreviation | Name |
|-------------|------|
| PVOH | Polyvinyl Alcohol |
| PLA | Polylactic Acid |
| PHB | Polyhydroxybutyrate |
| PHBV | Poly(hydroxybutyrate-co-hydroxyvalerate) |

**Natural:** Collagen, Cellulose, Starch, Chitosan, Fibrin
**Synthetic:** Polyesters, Polyanhydrides, Polyphosphazenes

**Advantages:** Less energy in manufacture; no harmful products on decomposition; decompose faster.
**Applications:** Agricultural mulch-films, food packaging, disposable tableware, medical drug delivery, dissolvable sutures.

---

## 10. Molecular Weight of Polymers

**Polymers = mixture of chains of different lengths → expressed as average MW.**

**A. Number Average Molecular Weight (M̄ₙ):**
- **M̄ₙ = Σ(nᵢMᵢ) / Σ(nᵢ)**
- Depends only on **number** of polymer units, not their size
- Determined by **colligative properties** (osmotic pressure, boiling point elevation, freezing point depression)

**B. Weight Average Molecular Weight (M̄ᵥ):**
- **M̄ᵥ = Σ(nᵢMᵢ²) / Σ(nᵢMᵢ)**
- Depends on **size** of polymer units
- Determined by **sedimentation velocity or light scattering**

**Polydispersity Index (PDI) = M̄ᵥ / M̄ₙ**
- PDI = 1: monodisperse (all chains equal length)
- PDI > 1: polydisperse (broader distribution)

**C. Viscosity Average Molecular Weight (M̄ᵥ):**
- **M̄ᵥ = [Σ(nᵢMᵢ^(1+a)) / Σ(nᵢMᵢ)]^(1/a)**
- 'a' = Mark-Houwink parameter (0.5 ≤ a ≤ 0.9); depends on polymer-solvent system at specific T
- Closer to M̄ᵥ (weight average) as larger molecules contribute more to viscosity

---

## 11. OLEDs (Organic Light Emitting Diodes)

**Quick Review:** Solid-state devices using organic molecules instead of inorganic semiconductors. First OLED made by Eastman Kodak in 1980s.

**Working principle:** Similar to conventional LED but uses organic molecules. Emissive electroluminescent organic layer emits light in response to electric current.

### Construction — Layer Structure (bottom to top)
1. **Substrate:** Glass (support)
2. **Anode:** **ITO (Indium Tin Oxide)** — transparent; injects holes
3. **Conductive layer (Hole Transport Layer):** Organic molecules/polymers transporting holes from anode; e.g., **polyaniline**
4. **Emissive layer (Electron Transport Layer):** Organic molecules transporting electrons from cathode; recombination → light; e.g., **polyfluorene, Alq₃** (tris(8-hydroxyquinolinato)aluminium)
5. **Cathode:** Injects electrons; **Mg (Magnesium) or Al**

### Working
1. Voltage applied across anode and cathode
2. Electrons flow from cathode through organic layers
3. Electrons transported to emissive layer from cathode
4. Holes transported to emissive layer from anode
5. Electrons + holes **recombine in emissive layer → emit photons (light)**
6. Colour depends on organic molecules; intensity depends on current density

**Why multi-layered structure?** Organic materials have lower conductivity than inorganic. HOMOs/LUMOs of layered materials act as **small steps to guide charge carriers** to emissive layer → smoother carrier flow.

**Emissive layer:** Photoluminescent (fluorescent and phosphorescent) organic materials; electrically conductive due to delocalised π-electrons; conductivity in semiconductor range. **Blue-emitting materials have larger band gap.**

### Advantages
- Thinner, lighter, more flexible than LED/LCD
- **No backlight required** (layers are self-emissive)
- Much less power consumption
- Easier to produce; can be larger
- Wide field of view (~170°)
- High colour contrast; faster response time

### Disadvantages
- Shorter lifespan
- Expensive manufacturing
- Water easily damages them

### Applications
- Smartphone displays (Samsung Galaxy, iPhone OLED); TV screens (LG OLED); flexible/foldable displays; wearables; automotive displays; architectural lighting panels

---

## 12. Green Chemistry — 12 Principles

**Definition (Anastas & Warner):** "Utilization of a set of principles that reduces or eliminates the use or generation of hazardous substances in the design, manufacture and application of products."

**Focus:** Reduction/elimination of toxic substances; alternative routes to minimise environmental impact; sustainable development.

| # | Principle | Key Point |
|---|-----------|----------|
| 1 | **Prevent Waste** | Synthesise only targeted product; minimum/no by-products |
| 2 | **Atom Economy** | Maximise incorporation of all materials into final product; % Atom Economy = (FW of atoms utilised / FW of all reactants) × 100; Addition & Rearrangement reactions = 100% |
| 3 | **Less Hazardous Synthesis** | Minimise/prevent production of hazardous substances; e.g., polyurethane with CO₂ instead of phosgene |
| 4 | **Designing Safer Chemicals** | Preserve function while reducing toxicity; especially cosmetics, pharmaceuticals |
| 5 | **Safer Solvents** | Use safer solvents: aqueous medium, liquid CO₂, ionic liquids, solvent-free systems; avoid chloroform, pyridine |
| 6 | **Design for Energy Efficiency** | Reactions at mild conditions; less time; use bio-catalysts/heterogeneous catalysts; solvent-free synthesis, microwave, ultrasound |
| 7 | **Renewable Feedstock** | Use biomass, agricultural waste rather than coal/petroleum |
| 8 | **Reduce Derivatives** | Avoid unnecessary derivatisation (protection/deprotection); wastes atoms; reduces atom economy |
| 9 | **Catalysis** | Catalytic reactions: faster, recyclable, selective, less energy, high yield high purity vs stoichiometric |
| 10 | **Design for Degradation** | Chemicals should degrade to innocuous substances after use; especially insecticides, pesticides, polymers |
| 11 | **Real-Time Pollution Prevention** | Continuous monitoring; unreacted reactants recycled; minimise hazardous substance formation |
| 12 | **Accident Prevention** | Minimise potential for fires, explosions, toxic releases; safety mechanisms in plants; ⚠️ Bhopal Gas Tragedy — worst industrial disaster |

**Atom Economy by reaction type:**
- Addition reaction: **100%**
- Rearrangement reaction: **100%**
- Substitution: Moderate
- Elimination: Low

---

## Unit 4 — Key Facts at a Glance

- Graphene bond length: **1.42 Å**; Nobel 2010 (Geim & Novoselov)
- Conducting polymers Nobel: **2000** (Heeger, MacDiarmid, Shirakawa)
- Kevlar: ~5× stronger than steel; H-bonds + π-π stacking explain strength
- PMMA Tg: **105°C**
- Carbon fibre graphitisation: **2000°C** inert atmosphere
- OLED anode: **ITO**; cathode: **Mg or Al**; emissive layer: polyfluorene, Alq₃
- PDI = M̄ᵥ / M̄ₙ; monodisperse PDI = 1
- Green chemistry: Atom economy = 100% for addition and rearrangement reactions
- Superparamagnetism: single magnetic domain nanoparticles
- Blue shift: smaller nanoparticles → larger band gap → absorbed wavelength shifts to shorter values

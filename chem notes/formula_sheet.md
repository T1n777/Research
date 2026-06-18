# Engineering Chemistry: Complete Formula Sheet & Numericals

This document compiles all the formulas, equations, and solved numerical problems from the course notes across all 4 units.

---

## Unit 1: Molecular Spectroscopy, Phase Equilibria & Computational Chemistry

### 📐 Formulas
**1. Electromagnetic Radiation:**
- Photon Energy: `E = hν = hcν̃ = hc/λ`
- Frequency: `ν = c/λ`
- Wavenumber: `ν̃ = 1/λ = ν/c`
- Wavenumber to nm conversion: `ν̃ (cm⁻¹) = 10⁷ / λ (nm)`
*(Constants: c = 3 × 10¹⁰ cm/s or 3 × 10⁸ m/s, h = 6.626 × 10⁻³⁴ J·s, 1 cm⁻¹ = 1.986 × 10⁻²³ J)*

**2. Microwave (Rotational) Spectroscopy:**
- Reduced Mass: `μ = (m₁ × m₂) / (m₁ + m₂)`
- Moment of Inertia: `I = μr₀²`
- Rotational Energy (Joules): `E_J = (h² / 8π²I) J(J+1)`
- Rotational Energy (cm⁻¹): `ε_J = B J(J+1)`
- Rotational Constant: `B = h / (8π²Ic)` cm⁻¹
- Energy absorbed in transition `J` → `J+1`: `Δε = 2B(J+1)` cm⁻¹

**3. Infrared (Vibrational) Spectroscopy:**
- **Simple Harmonic Oscillator (SHO):**
  - Energy (cm⁻¹): `ε_v = ν̃_osc(v + ½)`
  - Oscillation Wavenumber: `ν̃_osc = (1/2πc) √(k/μ)` cm⁻¹
  - Zero Point Energy (ZPE): `ε₀ = ½ ν̃_osc`
- **Anharmonic Oscillator (Morse Potential):**
  - Energy (cm⁻¹): `ε_v = ν̃_e(v + ½) - ν̃_e x_e(v + ½)²`  *(where x_e is the anharmonicity constant)*
  - Fundamental Transition (v=0 → 1): `ν̃_e(1 - 2x_e)`
  - First Overtone (v=0 → 2): `2ν̃_e(1 - 3x_e)`
  - Second Overtone (v=0 → 3): `3ν̃_e(1 - 4x_e)`

### 🔢 Solved Numericals

**Q1. A spectral line has a wavenumber of 2500 cm⁻¹. Calculate its frequency and energy.**
**Solution:**
- Frequency: `ν = c × ν̃` = `(3 × 10¹⁰ cm/s) × 2500 cm⁻¹` = **7.5 × 10¹² Hz**
- Energy: `E = hν` = `(6.626 × 10⁻³⁴ J·s) × (7.5 × 10¹² Hz)` = **4.97 × 10⁻²¹ J**

**Q2. Calculate the rotational constant of NO. Given: mN = 14.004 amu, mO = 15.9994 amu, r₀ = 115 pm.**
**Solution:**
- `μ = (14.004 × 15.9994) / (14.004 + 15.9994) × 1.66×10⁻²⁷` = 1.2396 × 10⁻²⁶ kg
- `I = μr₀²` = 1.2396×10⁻²⁶ × (115×10⁻¹²)² = 1.6394 × 10⁻⁴⁶ kg·m²
- `B = h / (8π²Ic)` = (6.6×10⁻³⁴) / [8 × (3.14)² × 1.6394×10⁻⁴⁶ × 3×10⁸] = 170.13 m⁻¹ = **1.70 cm⁻¹**

**Q3. Find the bond length of HCl if the first rotational line is at 21.18 cm⁻¹. mH = 1.008 amu, mCl = 35.45 amu.**
**Solution:**
- First line is at 2B = 21.18 cm⁻¹ → `B = 10.59 cm⁻¹` = 10.59 × 10² m⁻¹
- `μ = (1.008 × 35.45) / (1.008 + 35.45) × 1.66×10⁻²⁷` = 1.627 × 10⁻²⁷ kg
- `I = h / (8π²Bc)` = (6.6×10⁻³⁴) / [8 × (3.14)² × 10.59×10² × 3×10⁸] = 2.634 × 10⁻⁴⁷ kg·m²
- `r₀ = √(I/μ)` = √(2.634×10⁻⁴⁷ / 1.627×10⁻²⁷) = 1.27 × 10⁻¹⁰ m = **127 pm**

**Q4. For HBr, the first rotational line is at 17.19 cm⁻¹. The fundamental vibrational absorption is at 2559.08 cm⁻¹, and the first overtone is at 5027.54 cm⁻¹. Find the bond length r₀ and anharmonicity constant x_e.**
**Solution:**
- `2B = 17.19` → `B = 8.595 cm⁻¹`. Using `μ = 1.6395×10⁻²⁷ kg` → `I = 3.30×10⁻⁴⁷ kg·m²`. `r₀ = √(I/μ)` = **141.8 pm**.
- Fundamental: `ν̃_e(1 - 2x_e) = 2559.08`
- First Overtone: `2ν̃_e(1 - 3x_e) = 5027.54`
- Ratio: `2(1 - 3x_e) / (1 - 2x_e) = 5027.54 / 2559.08` = 1.9646
- Solving gives: **x_e = 0.0171**

**Q5. For KCl, the fundamental vibrational frequency is 378 cm⁻¹. Find the reduced mass, force constant k, and zero point energy.**
**Solution:**
- `μ(KCl) = (39 × 35.5) / (39 + 35.5) / (6.023×10²³)` = **3.085 × 10⁻²⁶ kg**
- `k = 4π² ν̃² c² μ` = 4 × (3.14)² × (378×10²)² × (3×10⁸)² × 3.085×10⁻²⁶ = **156.3 N/m**
- Zero Point Energy = `½ ν̃_osc` = ½ × 378 = **189 cm⁻¹**

**Q6. For CO, the force constant k is 1840 N/m. Find the oscillation frequency and wavenumber.**
**Solution:**
- `μ(CO) = (12.000 × 15.9994) / (12.000 + 15.9994) × 1.66×10⁻²⁷` = 1.1383 × 10⁻²⁶ kg
- `ν = (1/2π) √(k/μ)` = (1 / 2×3.14) √(1840 / 1.1383×10⁻²⁶) = **6.402 × 10¹³ Hz**
- `ν̃ = ν/c` = 6.402×10¹³ / 3×10⁸ = 2.134×10⁵ m⁻¹ = **2134 cm⁻¹**

---

## Unit 2: Electrochemistry & Corrosion

### 📐 Formulas
**1. Cell Potential & Thermodynamics:**
- Cell potential: `E_cell = E_cathode - E_anode`
- Free energy change: `ΔG = -nFE_cell`
- Standard free energy: `ΔG° = -nFE°_cell`

**2. Nernst Equation:**
- General (at 298 K): `E = E° - (0.0591/n) log Q`
- Metal-ion electrode: `E = E° + (0.0591/n) log[Mⁿ⁺]`
- Calomel electrode: `E = E° - 0.0591 log[Cl⁻]`
- Hydrogen gas electrode: `E = E° - (0.0591/2) log(p_H₂ / [H⁺]²)`
- Redox electrode (e.g. Sn⁴⁺/Sn²⁺): `E = E° - (0.0591/2) log([Sn²⁺]/[Sn⁴⁺])`

**3. Concentration Cells:**
- Electrolyte conc. cell: `E_cell = (0.0591/n) log(c₂/c₁)` (where c₂ > c₁)
- Electrode conc. cell: `E_cell = (0.0591/n) log(c₁/c₂)`
- Gas conc. cell: `E_cell = (0.0591/2) log(p₁/p₂)` (where p₁ > p₂)

**4. Glass Electrode & pH Determination:**
- Glass electrode potential: `E_G = E°_G - 0.0591 pH`
- Cell EMF: `E_cell = E_G - E_calomel = E°_G - 0.0591 pH - E_SCE`
- pH Calculation: `pH = (E°_G - E_SCE - E_cell) / 0.0591`
- Calibration (Finding E°_G): `E°_G = E_cell(buffer) + 0.0591×pH(buffer) + E_SCE`

*(Standard Values: F = 96500 C/mol, E_SCE = 0.2422 V, Decinormal Calomel = 0.3358 V, Normal Calomel = 0.2824 V)*

### 🔢 Solved Numericals

**Q1. Cell: Fe/Fe²⁺(0.05M) // Ag⁺(0.1M)/Ag | E°(Fe²⁺/Fe) = –0.44V, E°(Ag⁺/Ag) = +0.80V. Find E°cell and Ecell at 25°C.**
**Solution:**
- Anode: Fe (oxidised). Cathode: Ag (reduced). `n = 2`.
- `E°cell = E°cathode – E°anode = 0.80 – (–0.44)` = **1.24 V**
- `Q = [Fe²⁺]/[Ag⁺]²` = 0.05 / (0.1)² = 0.05 / 0.01 = 5
- `Ecell = 1.24 – (0.0591/2) × log(5)` = 1.24 – 0.02955 × 0.699 = 1.24 – 0.02066 = **1.2193 V**

**Q2. Concentration cell: Pt/H₂(8atm)/HCl(0.3M)/H₂(2atm)/Pt. Find Ecell at 25°C.**
**Solution:**
- Gas electrode concentration cell. Higher pressure (8atm) → Anode. Lower pressure (2atm) → Cathode. `n = 2`.
- `Ecell = (0.0591/2) × log(p_anode / p_cathode)` = (0.0591/2) × log(8/2)
- `Ecell = 0.02955 × log(4)` = 0.02955 × 0.6021 = **0.01779 V**

**Q3. Decinormal calomel (cathode) + Saturated calomel (anode), Ecell = 0.0988V at 25°C. Find [Cl⁻] in saturated electrode.**
**Solution:**
- Cell: `Pt/Hg/Hg₂Cl₂/Cl⁻(x) // Cl⁻(0.1M)/Hg₂Cl₂/Hg/Pt`
- `Ecell = E_cathode – E_anode = [E° – 0.0591 log(0.1)] – [E° – 0.0591 log(x)]`
- `0.0988 = 0.0591 × [log(x) – log(0.1)] = 0.0591 × log(x/0.1)`
- `0.0988 / 0.0591 = log(x) - log(0.1)` → `1.6717 = log(x) + 1`
- `log(x) = 0.6717` → `x = Antilog(0.6717)` = **4.69 M**

**Q4. Cell: Ag/AgCl/Cl⁻(0.1M) // Fe²⁺(0.29M),Fe³⁺(0.18M)/Pt | E°(Fe³⁺/Fe²⁺) = 0.77V, E°(Ag/AgCl/Cl⁻) = 0.222V**
**Solution:**
- Anode: `Ag + Cl⁻ → AgCl + e⁻`. Cathode: `Fe³⁺ + e⁻ → Fe²⁺`. `n = 1`.
- `E°cell = 0.77 – 0.222` = **0.548 V**
- `Q = [Fe²⁺] / ([Fe³⁺][Cl⁻])` = 0.29 / (0.18 × 0.1) = 16.11
- `Ecell = 0.548 – (0.0591/1) × log(16.11)` = 0.548 – 0.0591 × 1.2073 = 0.548 – 0.07135 = **0.4767 V**

**Q5. Concentration cell: Au/Au³⁺(0.05M) // Au³⁺(0.12M)/Au at 25°C.**
**Solution:**
- Electrolyte concentration cell. Higher concentration (0.12M) → Cathode. `n = 3`.
- `Ecell = (0.0591/3) × log(c_cathode / c_anode)` = (0.0591/3) × log(0.12 / 0.05)
- `Ecell = 0.01970 × log(2.4)` = 0.01970 × 0.3802 = **0.00749 V**

**Q6. Decinormal calomel + redox electrode Pt/Cu²⁺(0.58M),Cu⁺(0.08M) | E°(calomel) = 0.281V, E°(Cu²⁺/Cu⁺) = 0.153V**
**Solution:**
- Anode: `Cu⁺ → Cu²⁺ + e⁻`. Cathode (Calomel): `Hg₂Cl₂ + 2e⁻ → 2Hg + 2Cl⁻`. `n = 2`.
- `E°cell = E°cathode – E°anode` = 0.281 – 0.153 = **0.128 V**
- `Q = [Cl⁻]²[Cu²⁺] / [Cu⁺]²` = (0.1)² × (0.58) / (0.08)² = 0.0058 / 0.0064 = 0.906
- `Ecell = 0.128 – (0.0591/2) × log(0.906)` = 0.128 – (0.02955 × (–0.0426)) = 0.128 + 0.00126 = **0.1362 V**

**Q7. Cell: Fe/Fe²⁺(0.1M) // Au³⁺(0.5M)/Au | E°(Au³⁺/Au) = 1.52V, E°(Fe²⁺/Fe) = –0.44V**
**Solution:**
- Balance electrons: `n = 6` (3Fe + 2Au³⁺ → 3Fe²⁺ + 2Au)
- `E°cell = 1.52 – (–0.44)` = **1.96 V**
- `Q = [Fe²⁺]³ / [Au³⁺]²` = (0.1)³ / (0.5)² = 0.001 / 0.25 = 0.004
- `Ecell = 1.96 – (0.0591/6) × log(0.004)` = 1.96 – 0.00985 × (–2.398) = 1.96 + 0.02362 = **1.9836 V**

**Q8. Glass electrode + SCE: Ecell₁ = 0.215V (buffer pH=7), Ecell₂ = 0.385V (unknown pH). E(SCE) = 0.244V**
**Solution:**
- Find E°G using buffer: `E°G = Ecell + 0.0591×pH + E(SCE)`
- `E°G = 0.215 + (0.0591×7) + 0.244` = 0.215 + 0.4137 + 0.244 = **0.8727 V**
- Find unknown pH: `pH = (E°G – E(SCE) – Ecell) / 0.0591`
- `pH = (0.8727 – 0.244 – 0.385) / 0.0591` = 0.2437 / 0.0591 = **4.12**

---

## Unit 3: Energy Storage & Sensors

### 📐 Formulas
**1. Battery Characteristics:**
- Capacity (C): `C = I × t = (W × n × F) / M`
- Electricity Storage Density (ESD): `ESD = Capacity / Weight of battery` (Ah/kg or As/g)
- Energy Density (ED): `ED = (I × E_cell × t) / W = (C × E_cell) / W` (Wh/kg)
- Power Density (PD): `PD = (I × E_cell) / W = ED / t` (W/kg)
*(Where C = capacity in Ah or As, W = mass, n = electrons transferred, F = 96500 C/mol, M = molar mass, I = current, E_cell = EMF, t = time)*

**2. Fuel Cells:**
- Efficiency: `η% = (ΔG / ΔH) × 100 = (−nFE / ΔH) × 100`

**3. Supercapacitors:**
- Charge: `Q = C × V`
- Total Capacitance (series interface): `1/C_T = 1/C₊ + 1/C₋`

**4. Sensors (Nernst eq):**
- `E_cell = E° + (0.0591/n) log(C_O / C_R)`

### 🔢 Solved Numericals

**Q1. A battery using Zn as anodic material lasts for 2 hours when a constant current of 1.25 A is drawn from it. What weight of Zn is present in the battery if the reaction at the anode is Zn → Zn²⁺ + 2e⁻? If the electricity storage density of the battery is 180 As/g, determine the weight of the entire battery. (Given: molar mass of Zn = 65 g, F = 96500 C/mol).**
**Solution:**
- Capacity `C = I × t` = `1.25 A × (2 × 60 × 60 s)` = 9000 As.
- Weight of Zn: `C = (W × n × F) / M` → `W = (C × M) / (n × F)`
- `W = (9000 × 65) / (2 × 96500)` = **3.03 g of Zn**
- `ESD = Capacity / Weight of battery` → `Weight of battery = Capacity / ESD`
- `Weight of battery = 9000 As / 180 As/g` = **50 g**

**Q2. Calculate the electricity storage density of a lithium battery which stores 2.0 g of lithium. The total weight of the battery is 65 g. Give the answer in Ah/kg. (Given: Atomic mass of lithium is 7).**
**Solution:**
- Capacity `C = (W × n × F) / M` = `(2.0 × 1 × 96500) / 7` = 27571.428 As
- Convert to Ah: `27571.428 / 3600` = 7.658 Ah
- Total weight = 65 g = 65 × 10⁻³ kg
- `ESD = Capacity / Weight` = `7.658 Ah / (65 × 10⁻³ kg)` = **117.8 Ah/kg**

**Q3. Calculate the energy density and power density of a 20 kg lead acid battery which contains 5 kg anode material and discharges constant current for 10 hours. The voltage of the battery is 2 V. (Given: Atomic mass of lead is 207.2, n=2, F=96500).**
**Solution:**
- Capacity `C = (W × n × F) / M` = `(5 × 10³ g × 2 × 96500) / 207.2` = 4,657,335.9 As
- Convert to Ah: `4,657,335.9 / 3600` = 1293.7 Ah
- Energy density `ED = (Capacity × Voltage) / Weight` = `(1293.7 Ah × 2 V) / 20 kg` = **129.4 Wh/kg**
- Power density `PD = Energy Density / time` = `129.4 Wh/kg / 10 h` = **12.9 W/kg**

---

## Unit 4: Nanomaterials, Functional Materials, Polymers, OLEDs & Green Chemistry

### 📐 Formulas
**1. Polymers:**
- Degree of Polymerisation: `DP = Molecular weight of polymer / Molecular weight of monomer`

**2. Nanoscale:**
- `1 nm = 10⁻⁹ m`

*(Unit 4 is primarily conceptual, covering definitions, theory, characteristics, synthesis of nanomaterials/polymers/OLEDs, and the 12 principles of green chemistry; hence, it contains no mathematical numerical problems in the study material.)*

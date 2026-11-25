# Research Notes — Mathematical Formulation & Limitations

## 1. 3-Channel Diffusive Lenia Morphogenesis
Each organism owns a localized `40×40×3` continuous field $A = (A_0, A_1, A_2)$:
- **$A_0$ (Cytoplasm)**: Convolved with a heading-aligned, stretched, spiral-twisted multi-ring kernel $K_\theta$ (adapted from Lenia, Bert Chan 2018, MIT) and updated via Gaussian growth $G(u; \mu_{\text{eff}}, \sigma) = 2\exp(-(u-\mu_{\text{eff}})^2 / 2\sigma^2) - 1$ with 9-tap Gaussian smoothing (`0.45` blend) to suppress Nyquist aliasing.
- **$A_1$ (Membrane)**: Tracks the spatial gradient magnitude $\|\nabla A_0\|$ and boundary band $4A_0(1-A_0)$, shielding cytoplasm from ambient toxin loss while increasing metabolic maintenance cost.
- **$A_2$ (Organelles)**: Differentiates inside dense cytoplasm regions (`u_cyto` near `org_mu`), enhancing emergent motility and bioluminescent core intensity.

## 2. Coherent Ecological Resource Accounting
- **Absorption**: Scales with local `EcoField` nutrient density, membrane surface (`membrane_mass`), and heritable `nutrient_sensitivity`, depleting the local field.
- **Maintenance**: Scales with cytoplasm mass, membrane mass, kinetic swimming expenditure ($v_x^2 + v_y^2$), and universe metabolic multiplier.
- **Predation**: Inflicts real cellular damage (`LeniaBody::apply_damage`) on prey, transferring energy strictly bounded by prey biomass lost and current prey energy.
- **Decomposition**: Dying organisms return `(0.25 + 1.9 * mass)` directly to the local `EcoField` nutrient channel.

## 3. Honest Current Limitations
- Localized `40×40×3` Lenia stepping currently runs on **CPU via Rayon** rather than in a batched WGSL compute shader; scaling beyond ~320 simultaneous organisms awaits GPU-resident body stepping (planned for v0.4.0).
- `MorphGenome::crossover` is implemented and unit-tested, while live simulation reproduction currently uses asexual mitotic division with mutation; sexual recombination on contact is planned.
- No claim is made of open-ended biological evolution or general intelligence; `NeuralBrain` is a compact 6-unit recurrent controller subject to local selection pressures.

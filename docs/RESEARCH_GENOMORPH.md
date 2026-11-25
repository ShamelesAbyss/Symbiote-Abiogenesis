# Project GENOMORPH — Artificial-Life Research & Architectural Synthesis

This document records the scientific literature and open artificial-life research that inspired **Symbiote: Abiogenesis v0.4.0 (Project GENOMORPH)**, along with our original architectural decisions. No proprietary source code, sprites, or interface assets were copied from any external project.

## 1. Research Inspirations

### The Bibites: Digital Life (Léo Caussan)
- **0.6.0 Organs & Science**: Demonstrated the evolutionary value of **physical organ trade-offs** (WAGG weighted apportionment), where armor, stomach capacity, throat/mouth geometry, and reproductive organs compete for finite body space and metabolic upkeep.
- **BIOME Modular Genetics** *(publicly documented research direction)*: Moving beyond fixed trait vectors toward modular, tree/network-structured genetic architectures where sensors, internal organs, and actuators can evolve new topologies.
- **Grounded Sensorimotor Loops & Digestion**: Input neurons read local world geometry and internal state (energy ratio, stomach fullness, damage, nearby food/meat, pheromones), while digestion converts stomach contents gradually into energy at a metabolic cost rather than instantaneously.

### Karl Sims (1994) — Evolved Virtual Creatures
- Encoded creature morphology and neural control as **directed graphs** supporting structural duplication, deletion, symmetry, and co-evolution of body plans and locomotion strategies.

### NEAT, HyperNEAT & CPPNs (Stanley & Miikkulainen)
- **NEAT (NeuroEvolution of Augmenting Topologies)**: Starts with minimal neural topologies, augments them via edge-splitting (`AddNode`) and `AddConnection`, and uses **innovation numbers** to align homologous structures during crossover.
- **Developmental Encodings & CPPNs**: Compact genetic recipes that express bilateral symmetry, axial segmentation, and repeating organic motifs without encoding every pixel directly.

### Lenia (Bert Chan, 2018) & MAP-Elites (Mouret & Clune, 2015)
- **Continuous Cellular Automata (Lenia)**: Smooth continuous cellular fields producing organic membranes and self-organizing internal structures (MIT attribution preserved in `THIRD_PARTY_LICENSES.md`).
- **MAP-Elites Quality-Diversity**: Preserves diverse high-performing morphological and behavioral niches rather than collapsing evolution into a single global optimum.

## 2. Original GENOMORPH Architecture in Abiogenesis

Abiogenesis synthesizes these principles into a **hybrid continuous-cellular + articulated-anatomical** organism representation:
1. **`GenomorphGenome` (`src/convergence/genomorph.rs`)**: A hierarchical tree of `AnatomyGene`s with stable homology `gene_id`s, supporting bilateral symmetry, symmetry-breaking, `DuplicateGene`, `DeleteGene`, `AddInnovation`, `SpecializeOrgan`, and homology-matched crossover.
2. **Artificial Embryogenesis (`LifeStage`)**: Organisms grow from `Embryo (0.25)` → `Juvenile` → `Adult (1.0)` → `Elder`, with childhood nutrition shaping adult scale.
3. **Functional Organ Trade-offs (`AnatomyPhenotype`)**: Every organ directly determines physics and ecology — `PropulsionFin`/`UndulatingTail` drive thrust & drag, `FeedingJaw` governs directional bite power & range, `FilterMembrane` governs ambient nutrient absorption, `SensoryAntenna` governs directional detection range, `ArmorCarapace` mitigates incoming bite damage at a drag/mass penalty, and `DigestiveVacuole` expands stomach capacity.
4. **NEAT-Inspired `NeuralGraph` (`src/convergence/neural_graph.rs`)**: 10 grounded sensory inputs → evolvable hidden nodes with innovation-tracked synapses (`Tanh`, `Sigmoid`, `Sin`, `Relu`) → 4 motor/physiological outputs (`thrust`, `turn`, `bite`, `secrete`).
5. **Continuous Lenia Coupling**: Every organism retains its `40×40×3` continuous Lenia cellular body seamlessly integrated with its articulated bioluminescent anatomy in the WGSL observatory.

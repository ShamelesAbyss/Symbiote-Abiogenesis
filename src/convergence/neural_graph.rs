//! NEAT-Inspired Evolvable Neural Graph (`NeuralGraph`).
//!
//! Replaces the fixed 6->6->3 network with a genetically encoded neural topology
//! supporting innovation-tracked structural mutations (`AddConnection`, `AddNode`,
//! `ToggleConnection`, `MutateWeight`, `MutateActivation`) and homologous crossover.
//!
//! 10 Grounded Sensory Inputs:
//!   0: local_nutrient       1: nutrient_grad_x     2: nutrient_grad_y
//!   3: local_toxin          4: local_pheromone     5: nearest_org_angle (sin)
//!   6: nearest_org_closeness 7: nearest_org_rel_size 8: energy_norm
//!   9: stomach_fill
//!
//! 4 Functional Motor / Physiological Outputs:
//!   0: propulsion_thrust    1: turn_torque         2: want_to_bite
//!   3: chemical_secrete

use rand::{rngs::StdRng, Rng};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub const NEURAL_INPUTS: usize = 10;
pub const NEURAL_OUTPUTS: usize = 4;
pub const MAX_HIDDEN_NODES: usize = 12;
pub const MAX_CONNECTIONS: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeKind {
    Input,
    Hidden,
    Output,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Activation {
    Tanh,
    Sigmoid,
    Sin,
    Relu,
}

impl Activation {
    pub fn apply(self, x: f32) -> f32 {
        match self {
            Self::Tanh => x.tanh(),
            Self::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Self::Sin => x.sin(),
            Self::Relu => x.max(0.0).min(2.0),
        }
    }

    pub fn random(rng: &mut StdRng) -> Self {
        match rng.gen_range(0..4) {
            0 => Self::Tanh,
            1 => Self::Sigmoid,
            2 => Self::Sin,
            _ => Self::Relu,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct NodeGene {
    pub id: u32,
    pub kind: NodeKind,
    pub activation: Activation,
    pub bias: f32,
    pub value: f32,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct ConnGene {
    pub innovation: u64,
    pub from: u32,
    pub to: u32,
    pub weight: f32,
    pub enabled: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NeuralGraph {
    pub nodes: Vec<NodeGene>,
    pub connections: Vec<ConnGene>,
    pub next_node_id: u32,
    pub next_innovation: u64,
    pub last_out: [f32; NEURAL_OUTPUTS],
}

impl NeuralGraph {
    /// Minimal topology (useful for Genesis Zero — organisms evolve complexity from scratch).
    pub fn minimal(rng: &mut StdRng) -> Self {
        let mut nodes = Vec::with_capacity(NEURAL_INPUTS + NEURAL_OUTPUTS + 4);
        for i in 0..NEURAL_INPUTS as u32 {
            nodes.push(NodeGene {
                id: i,
                kind: NodeKind::Input,
                activation: Activation::Tanh,
                bias: 0.0,
                value: 0.0,
            });
        }
        for o in 0..NEURAL_OUTPUTS as u32 {
            nodes.push(NodeGene {
                id: NEURAL_INPUTS as u32 + o,
                kind: NodeKind::Output,
                activation: Activation::Tanh,
                bias: rng.gen_range(-0.15..0.15),
                value: 0.0,
            });
        }

        let mut connections = Vec::new();
        let mut next_innov = 1u64;
        let mut add = |from: u32, to: u32, w: f32| {
            connections.push(ConnGene {
                innovation: next_innov,
                from,
                to,
                weight: w,
                enabled: true,
            });
            next_innov += 1;
        };

        // Minimal chemotactic reflexes: nutrient gradient -> thrust & turn
        add(0, 10, rng.gen_range(0.3..0.8)); // nutrient -> thrust
        add(1, 10, rng.gen_range(0.4..0.9)); // grad_x -> thrust
        add(2, 11, rng.gen_range(-0.8..0.8)); // grad_y -> turn
        add(5, 11, rng.gen_range(-0.6..0.6)); // nearest org angle -> turn
        add(6, 12, rng.gen_range(-0.4..0.7)); // closeness -> bite
        add(8, 13, rng.gen_range(-0.5..0.5)); // energy -> secrete

        Self {
            nodes,
            connections,
            next_node_id: (NEURAL_INPUTS + NEURAL_OUTPUTS) as u32,
            next_innovation: next_innov,
            last_out: [0.0; NEURAL_OUTPUTS],
        }
    }

    /// Richer founding neural graph with 2 initial hidden interneurons.
    pub fn random(rng: &mut StdRng) -> Self {
        let mut g = Self::minimal(rng);
        for _ in 0..2 {
            g.add_hidden_node_mutation(rng);
        }
        for _ in 0..4 {
            g.add_connection_mutation(rng);
        }
        g
    }

    /// Evaluate the recurrent neural graph for one tick.
    pub fn step(&mut self, inputs: [f32; NEURAL_INPUTS]) -> [f32; NEURAL_OUTPUTS] {
        for (i, &val) in inputs.iter().enumerate() {
            if let Some(n) = self.nodes.get_mut(i) {
                n.value = val.clamp(-3.0, 3.0);
            }
        }

        // Snapshot previous node activations for clean synchronous recurrent update
        let prev_vals: HashMap<u32, f32> = self.nodes.iter().map(|n| (n.id, n.value)).collect();
        let mut accum: HashMap<u32, f32> = HashMap::with_capacity(self.nodes.len());

        for c in &self.connections {
            if !c.enabled {
                continue;
            }
            if let Some(&src_v) = prev_vals.get(&c.from) {
                *accum.entry(c.to).or_insert(0.0) += src_v * c.weight;
            }
        }

        for n in &mut self.nodes {
            if n.kind != NodeKind::Input {
                let sum = n.bias + accum.get(&n.id).copied().unwrap_or(0.0);
                n.value = n.activation.apply(sum);
            }
        }

        let mut out = [0.0f32; NEURAL_OUTPUTS];
        for o in 0..NEURAL_OUTPUTS {
            if let Some(n) = self.nodes.get(NEURAL_INPUTS + o) {
                out[o] = n.value.clamp(-1.0, 1.0);
            }
        }
        self.last_out = out;
        out
    }

    pub fn mutate(&self, rng: &mut StdRng, rate: f32) -> Self {
        let mut child = self.clone();
        for n in &mut child.nodes {
            n.value = 0.0;
        }
        let r = rate.clamp(0.01, 0.35);

        // Weight & bias drift
        for n in &mut child.nodes {
            if n.kind != NodeKind::Input && rng.gen_bool(0.30) {
                n.bias = (n.bias + rng.gen_range(-r..r) * 1.4).clamp(-2.5, 2.5);
            }
            if n.kind == NodeKind::Hidden && rng.gen_bool((r * 0.4).min(0.12) as f64) {
                n.activation = Activation::random(rng);
            }
        }
        for c in &mut child.connections {
            if rng.gen_bool(0.35) {
                c.weight = (c.weight + rng.gen_range(-r..r) * 1.8).clamp(-3.0, 3.0);
            }
            if rng.gen_bool((r * 0.25).min(0.08) as f64) {
                c.enabled = !c.enabled;
            }
        }

        // Structural mutations: AddConnection & AddNode
        if rng.gen_bool((r * 1.4).min(0.24) as f64) {
            child.add_connection_mutation(rng);
        }
        if rng.gen_bool((r * 0.7).min(0.12) as f64) {
            child.add_hidden_node_mutation(rng);
        }

        child
    }

    pub fn add_connection_mutation(&mut self, rng: &mut StdRng) {
        if self.connections.len() >= MAX_CONNECTIONS || self.nodes.len() < 2 {
            return;
        }
        let from_n = self.nodes[rng.gen_range(0..self.nodes.len())].id;
        let targets: Vec<u32> = self
            .nodes
            .iter()
            .filter(|n| n.kind != NodeKind::Input)
            .map(|n| n.id)
            .collect();
        if targets.is_empty() {
            return;
        }
        let to_n = targets[rng.gen_range(0..targets.len())];
        if self
            .connections
            .iter()
            .any(|c| c.from == from_n && c.to == to_n)
        {
            return;
        }
        let innov = self.next_innovation;
        self.next_innovation += 1;
        self.connections.push(ConnGene {
            innovation: innov,
            from: from_n,
            to: to_n,
            weight: rng.gen_range(-1.2..1.2),
            enabled: true,
        });
    }

    pub fn add_hidden_node_mutation(&mut self, rng: &mut StdRng) {
        let hidden_count = self
            .nodes
            .iter()
            .filter(|n| n.kind == NodeKind::Hidden)
            .count();
        if hidden_count >= MAX_HIDDEN_NODES
            || self.connections.is_empty()
            || self.connections.len() + 2 > MAX_CONNECTIONS
        {
            return;
        }
        let idx = rng.gen_range(0..self.connections.len());
        let old = self.connections[idx];
        if !old.enabled {
            return;
        }
        self.connections[idx].enabled = false;

        let nid = self.next_node_id;
        self.next_node_id += 1;
        self.nodes.push(NodeGene {
            id: nid,
            kind: NodeKind::Hidden,
            activation: Activation::random(rng),
            bias: rng.gen_range(-0.2..0.2),
            value: 0.0,
        });

        let i1 = self.next_innovation;
        let i2 = i1 + 1;
        self.next_innovation += 2;
        self.connections.push(ConnGene {
            innovation: i1,
            from: old.from,
            to: nid,
            weight: 1.0,
            enabled: true,
        });
        self.connections.push(ConnGene {
            innovation: i2,
            from: nid,
            to: old.to,
            weight: old.weight,
            enabled: true,
        });
    }

    pub fn hidden_count(&self) -> usize {
        self.nodes
            .iter()
            .filter(|n| n.kind == NodeKind::Hidden)
            .count()
    }

    pub fn active_connection_count(&self) -> usize {
        self.connections.iter().filter(|c| c.enabled).count()
    }
}

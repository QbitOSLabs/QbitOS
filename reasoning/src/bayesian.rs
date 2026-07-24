//! Bayesian network with variable elimination inference.

use ahash::AHashMap;
use serde::{Deserialize, Serialize};

/// A random variable in the Bayesian network.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct RandomVariable {
    /// Variable name.
    pub name: String,
    /// Possible values.
    pub values: Vec<String>,
}

/// A conditional probability table: P(X | parents).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConditionalProbability {
    /// The variable this CPT is for.
    pub variable: String,
    /// Parent variable names.
    pub parents: Vec<String>,
    /// Probability entries: (parent_values, variable_value) → probability.
    /// Key format: "parent1=val1,parent2=val2|var=val"
    pub entries: AHashMap<String, f64>,
}

impl ConditionalProbability {
    /// Look up P(var=val | parents=parent_vals).
    #[must_use]
    pub fn probability(&self, var_value: &str, parent_values: &AHashMap<String, String>) -> f64 {
        let key = self.make_key(var_value, parent_values);
        *self.entries.get(&key).unwrap_or(&0.0)
    }

    fn make_key(&self, var_value: &str, parent_values: &AHashMap<String, String>) -> String {
        let parent_part: Vec<String> = self.parents.iter()
            .map(|p| format!("{p}={}", parent_values.get(p).map_or("?", |v| v)))
            .collect();
        format!("{}|{var_value}", parent_part.join(","))
    }
}

/// A Bayesian network.
pub struct BayesianNetwork {
    /// All variables.
    variables: Vec<RandomVariable>,
    /// CPTs indexed by variable name.
    cpts: AHashMap<String, ConditionalProbability>,
}

/// A query result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    /// The queried variable.
    pub variable: String,
    /// Probability distribution: value → probability.
    pub distribution: AHashMap<String, f64>,
}

impl BayesianNetwork {
    /// Create a new empty network.
    #[must_use]
    pub fn new() -> Self {
        Self { variables: Vec::new(), cpts: AHashMap::new() }
    }

    /// Add a variable.
    pub fn add_variable(&mut self, var: RandomVariable) {
        self.variables.push(var);
    }

    /// Add a CPT.
    pub fn add_cpt(&mut self, cpt: ConditionalProbability) {
        self.cpts.insert(cpt.variable.clone(), cpt);
    }

    /// Query P(variable | evidence) using variable elimination.
    ///
    /// `evidence` maps variable names to their observed values.
    #[must_use]
    pub fn query(&self, variable: &str, evidence: &AHashMap<String, String>) -> QueryResult {
        let var = self.variables.iter().find(|v| v.name == variable);
        if var.is_none() {
            return QueryResult {
                variable: variable.to_string(),
                distribution: AHashMap::new(),
            };
        }
        let var = var.unwrap();

        // For each value of the query variable, compute the joint probability
        let mut distribution = AHashMap::new();
        let mut total = 0.0_f64;

        for value in &var.values {
            let mut full_evidence = evidence.clone();
            full_evidence.insert(variable.to_string(), value.clone());
            let prob = self.joint_probability(&full_evidence);
            distribution.insert(value.clone(), prob);
            total += prob;
        }

        // Normalize
        if total > 0.0 {
            for prob in distribution.values_mut() {
                *prob /= total;
            }
        }

        QueryResult {
            variable: variable.to_string(),
            distribution,
        }
    }

    /// Compute the joint probability of a complete assignment.
    fn joint_probability(&self, assignment: &AHashMap<String, String>) -> f64 {
        let mut prob = 1.0_f64;

        for var in &self.variables {
            if let Some(cpt) = self.cpts.get(&var.name) {
                let var_value = assignment.get(&var.name).map_or("?", |v| v);
                let parent_values: AHashMap<String, String> = cpt.parents.iter()
                    .map(|p| (p.clone(), assignment.get(p).cloned().unwrap_or_default()))
                    .collect();
                prob *= cpt.probability(var_value, &parent_values);
            }
        }

        prob
    }

    /// Get all variables.
    #[must_use]
    pub fn variables(&self) -> &[RandomVariable] { &self.variables }
}

impl Default for BayesianNetwork {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_bayesian_inference() {
        let mut net = BayesianNetwork::new();

        // Rain → Wet (simple causal model)
        net.add_variable(RandomVariable { name: "rain".into(), values: vec!["yes".into(), "no".into()] });
        net.add_variable(RandomVariable { name: "wet".into(), values: vec!["yes".into(), "no".into()] });

        // P(rain=yes) = 0.3
        net.add_cpt(ConditionalProbability {
            variable: "rain".into(),
            parents: vec![],
            entries: {
                let mut e = AHashMap::new();
                e.insert("|yes".into(), 0.3);
                e.insert("|no".into(), 0.7);
                e
            },
        });

        // P(wet=yes | rain=yes) = 0.9, P(wet=yes | rain=no) = 0.1
        net.add_cpt(ConditionalProbability {
            variable: "wet".into(),
            parents: vec!["rain".into()],
            entries: {
                let mut e = AHashMap::new();
                e.insert("rain=yes|yes".into(), 0.9);
                e.insert("rain=yes|no".into(), 0.1);
                e.insert("rain=no|yes".into(), 0.1);
                e.insert("rain=no|no".into(), 0.9);
                e
            },
        });

        // Query P(rain | wet=yes) — should be higher than prior 0.3
        let mut evidence = AHashMap::new();
        evidence.insert("wet".into(), "yes".into());
        let result = net.query("rain", &evidence);

        let p_rain_yes = result.distribution.get("yes").copied().unwrap_or(0.0);
        let p_rain_no = result.distribution.get("no").copied().unwrap_or(0.0);

        assert!((p_rain_yes + p_rain_no - 1.0).abs() < 1e-6, "probabilities should sum to 1");
        assert!(p_rain_yes > 0.3, "posterior P(rain=yes|wet=yes) should be > prior 0.3, got {p_rain_yes}");
    }
}

//! Constraint Satisfaction Problem (CSP) solver.
//!
//! Implements backtracking search with:
//! - AC-3 arc consistency
//! - Forward checking
//! - Minimum remaining values (MRV) heuristic
//! - Least constraining value (LCV) heuristic

use ahash::AHashMap;
use serde::{Deserialize, Serialize};

/// A CSP variable.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Variable {
    /// Variable name.
    pub name: String,
    /// Domain (possible values).
    pub domain: Domain,
}

/// A variable's domain (set of possible values).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Domain {
    /// The values as JSON.
    pub values: Vec<serde_json::Value>,
}

impl Domain {
    /// Create a domain from a list of values.
    #[must_use]
    pub fn new(values: Vec<serde_json::Value>) -> Self { Self { values } }

    /// Create a domain from integers.
    #[must_use]
    pub fn from_ints(range: std::ops::Range<i64>) -> Self {
        Self { values: range.map(|i| serde_json::json!(i)).collect() }
    }

    /// Returns the number of values.
    #[must_use]
    pub fn len(&self) -> usize { self.values.len() }

    /// Returns `true` if the domain is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool { self.values.is_empty() }
}

/// A constraint between variables.
pub struct Constraint {
    /// The variables involved.
    pub variables: Vec<String>,
    /// The constraint function: takes a partial assignment and returns
    /// `true` if the constraint is satisfied.
    pub check: Box<dyn Fn(&AHashMap<String, serde_json::Value>) -> bool + Send + Sync>,
}

impl std::fmt::Debug for Constraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Constraint")
            .field("variables", &self.variables)
            .finish()
    }
}

/// A CSP problem definition.
pub struct CspProblem {
    /// Variables.
    pub variables: Vec<Variable>,
    /// Constraints.
    pub constraints: Vec<Constraint>,
}

/// Result of CSP solving.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CspSolution {
    /// The solution (variable → value mapping), if found.
    pub assignment: Option<AHashMap<String, serde_json::Value>>,
    /// Number of nodes explored.
    pub nodes_explored: usize,
    /// Whether a solution was found.
    pub solved: bool,
}

/// CSP solver with backtracking + AC-3 + forward checking.
pub struct CspSolver {
    /// Enable forward checking.
    pub forward_checking: bool,
    /// Enable AC-3 preprocessing.
    pub ac3: bool,
    /// Maximum nodes to explore.
    pub max_nodes: usize,
}

impl Default for CspSolver {
    fn default() -> Self {
        Self { forward_checking: true, ac3: true, max_nodes: 100_000 }
    }
}

impl CspSolver {
    /// Create a new solver.
    #[must_use]
    pub fn new() -> Self { Self::default() }

    /// Solve a CSP problem.
    #[must_use]
    pub fn solve(&self, problem: &CspProblem) -> CspSolution {
        let var_names: Vec<String> = problem.variables.iter().map(|v| v.name.clone()).collect();
        let mut domains: AHashMap<String, Domain> = problem.variables.iter()
            .map(|v| (v.name.clone(), v.domain.clone()))
            .collect();

        // AC-3 preprocessing
        if self.ac3 {
            self.ac3_preprocess(&mut domains, &problem.constraints, &var_names);
        }

        let mut nodes = 0usize;
        let assignment = self.backtrack(AHashMap::new(), &mut domains, &problem.constraints, &var_names, &mut nodes);

        CspSolution {
            solved: assignment.is_some(),
            assignment,
            nodes_explored: nodes,
        }
    }

    /// Backtracking search with MRV heuristic.
    fn backtrack(
        &self,
        mut assignment: AHashMap<String, serde_json::Value>,
        domains: &mut AHashMap<String, Domain>,
        constraints: &[Constraint],
        var_names: &[String],
        nodes: &mut usize,
    ) -> Option<AHashMap<String, serde_json::Value>> {
        *nodes += 1;
        if *nodes > self.max_nodes { return None; }

        // Check if complete
        if assignment.len() == var_names.len() {
            // Verify all constraints
            if constraints.iter().all(|c| {
                // Only check if all constraint variables are assigned
                if c.variables.iter().all(|v| assignment.contains_key(v)) {
                    (c.check)(&assignment)
                } else {
                    true
                }
            }) {
                return Some(assignment);
            }
            return None;
        }

        // Select unassigned variable with MRV (minimum remaining values)
        let unassigned = self.select_unassigned(var_names, &assignment, domains);
        let var = match unassigned {
            Some(v) => v,
            None => return None,
        };

        let var_domain = domains.get(&var).cloned().unwrap_or_default();

        // Try each value in the domain (LCV ordering)
        let ordered_values = self.order_values(&var_domain, &var, &assignment, constraints);

        for value in ordered_values {
            assignment.insert(var.clone(), value.clone());

            // Check constraints with current assignment (only if all vars in constraint are assigned)
            let consistent = constraints.iter()
                .filter(|c| c.variables.contains(&var))
                .all(|c| {
                    if c.variables.iter().all(|v| assignment.contains_key(v)) {
                        (c.check)(&assignment)
                    } else {
                        true
                    }
                });

            if consistent {
                // Forward checking: remove inconsistent values from neighbors
                let saved_domains = if self.forward_checking {
                    let saved = domains.clone();
                    self.forward_check(&var, &value, domains, constraints);
                    Some(saved)
                } else {
                    None
                };

                // Check for empty domains
                let has_empty = domains.values().any(|d| d.is_empty());
                if !has_empty {
                    if let Some(result) = self.backtrack(assignment.clone(), domains, constraints, var_names, nodes) {
                        return Some(result);
                    }
                }

                // Restore domains
                if let Some(saved) = saved_domains {
                    *domains = saved;
                }
            }

            assignment.remove(&var);
        }

        None
    }

    /// Select unassigned variable using MRV heuristic.
    fn select_unassigned(
        &self,
        var_names: &[String],
        assignment: &AHashMap<String, serde_json::Value>,
        domains: &AHashMap<String, Domain>,
    ) -> Option<String> {
        var_names.iter()
            .filter(|v| !assignment.contains_key(*v))
            .min_by_key(|v| domains.get(*v).map_or(usize::MAX, |d| d.len()))
            .cloned()
    }

    /// Order values using LCV (least constraining value) heuristic.
    fn order_values(
        &self,
        domain: &Domain,
        var: &str,
        assignment: &AHashMap<String, serde_json::Value>,
        constraints: &[Constraint],
    ) -> Vec<serde_json::Value> {
        let mut scored: Vec<(serde_json::Value, usize)> = domain.values.iter()
            .map(|value| {
                let mut test_assignment = assignment.clone();
                test_assignment.insert(var.to_string(), value.clone());
                // Count how many constraints remain satisfiable
                let consistent_count = constraints.iter()
                    .filter(|c| c.variables.contains(&var.to_string()))
                    .filter(|c| (c.check)(&test_assignment))
                    .count();
                (value.clone(), consistent_count)
            })
            .collect();
        // Sort by most consistent (LCV: least constraining = most remaining options)
        scored.sort_by(|a, b| b.1.cmp(&a.1));
        scored.into_iter().map(|(v, _)| v).collect()
    }

    /// Forward checking: remove values from neighboring variables' domains
    /// that are inconsistent with the current assignment.
    fn forward_check(
        &self,
        var: &str,
        value: &serde_json::Value,
        domains: &mut AHashMap<String, Domain>,
        constraints: &[Constraint],
    ) {
        for constraint in constraints {
            if constraint.variables.contains(&var.to_string()) {
                for other_var in &constraint.variables {
                    if other_var != var && !domains.get(other_var).map_or(true, |d| d.is_empty()) {
                        let other_domain = domains.get(other_var).cloned().unwrap_or_default();
                        let filtered: Vec<serde_json::Value> = other_domain.values.iter()
                            .filter(|v| {
                                let mut test = AHashMap::new();
                                test.insert(var.to_string(), value.clone());
                                test.insert(other_var.clone(), (*v).clone());
                                (constraint.check)(&test)
                            })
                            .cloned()
                            .collect();
                        domains.insert(other_var.clone(), Domain { values: filtered });
                    }
                }
            }
        }
    }

    /// AC-3 arc consistency algorithm.
    fn ac3_preprocess(
        &self,
        domains: &mut AHashMap<String, Domain>,
        constraints: &[Constraint],
        _var_names: &[String],
    ) {
        // Build arc queue
        let mut queue: Vec<(String, String)> = Vec::new();
        for constraint in constraints {
            for i in 0..constraint.variables.len() {
                for j in 0..constraint.variables.len() {
                    if i != j {
                        queue.push((constraint.variables[i].clone(), constraint.variables[j].clone()));
                    }
                }
            }
        }

        while let Some((xi, xj)) = queue.pop() {
            if self.revise(&xi, &xj, domains, constraints) {
                if domains.get(&xi).map_or(true, |d| d.is_empty()) {
                    return; // Domain wipeout — no solution
                }
                // Add neighbors back to queue
                for constraint in constraints {
                    if constraint.variables.contains(&xi) {
                        for other in &constraint.variables {
                            if other != &xi && other != &xj {
                                queue.push((other.clone(), xi.clone()));
                            }
                        }
                    }
                }
            }
        }
    }

    /// Revise arc (xi, xj): remove values from xi's domain that have
    /// no support in xj's domain.
    fn revise(
        &self,
        xi: &str,
        xj: &str,
        domains: &mut AHashMap<String, Domain>,
        constraints: &[Constraint],
    ) -> bool {
        let xi_domain = domains.get(xi).cloned().unwrap_or_default();
        let xj_domain = domains.get(xj).cloned().unwrap_or_default();

        let mut revised = false;
        let filtered: Vec<serde_json::Value> = xi_domain.values.iter()
            .filter(|vi| {
                // Check if there exists a vj in xj's domain such that
                // the constraint is satisfied
                xj_domain.values.iter().any(|vj| {
                    let mut test = AHashMap::new();
                    test.insert(xi.to_string(), (*vi).clone());
                    test.insert(xj.to_string(), vj.clone());
                    constraints.iter()
                        .filter(|c| c.variables.contains(&xi.to_string()) && c.variables.contains(&xj.to_string()))
                        .all(|c| (c.check)(&test))
                })
            })
            .cloned()
            .collect();

        if filtered.len() != xi_domain.values.len() {
            revised = true;
            domains.insert(xi.to_string(), Domain { values: filtered });
        }

        revised
    }
}

impl Default for Domain {
    fn default() -> Self { Self { values: Vec::new() } }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solve_simple_csp() {
        // x in {1,2,3}, y in {1,2,3}, x != y, x < y
        let problem = CspProblem {
            variables: vec![
                Variable { name: "x".into(), domain: Domain::from_ints(1..4) },
                Variable { name: "y".into(), domain: Domain::from_ints(1..4) },
            ],
            constraints: vec![
                Constraint {
                    variables: vec!["x".into(), "y".into()],
                    check: Box::new(|a| {
                        let x = a.get("x").and_then(|v| v.as_i64()).unwrap_or(0);
                        let y = a.get("y").and_then(|v| v.as_i64()).unwrap_or(0);
                        x != y && x < y
                    }),
                },
            ],
        };
        let solver = CspSolver::new();
        let result = solver.solve(&problem);
        assert!(result.solved);
        let assignment = result.assignment.unwrap();
        let x = assignment["x"].as_i64().unwrap();
        let y = assignment["y"].as_i64().unwrap();
        assert!(x < y);
        assert_ne!(x, y);
    }

    #[test]
    fn solve_n_queens_4() {
        // 4-queens problem
        let n = 4;
        let variables: Vec<Variable> = (0..n)
            .map(|i| Variable { name: format!("q{i}"), domain: Domain::from_ints(0..n as i64) })
            .collect();
        let constraints: Vec<Constraint> = (0..n)
            .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
            .map(|(i, j)| Constraint {
                variables: vec![format!("q{i}"), format!("q{j}")],
                check: Box::new(move |a| {
                    let qi = a.get(&format!("q{i}")).and_then(|v| v.as_i64()).unwrap_or(-1);
                    let qj = a.get(&format!("q{j}")).and_then(|v| v.as_i64()).unwrap_or(-1);
                    qi != qj && (qi - qj).abs() != (i as i64 - j as i64).abs()
                }),
            })
            .collect();

        let problem = CspProblem { variables, constraints };
        let solver = CspSolver::new();
        let result = solver.solve(&problem);
        assert!(result.solved, "4-queens should have a solution");
    }

    #[test]
    fn unsolvable_csp() {
        // x in {1}, y in {1}, x != y — impossible
        let problem = CspProblem {
            variables: vec![
                Variable { name: "x".into(), domain: Domain::new(vec![serde_json::json!(1)]) },
                Variable { name: "y".into(), domain: Domain::new(vec![serde_json::json!(1)]) },
            ],
            constraints: vec![
                Constraint {
                    variables: vec!["x".into(), "y".into()],
                    check: Box::new(|a| {
                        a.get("x") != a.get("y")
                    }),
                },
            ],
        };
        let solver = CspSolver::new();
        let result = solver.solve(&problem);
        assert!(!result.solved);
    }
}

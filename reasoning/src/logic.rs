//! Propositional logic engine with resolution-based theorem proving.

use ahash::AHashSet;
use serde::{Deserialize, Serialize};

/// A logical proposition: either an atom, negation, conjunction, disjunction, or implication.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum Proposition {
    /// An atomic proposition (e.g., "raining").
    Atom(String),
    /// Negation: NOT p.
    Not(Box<Proposition>),
    /// Conjunction: p AND q.
    And(Box<Proposition>, Box<Proposition>),
    /// Disjunction: p OR q.
    Or(Box<Proposition>, Box<Proposition>),
    /// Implication: p → q (equivalent to NOT p OR q).
    Implies(Box<Proposition>, Box<Proposition>),
    /// Biconditional: p ↔ q.
    Iff(Box<Proposition>, Box<Proposition>),
    /// Constant true.
    True,
    /// Constant false.
    False,
}

impl Proposition {
    /// Create an atom.
    pub fn atom(name: impl Into<String>) -> Self { Self::Atom(name.into()) }

    /// Convert to conjunctive normal form (CNF) — a set of clauses.
    ///
    /// Each clause is a set of literals (atoms or negated atoms).
    #[must_use]
    pub fn to_cnf(&self) -> Vec<Vec<Literal>> {
        let nnf = self.to_nnf();
        nnf.distribute().simplify_clauses()
    }

    /// Convert to negation normal form (NNF).
    fn to_nnf(&self) -> Self {
        match self {
            Self::Atom(_) | Self::True | Self::False => self.clone(),
            Self::Not(inner) => {
                match inner.as_ref() {
                    Self::Not(p) => p.to_nnf(), // ~~p = p
                    Self::And(p, q) => Self::Or(
                        Box::new(Self::Not(p.clone()).to_nnf()),
                        Box::new(Self::Not(q.clone()).to_nnf()),
                    ),
                    Self::Or(p, q) => Self::And(
                        Box::new(Self::Not(p.clone()).to_nnf()),
                        Box::new(Self::Not(q.clone()).to_nnf()),
                    ),
                    Self::Implies(p, q) => Self::And(
                        Box::new(p.to_nnf()),
                        Box::new(Self::Not(q.clone()).to_nnf()),
                    ),
                    Self::Iff(p, q) => Self::And(
                        Box::new(Self::Implies(p.clone(), q.clone()).to_nnf()),
                        Box::new(Self::Implies(q.clone(), p.clone()).to_nnf()),
                    ),
                    Self::Atom(_) | Self::True | Self::False => self.clone(),
                }
            }
            Self::And(p, q) => Self::And(
                Box::new(p.to_nnf()),
                Box::new(q.to_nnf()),
            ),
            Self::Or(p, q) => Self::Or(
                Box::new(p.to_nnf()),
                Box::new(q.to_nnf()),
            ),
            Self::Implies(p, q) => Self::Or(
                Box::new(Self::Not(p.clone()).to_nnf()),
                Box::new(q.to_nnf()),
            ),
            Self::Iff(p, q) => Self::And(
                Box::new(Self::Implies(p.clone(), q.clone()).to_nnf()),
                Box::new(Self::Implies(q.clone(), p.clone()).to_nnf()),
            ),
        }
    }

    /// Distribute OR over AND to get CNF.
    fn distribute(&self) -> Self {
        match self {
            Self::And(p, q) => Self::And(
                Box::new(p.distribute()),
                Box::new(q.distribute()),
            ),
            Self::Or(p, q) => {
                let pd = p.distribute();
                let qd = q.distribute();
                match (&pd, &qd) {
                    (Self::And(a, b), _) => Self::And(
                        Box::new(Self::Or(a.clone(), Box::new(qd.clone())).distribute()),
                        Box::new(Self::Or(b.clone(), Box::new(qd)).distribute()),
                    ),
                    (_, Self::And(a, b)) => Self::And(
                        Box::new(Self::Or(Box::new(pd.clone()), a.clone()).distribute()),
                        Box::new(Self::Or(Box::new(pd), b.clone()).distribute()),
                    ),
                    _ => Self::Or(Box::new(pd), Box::new(qd)),
                }
            }
            _ => self.clone(),
        }
    }

    /// Convert to clause form (list of clauses, each a list of literals).
    fn simplify_clauses(&self) -> Vec<Vec<Literal>> {
        match self {
            Self::And(p, q) => {
                let mut clauses = p.simplify_clauses();
                clauses.extend(q.simplify_clauses());
                clauses
            }
            Self::Or(_, _) => {
                let mut literals = Vec::new();
                self.collect_literals(&mut literals);
                if literals.is_empty() { vec![vec![]] } else { vec![literals] }
            }
            Self::Atom(name) => vec![vec![Literal::Positive(name.clone())]],
            Self::Not(inner) => {
                if let Self::Atom(name) = inner.as_ref() {
                    vec![vec![Literal::Negative(name.clone())]]
                } else {
                    vec![vec![]]
                }
            }
            Self::True => vec![],
            Self::False => vec![vec![]],
            _ => vec![],
        }
    }

    fn collect_literals(&self, literals: &mut Vec<Literal>) {
        match self {
            Self::Or(p, q) => {
                p.collect_literals(literals);
                q.collect_literals(literals);
            }
            Self::Atom(name) => literals.push(Literal::Positive(name.clone())),
            Self::Not(inner) => {
                if let Self::Atom(name) = inner.as_ref() {
                    literals.push(Literal::Negative(name.clone()));
                }
            }
            _ => {}
        }
    }
}

/// A literal: either a positive or negative atom.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Literal {
    /// Positive literal (e.g., "raining").
    Positive(String),
    /// Negative literal (e.g., "NOT raining").
    Negative(String),
}

impl Literal {
    /// Get the negation of this literal.
    #[must_use]
    pub fn negate(&self) -> Self {
        match self {
            Self::Positive(name) => Self::Negative(name.clone()),
            Self::Negative(name) => Self::Positive(name.clone()),
        }
    }

    /// Get the atom name.
    #[must_use]
    pub fn atom(&self) -> &str {
        match self {
            Self::Positive(name) | Self::Negative(name) => name,
        }
    }
}

/// Result of a proof attempt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofResult {
    /// Whether the proposition is provable.
    pub proven: bool,
    /// Number of resolution steps.
    pub steps: usize,
    /// The clauses used in the proof.
    pub clauses: Vec<Vec<Literal>>,
}

/// Resolution-based theorem prover.
pub struct LogicEngine {
    /// Known facts (clauses).
    clauses: Vec<Vec<Literal>>,
}

impl LogicEngine {
    /// Create a new empty engine.
    #[must_use]
    pub fn new() -> Self {
        Self { clauses: Vec::new() }
    }

    /// Add a known fact (proposition).
    pub fn add_fact(&mut self, prop: Proposition) {
        let clauses = prop.to_cnf();
        self.clauses.extend(clauses);
    }

    /// Try to prove a proposition by resolution.
    ///
    /// Converts the negation of the target to CNF, adds it to the clause set,
    /// and applies resolution until either an empty clause is derived (proof
    /// found) or no new clauses can be generated (proof not found).
    #[must_use]
    pub fn prove(&self, target: &Proposition) -> ProofResult {
        // Negate the target and convert to CNF
        let negated = Proposition::Not(Box::new(target.clone()));
        let mut all_clauses: Vec<Vec<Literal>> = self.clauses.clone();
        all_clauses.extend(negated.to_cnf());

        let mut steps = 0;
        let max_steps = 1000;

        // Resolution loop
        let mut new_clauses = Vec::new();
        let mut clause_set: AHashSet<Vec<Literal>> = all_clauses.iter()
            .map(|c| {
                let mut sorted = c.clone();
                sorted.sort_by(|a, b| a.atom().cmp(b.atom()));
                sorted
            })
            .collect();

        loop {
            if steps >= max_steps {
                break;
            }

            let clauses_vec: Vec<Vec<Literal>> = clause_set.iter().cloned().collect();
            let mut found_empty = false;
            new_clauses.clear();

            for i in 0..clauses_vec.len() {
                for j in (i + 1)..clauses_vec.len() {
                    let resolvents = resolve(&clauses_vec[i], &clauses_vec[j]);
                    for resolvent in &resolvents {
                        if resolvent.is_empty() {
                            found_empty = true;
                        }
                        let mut sorted = resolvent.clone();
                        sorted.sort_by(|a, b| a.atom().cmp(b.atom()));
                        if !clause_set.contains(&sorted) {
                            new_clauses.push(sorted);
                        }
                    }
                }
                steps += 1;
            }

            if found_empty {
                return ProofResult { proven: true, steps, clauses: clauses_vec };
            }

            if new_clauses.is_empty() {
                break;
            }

            for c in new_clauses.drain(..) {
                clause_set.insert(c);
            }
        }

        ProofResult { proven: false, steps, clauses: clause_set.into_iter().collect() }
    }
}

impl Default for LogicEngine {
    fn default() -> Self { Self::new() }
}

/// Resolve two clauses: find complementary literals and produce resolvents.
fn resolve(c1: &[Literal], c2: &[Literal]) -> Vec<Vec<Literal>> {
    let mut resolvents = Vec::new();

    for (i, l1) in c1.iter().enumerate() {
        for (j, l2) in c2.iter().enumerate() {
            if l1.atom() == l2.atom() && l1 != l2 {
                // Complementary literals found — resolve
                let mut resolvent: Vec<Literal> = Vec::new();
                resolvent.extend(c1[..i].iter().chain(&c1[i + 1..]).cloned());
                resolvent.extend(c2[..j].iter().chain(&c2[j + 1..]).cloned());
                // Remove duplicates
                resolvent.sort_by(|a, b| a.atom().cmp(b.atom()));
                resolvent.dedup_by(|a, b| a == b);
                resolvents.push(resolvent);
            }
        }
    }

    resolvents
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prove_modus_ponens() {
        let mut engine = LogicEngine::new();
        // If raining then wet. Raining. Therefore wet.
        engine.add_fact(Proposition::Implies(
            Box::new(Proposition::atom("raining")),
            Box::new(Proposition::atom("wet")),
        ));
        engine.add_fact(Proposition::atom("raining"));

        let result = engine.prove(&Proposition::atom("wet"));
        assert!(result.proven, "should prove 'wet' from modus ponens");
    }

    #[test]
    fn cannot_prove_unrelated() {
        let mut engine = LogicEngine::new();
        engine.add_fact(Proposition::atom("raining"));
        let result = engine.prove(&Proposition::atom("sunny"));
        assert!(!result.proven, "should not prove 'sunny' from 'raining'");
    }

    #[test]
    fn prove_chain() {
        let mut engine = LogicEngine::new();
        // a → b, b → c, a. Therefore c.
        engine.add_fact(Proposition::Implies(
            Box::new(Proposition::atom("a")),
            Box::new(Proposition::atom("b")),
        ));
        engine.add_fact(Proposition::Implies(
            Box::new(Proposition::atom("b")),
            Box::new(Proposition::atom("c")),
        ));
        engine.add_fact(Proposition::atom("a"));

        let result = engine.prove(&Proposition::atom("c"));
        assert!(result.proven, "should prove 'c' from chain a→b→c with a");
    }

    #[test]
    fn cnf_conversion() {
        let prop = Proposition::Implies(
            Box::new(Proposition::atom("p")),
            Box::new(Proposition::atom("q")),
        );
        let cnf = prop.to_cnf();
        // p → q should become (NOT p OR q) in CNF
        assert!(!cnf.is_empty());
        let clause = &cnf[0];
        assert_eq!(clause.len(), 2);
        assert!(clause.iter().any(|l| matches!(l, Literal::Negative(name) if name == "p")));
        assert!(clause.iter().any(|l| matches!(l, Literal::Positive(name) if name == "q")));
    }
}

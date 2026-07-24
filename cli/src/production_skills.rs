//! Production skill registry — pre-built skills registered at App startup.
//!
//! These skills are available out-of-the-box without requiring dataset mining.
//! Each skill is composed of Read / Infer / Tool / Verify / Return steps.

use qbitos_skills::registry::{Skill, SkillStep, StepKind};
use qbitos_skills::SkillModule;
use std::sync::Arc;

/// Register all production skills into the given SkillModule.
/// Returns the number of skills registered.
#[must_use]
pub fn register_production_skills(module: &Arc<SkillModule>) -> usize {
    let skills = production_skills();
    for skill in skills {
        module.registry.register(skill);
    }
    module.registry.len()
}

/// Returns the full list of production skills.
#[must_use]
pub fn production_skills() -> Vec<Skill> {
    vec![
        skill_summarize(),
        skill_translate(),
        skill_code_review(),
        skill_qa(),
        skill_entity_extraction(),
        skill_sentiment_analysis(),
        skill_text_classification(),
        skill_keyword_extraction(),
        skill_rewrite(),
        skill_fact_check(),
        skill_outline(),
        skill_compare(),
    ]
}

// ─── Individual skill definitions ────────────────────────────────────────────

fn skill_summarize() -> Skill {
    let mut s = Skill::new("summarize", "Summarize a piece of text into a concise summary");
    s.add_step(SkillStep::new(
        "read_input",
        StepKind::Read,
        serde_json::json!({"text": "${input.text}"}),
    ));
    s.add_step(SkillStep::new(
        "infer_summary",
        StepKind::Infer,
        serde_json::json!({"prompt": "Summarize the following text in 2-3 sentences: ${text}"}),
    ));
    s.add_step(SkillStep::new(
        "verify_length",
        StepKind::Verify,
        serde_json::json!({"max_chars": 500}),
    ));
    s.add_step(SkillStep::new(
        "return_summary",
        StepKind::Return,
        serde_json::json!({"value": "${infer_summary.text}"}),
    ));
    s.success_criteria.push("summary_is_concise".into());
    s.success_criteria.push("summary_captures_key_points".into());
    s.add_tag("text");
    s.add_tag("nlp");
    s
}

fn skill_translate() -> Skill {
    let mut s = Skill::new("translate", "Translate text from one language to another");
    s.add_step(SkillStep::new(
        "read_input",
        StepKind::Read,
        serde_json::json!({"text": "${input.text}", "target_language": "${input.target_language}"}),
    ));
    s.add_step(SkillStep::new(
        "infer_translation",
        StepKind::Infer,
        serde_json::json!({"prompt": "Translate the following text to ${target_language}: ${text}"}),
    ));
    s.add_step(SkillStep::new(
        "verify_translation",
        StepKind::Verify,
        serde_json::json!({"check": "meaning_preserved"}),
    ));
    s.add_step(SkillStep::new(
        "return_translation",
        StepKind::Return,
        serde_json::json!({"value": "${infer_translation.text}"}),
    ));
    s.success_criteria.push("translation_is_grammatically_correct".into());
    s.add_tag("text");
    s.add_tag("nlp");
    s.add_tag("multilingual");
    s
}

fn skill_code_review() -> Skill {
    let mut s = Skill::new("code_review", "Review code for bugs, style issues, and improvements");
    s.add_step(SkillStep::new(
        "read_code",
        StepKind::Read,
        serde_json::json!({"code": "${input.code}", "language": "${input.language}"}),
    ));
    s.add_step(SkillStep::new(
        "infer_review",
        StepKind::Infer,
        serde_json::json!({"prompt": "Review this ${language} code for bugs, style issues, and improvements. Code: ${code}"}),
    ));
    s.add_step(SkillStep::new(
        "return_review",
        StepKind::Return,
        serde_json::json!({"value": "${infer_review.text}"}),
    ));
    s.success_criteria.push("identifies_at_least_one_issue".into());
    s.add_tag("code");
    s.add_tag("development");
    s
}

fn skill_qa() -> Skill {
    let mut s = Skill::new("question_answer", "Answer a question based on provided context or general knowledge");
    s.add_step(SkillStep::new(
        "read_question",
        StepKind::Read,
        serde_json::json!({"question": "${input.question}", "context": "${input.context}"}),
    ));
    s.add_step(SkillStep::new(
        "infer_answer",
        StepKind::Infer,
        serde_json::json!({"prompt": "Answer this question: ${question}. Context: ${context}"}),
    ));
    s.add_step(SkillStep::new(
        "verify_relevance",
        StepKind::Verify,
        serde_json::json!({"check": "answer_addresses_question"}),
    ));
    s.add_step(SkillStep::new(
        "return_answer",
        StepKind::Return,
        serde_json::json!({"value": "${infer_answer.text}"}),
    ));
    s.success_criteria.push("answer_is_relevant".into());
    s.add_tag("qa");
    s.add_tag("reasoning");
    s
}

fn skill_entity_extraction() -> Skill {
    let mut s = Skill::new("entity_extraction", "Extract named entities (people, places, organizations) from text");
    s.add_step(SkillStep::new(
        "read_text",
        StepKind::Read,
        serde_json::json!({"text": "${input.text}"}),
    ));
    s.add_step(SkillStep::new(
        "infer_entities",
        StepKind::Infer,
        serde_json::json!({"prompt": "Extract all named entities (people, places, organizations, dates) from this text as a JSON list: ${text}"}),
    ));
    s.add_step(SkillStep::new(
        "return_entities",
        StepKind::Return,
        serde_json::json!({"value": "${infer_entities.text}"}),
    ));
    s.success_criteria.push("entities_are_valid_json".into());
    s.add_tag("nlp");
    s.add_tag("extraction");
    s
}

fn skill_sentiment_analysis() -> Skill {
    let mut s = Skill::new("sentiment_analysis", "Analyze the sentiment of a text (positive, negative, neutral)");
    s.add_step(SkillStep::new(
        "read_text",
        StepKind::Read,
        serde_json::json!({"text": "${input.text}"}),
    ));
    s.add_step(SkillStep::new(
        "infer_sentiment",
        StepKind::Infer,
        serde_json::json!({"prompt": "Analyze the sentiment of this text. Classify as positive, negative, or neutral and explain why: ${text}"}),
    ));
    s.add_step(SkillStep::new(
        "return_sentiment",
        StepKind::Return,
        serde_json::json!({"value": "${infer_sentiment.text}"}),
    ));
    s.success_criteria.push("sentiment_is_classified".into());
    s.add_tag("nlp");
    s.add_tag("sentiment");
    s
}

fn skill_text_classification() -> Skill {
    let mut s = Skill::new("text_classification", "Classify text into predefined categories");
    s.add_step(SkillStep::new(
        "read_input",
        StepKind::Read,
        serde_json::json!({"text": "${input.text}", "categories": "${input.categories}"}),
    ));
    s.add_step(SkillStep::new(
        "infer_category",
        StepKind::Infer,
        serde_json::json!({"prompt": "Classify this text into one of these categories: ${categories}. Text: ${text}"}),
    ));
    s.add_step(SkillStep::new(
        "return_category",
        StepKind::Return,
        serde_json::json!({"value": "${infer_category.text}"}),
    ));
    s.success_criteria.push("category_is_valid".into());
    s.add_tag("nlp");
    s.add_tag("classification");
    s
}

fn skill_keyword_extraction() -> Skill {
    let mut s = Skill::new("keyword_extraction", "Extract key phrases and keywords from text");
    s.add_step(SkillStep::new(
        "read_text",
        StepKind::Read,
        serde_json::json!({"text": "${input.text}"}),
    ));
    s.add_step(SkillStep::new(
        "infer_keywords",
        StepKind::Infer,
        serde_json::json!({"prompt": "Extract the top 5-10 keywords and key phrases from this text as a JSON list: ${text}"}),
    ));
    s.add_step(SkillStep::new(
        "return_keywords",
        StepKind::Return,
        serde_json::json!({"value": "${infer_keywords.text}"}),
    ));
    s.success_criteria.push("keywords_are_relevant".into());
    s.add_tag("nlp");
    s.add_tag("extraction");
    s
}

fn skill_rewrite() -> Skill {
    let mut s = Skill::new("rewrite", "Rewrite text in a different style or tone");
    s.add_step(SkillStep::new(
        "read_input",
        StepKind::Read,
        serde_json::json!({"text": "${input.text}", "style": "${input.style}"}),
    ));
    s.add_step(SkillStep::new(
        "infer_rewrite",
        StepKind::Infer,
        serde_json::json!({"prompt": "Rewrite this text in a ${style} style: ${text}"}),
    ));
    s.add_step(SkillStep::new(
        "verify_meaning",
        StepKind::Verify,
        serde_json::json!({"check": "meaning_preserved"}),
    ));
    s.add_step(SkillStep::new(
        "return_rewrite",
        StepKind::Return,
        serde_json::json!({"value": "${infer_rewrite.text}"}),
    ));
    s.success_criteria.push("style_is_applied".into());
    s.add_tag("text");
    s.add_tag("nlp");
    s
}

fn skill_fact_check() -> Skill {
    let mut s = Skill::new("fact_check", "Check a claim for factual accuracy using available tools");
    s.add_step(SkillStep::new(
        "read_claim",
        StepKind::Read,
        serde_json::json!({"claim": "${input.claim}"}),
    ));
    s.add_step(SkillStep::new(
        "search_evidence",
        StepKind::Tool,
        serde_json::json!({"tool": "http", "args": {"url": "https://api.example.com/search", "query": "${claim}"}}),
    ));
    s.add_step(SkillStep::new(
        "infer_verification",
        StepKind::Infer,
        serde_json::json!({"prompt": "Based on the search results, verify this claim: ${claim}. Results: ${search_evidence.output}"}),
    ));
    s.add_step(SkillStep::new(
        "return_verdict",
        StepKind::Return,
        serde_json::json!({"value": "${infer_verification.text}"}),
    ));
    s.success_criteria.push("verdict_is_supported_by_evidence".into());
    s.add_tag("verification");
    s.add_tag("reasoning");
    s
}

fn skill_outline() -> Skill {
    let mut s = Skill::new("outline", "Generate a structured outline from a topic or document");
    s.add_step(SkillStep::new(
        "read_input",
        StepKind::Read,
        serde_json::json!({"topic": "${input.topic}", "depth": "${input.depth}"}),
    ));
    s.add_step(SkillStep::new(
        "infer_outline",
        StepKind::Infer,
        serde_json::json!({"prompt": "Create a structured outline for: ${topic}. Use ${depth} levels of depth."}),
    ));
    s.add_step(SkillStep::new(
        "return_outline",
        StepKind::Return,
        serde_json::json!({"value": "${infer_outline.text}"}),
    ));
    s.success_criteria.push("outline_is_hierarchical".into());
    s.add_tag("text");
    s.add_tag("structure");
    s
}

fn skill_compare() -> Skill {
    let mut s = Skill::new("compare", "Compare two or more items and highlight differences");
    s.add_step(SkillStep::new(
        "read_items",
        StepKind::Read,
        serde_json::json!({"items": "${input.items}"}),
    ));
    s.add_step(SkillStep::new(
        "infer_comparison",
        StepKind::Infer,
        serde_json::json!({"prompt": "Compare these items and highlight key differences and similarities: ${items}"}),
    ));
    s.add_step(SkillStep::new(
        "return_comparison",
        StepKind::Return,
        serde_json::json!({"value": "${infer_comparison.text}"}),
    ));
    s.success_criteria.push("comparison_covers_all_items".into());
    s.add_tag("analysis");
    s.add_tag("reasoning");
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn production_skills_are_valid() {
        let skills = production_skills();
        assert!(skills.len() >= 10, "expected at least 10 production skills, got {}", skills.len());
        for skill in &skills {
            assert!(!skill.name.is_empty(), "skill name must not be empty");
            assert!(!skill.description.is_empty(), "skill description must not be empty");
            assert!(!skill.steps.is_empty(), "skill '{}' must have at least one step", skill.name);
            assert!(!skill.success_criteria.is_empty(), "skill '{}' must have success criteria", skill.name);
        }
    }

    #[test]
    fn skills_have_unique_names() {
        let skills = production_skills();
        let mut names: Vec<&str> = skills.iter().map(|s| s.name.as_str()).collect();
        names.sort();
        for w in names.windows(2) {
            assert_ne!(w[0], w[1], "duplicate skill name: {}", w[0]);
        }
    }

    #[test]
    fn all_step_kinds_are_used() {
        let skills = production_skills();
        let mut has_read = false;
        let mut has_infer = false;
        let mut has_tool = false;
        let mut has_verify = false;
        let mut has_return = false;
        for skill in &skills {
            for step in &skill.steps {
                match step.kind {
                    StepKind::Read => has_read = true,
                    StepKind::Infer => has_infer = true,
                    StepKind::Tool => has_tool = true,
                    StepKind::Verify => has_verify = true,
                    StepKind::Return => has_return = true,
                }
            }
        }
        assert!(has_read, "no Read steps found");
        assert!(has_infer, "no Infer steps found");
        assert!(has_tool, "no Tool steps found");
        assert!(has_verify, "no Verify steps found");
        assert!(has_return, "no Return steps found");
    }

    #[test]
    fn register_into_skill_module() {
        use qbitos_memory::MemoryFabric;
        let memory = Arc::new(MemoryFabric::with_deterministic_embeddings(384));
        let module = Arc::new(SkillModule::with_noop_backends(memory.episodic.clone()));
        let count = register_production_skills(&module);
        assert!(count >= 10, "expected at least 10 skills registered, got {}", count);
        // Verify skills are retrievable by name.
        assert!(module.registry.get_by_name("summarize").is_some());
        assert!(module.registry.get_by_name("translate").is_some());
        assert!(module.registry.get_by_name("code_review").is_some());
    }
}

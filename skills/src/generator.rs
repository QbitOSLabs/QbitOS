//! Skill generator — synthesizes new skills from templates.

use crate::registry::{Skill, SkillStep, StepKind};

/// A generated skill (before registration).
#[derive(Debug, Clone)]
pub struct GeneratedSkill {
    /// The generated skill.
    pub skill: Skill,
    /// The template name used to generate it.
    pub template: String,
}

/// The skill generator.
pub struct SkillGenerator {
    templates: ahash::AHashMap<String, Skill>,
}

impl SkillGenerator {
    /// Create a new generator with default templates.
    #[must_use]
    pub fn new() -> Self {
        let mut templates = ahash::AHashMap::new();
        // Q&A template
        let mut qa = Skill::new("qa", "Answer a question");
        qa.add_step(SkillStep::new("read", StepKind::Read, serde_json::json!({})));
        qa.add_step(SkillStep::new("infer", StepKind::Infer, serde_json::json!({"prompt": "${input}"})));
        qa.add_step(SkillStep::new("return", StepKind::Return, serde_json::json!({})));
        templates.insert("qa".into(), qa);

        // Tool-call template
        let mut tool = Skill::new("tool-call", "Call a tool");
        tool.add_step(SkillStep::new("read", StepKind::Read, serde_json::json!({})));
        tool.add_step(SkillStep::new("tool", StepKind::Tool, serde_json::json!({"tool": "${tool_name}", "args": "${input}"})));
        tool.add_step(SkillStep::new("return", StepKind::Return, serde_json::json!({})));
        templates.insert("tool-call".into(), tool);

        // Verify template
        let mut verify = Skill::new("verify", "Verify a claim");
        verify.add_step(SkillStep::new("read", StepKind::Read, serde_json::json!({})));
        verify.add_step(SkillStep::new("infer", StepKind::Infer, serde_json::json!({"prompt": "Verify: ${input}"})));
        verify.add_step(SkillStep::new("verify", StepKind::Verify, serde_json::json!({})));
        verify.add_step(SkillStep::new("return", StepKind::Return, serde_json::json!({})));
        templates.insert("verify".into(), verify);

        Self { templates }
    }

    /// Add a template.
    pub fn add_template(&mut self, name: impl Into<String>, skill: Skill) {
        self.templates.insert(name.into(), skill);
    }

    /// Generate a skill from a template.
    #[must_use]
    pub fn generate(&self, template: &str, name: impl Into<String>, description: impl Into<String>) -> Option<GeneratedSkill> {
        let template_skill = self.templates.get(template)?;
        let mut skill = template_skill.clone();
        skill.id = None;
        skill.name = name.into();
        skill.description = description.into();
        skill.execution_count = 0;
        skill.success_count = 0;
        skill.total_latency_ms = 0;
        skill.last_executed_at = None;
        Some(GeneratedSkill {
            skill,
            template: template.to_string(),
        })
    }

    /// List available template names.
    #[must_use]
    pub fn templates(&self) -> Vec<String> {
        self.templates.keys().cloned().collect()
    }

    /// Sanitize a skill name (lowercase, replace spaces with hyphens).
    #[must_use]
    pub fn sanitize_name(name: &str) -> String {
        name.to_lowercase()
            .chars()
            .map(|c| if c.is_alphanumeric() || c == '-' { c } else if c == ' ' { '-' } else { '-' })
            .collect::<String>()
            .trim_matches('-')
            .to_string()
    }
}

impl Default for SkillGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_from_qa_template() {
        let generator = SkillGenerator::new();
        let result = generator.generate("qa", "my-qa", "custom QA skill");
        assert!(result.is_some());
        let generated = result.unwrap();
        assert_eq!(generated.skill.name, "my-qa");
        assert_eq!(generated.skill.steps.len(), 3);
        assert_eq!(generated.template, "qa");
    }

    #[test]
    fn generate_from_unknown_template_returns_none() {
        let generator = SkillGenerator::new();
        assert!(generator.generate("nonexistent", "test", "test").is_none());
    }

    #[test]
    fn sanitize_name_replaces_spaces() {
        assert_eq!(SkillGenerator::sanitize_name("My Cool Skill"), "my-cool-skill");
        assert_eq!(SkillGenerator::sanitize_name("test!@#$"), "test");
    }

    #[test]
    fn list_templates_includes_defaults() {
        let generator = SkillGenerator::new();
        let templates = generator.templates();
        assert!(templates.contains(&"qa".into()));
        assert!(templates.contains(&"tool-call".into()));
        assert!(templates.contains(&"verify".into()));
    }
}

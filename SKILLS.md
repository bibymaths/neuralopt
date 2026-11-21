# NeuralOpt skills and Claude Code workflow

Claude Code automatically discovers `.claude/skills/<name>/SKILL.md`; this **top-level index** helps humans navigate the kit.

| Skill command | Use when |
| --- | --- |
| `/neuralopt-audit` | Inspect current code, experiment artifacts and baseline without making changes. |
| `/curvature-theory` | Derive normalized secant geometry; compare papers; test identifiability and invariances. |
| `/trajectory-curvature` | Implement a small, masked historical `(s,y)` representation in the current JAX model. |
| `/geometry-meta-learning` | Train learned preconditioners/directions using fair outcome objectives. |
| `/geometry-benchmarks` | Preregister, run, and audit matched-budget v0.6+ experiments. |
| `/certificate-guard` | Audit mathematical certification and independent replay if related code changes. |
| `/quality-gate` | Run regression, compatibility, documentation, packaging and honest report gates. |

Optional specialized read-only agents: `curvature-reviewer`, `experiment-auditor` (under `.claude/agents/`). They are additional **Claude Code subagents**, not alternative coding-agent products.

**Superpowers plugin:** optional, external and user-installed. Use `/plugin install superpowers@claude-plugins-official` in Claude Code. Its planning and TDD skills complement, rather than duplicate, the NeuralOpt domain skills. Do not vendor the plugin.

Recommended sequence: audit → curvature theory → design approval → narrow trajectory implementation → train/ablate → benchmark → certification regression → quality gate.

# Codex CLI in GRPO-MA

Project research workflows are installed in .agents/skills/ from the ARIS Codex-native package. Use /skills to browse or mention $research-review, $experiment-plan, $paper-write, or another skill in your prompt. Text following the skill name supplies its arguments.

Read the selected skill and its referenced files as needed. Shared guidance is in .agents/skills/shared-references/; helper tools are in .aris/tools/. Resolve project-relative paths from the repository root. Check required MCP services, credentials, and local tools before running a workflow; report missing capabilities without fabricating results.

The base Codex reviewer is same-family: preserve provisional review status where the skill requires it. This installation uses copies, so changes to the upstream ARIS checkout do not automatically update these skills.

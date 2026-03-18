# nanobot Skills

This directory contains built-in skills that extend nanobot's capabilities.

## Skill Format

Each skill is a directory containing a `SKILL.md` file with:
- YAML frontmatter (name, description, metadata)
- Markdown instructions for the agent

## Attribution

These skills are adapted from [OpenClaw](https://github.com/openclaw/openclaw)'s skill system.
The skill format and metadata structure follow OpenClaw's conventions to maintain compatibility.

## Available Skills

| Skill | Description |
|-------|-------------|
| `github` | Interact with GitHub using the `gh` CLI |
| `weather` | Get weather info using wttr.in and Open-Meteo |
| `summarize` | Summarize URLs, files, and YouTube videos |
| `tmux` | Remote-control tmux sessions |
| `clawhub` | Search and install skills from ClawHub registry |
| `skill-creator` | Create new skills |
| `bestseller-novel` | 爆款网文创作：按起点/番茄等平台畅销套路设计并写作 |
| `bestseller-comic` | 小说→爆款动态漫画：分镜脚本与平台爆款套路（B站/快看/抖音等） |
| `bestseller-evaluator` | 爆款评估：五种评估师视角 × 八维评分卡，量化小说/动态漫画的爆款潜力 |
| `ppt-expert` | PPT制作专家：两阶段生成分镜脚本、AI绘图prompt和口播稿 |
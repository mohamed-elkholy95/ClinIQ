# ClinIQ Continuous Build Task

You are building ClinIQ — a production clinical NLP platform at ~/projects/ClinIQ

PRD: ~/projects/ClinIQ/cliniq-16-month-roadmap.md
IGNORE ALL TIMELINES. Build everything as fast as possible.

## Pre-flight
1. Check git lock: fuser -v .git/index.lock 2>&1 || echo "NO_LOCK"
2. If locked or recent commit (<5min), STOP and report "Build session active. Skipping."
3. Read PRD and check what exists: find . -name "*.py" -o -name "*.ts" | head -50
4. git log --oneline -5 to see progress

## Build Strategy
Build the PRD fully from Phase 0 through Phase 6. After PRD is complete:
- Research online for clinical NLP best practices, healthcare AI patterns, production ML architecture
- Add enhancements: better error handling, more tests, performance optimizations, new features
- Update the PRD file (cliniq-16-month-roadmap.md) at end of each session with what was completed

## Commit Requirements
- Minimum 20 commits, target 30 per session
- One logical unit per commit
- After each session: update the PRD file to mark completed sections

## Commit Rules (ZERO TOLERANCE)
- NEVER add Co-Authored-By for any AI tool
- NEVER add AI-assisted or Generated-by attribution
- Use conventional commits: feat(), fix(), docs(), test(), chore(), ci(), refactor()
- Professional educational tone

## After Building
1. Push: git push origin main
2. Update PRD with completed work
3. Report commits and next target

Format: "🏗️ ClinIQ — [section] | X commits | Next: [target]"

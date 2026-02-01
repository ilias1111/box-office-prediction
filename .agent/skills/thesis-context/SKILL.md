---
name: Thesis Context
description: Access and reference the LaTeX thesis documentation for the box office prediction project
---

# Thesis Context Skill

## Overview

This skill provides access to the LaTeX thesis that documents the box office prediction project. The thesis is located in a separate repository but is directly related to this codebase.

## Thesis Location

**Primary Path**: `/Users/iliasx/Documents/GitHub/box-office-prediction-thesis`

The thesis is structured as follows:
- **Main file**: `/Users/iliasx/Documents/GitHub/box-office-prediction-thesis/latex/main.tex`
- **Chapters**: `/Users/iliasx/Documents/GitHub/box-office-prediction-thesis/latex/chapters/`
- **Introduction**: `/Users/iliasx/Documents/GitHub/box-office-prediction-thesis/latex/intro/`
- **Images**: `/Users/iliasx/Documents/GitHub/box-office-prediction-thesis/latex/images/`
- **Library/References**: `/Users/iliasx/Documents/GitHub/box-office-prediction-thesis/latex/library/`

## When to Use This Skill

Activate this skill when the user mentions:
- "thesis" or "the thesis"
- "check the thesis text"
- "look at the documentation"
- "what does the thesis say about..."
- "according to the thesis..."
- "thesis chapter" or "thesis section"
- References to theoretical foundations, implementation details, or methodology that might be documented in academic writing
- Requests to verify consistency between code and documentation

## Key Thesis Chapters

Based on the thesis structure, you should be aware of these typical chapters:

1. **Theoretical Foundations** (`chapter2_theoretical_foundations.tex`) - Background on tools, methods, and theory
2. **Implementation** (`chapter3_implementation.tex`) - Technical details about the codebase architecture
3. **Results/Evaluation** - Experimental results and analysis
4. **Discussion** - Ethical concerns, limitations, future work

## Usage Instructions

### 1. Initial Exploration

When first accessing the thesis for a query:

```bash
# List available chapters
ls -la /Users/iliasx/Documents/GitHub/box-office-prediction-thesis/latex/chapters/

# View main thesis structure
view_file /Users/iliasx/Documents/GitHub/box-office-prediction-thesis/latex/main.tex
```

### 2. Finding Relevant Content

Use `grep_search` to find specific topics across thesis files:

```bash
# Example: Search for "feature engineering" in thesis
grep_search /Users/iliasx/Documents/GitHub/box-office-prediction-thesis/latex/ "feature engineering"
```

### 3. Cross-Referencing Code and Thesis

When the user asks about implementation details:
1. First check the relevant code in this repository
2. Then check the thesis implementation chapter for documentation
3. Verify consistency between code and documentation
4. Report any discrepancies to the user

### 4. Updating Thesis Content

If the user asks to update thesis content:
1. Locate the relevant `.tex` file in the thesis repository
2. Make edits using standard LaTeX formatting
3. Ensure citations and references are properly formatted
4. Verify that changes align with the actual codebase

## Best Practices

1. **Always use absolute paths** when referencing thesis files
2. **Check both repositories** when answering questions about implementation
3. **Maintain consistency** between code comments and thesis documentation
4. **Respect LaTeX formatting** when editing thesis files
5. **Cite specific sections** when referencing thesis content (e.g., "According to Section 3.2 of the thesis...")

## Common Queries

### "What does the thesis say about [topic]?"

1. Search for the topic across thesis files
2. Identify relevant sections
3. Summarize findings with specific references

### "Update the thesis to reflect [code change]"

1. Identify which chapter/section needs updating
2. Locate the specific `.tex` file
3. Make appropriate edits
4. Ensure technical accuracy

### "Is the code consistent with the thesis?"

1. Review the relevant code section
2. Check corresponding thesis documentation
3. Report any inconsistencies
4. Suggest updates if needed

## Integration with Project

The thesis documents:
- **Feature Engineering**: KPIs, holidays, socioeconomic factors
- **Hyperparameter Configuration**: Grid search strategies
- **Wikipedia Scraping**: Async architecture, financial data parsing
- **Model Architecture**: ML pipeline, evaluation metrics
- **Results**: Performance analysis, comparisons

When working on these areas in the code, always cross-reference the thesis for context and documentation.

## Example Workflow

```
User: "Check the thesis to see how we handle holiday features"

Steps:
1. grep_search /Users/iliasx/Documents/GitHub/box-office-prediction-thesis/latex/ "holiday"
2. Identify relevant chapter (likely Implementation or Theoretical Foundations)
3. view_file the specific chapter file
4. Extract and summarize the holiday feature handling approach
5. Cross-reference with actual code in /Users/iliasx/Documents/GitHub/box-office-prediction/Code/
6. Report findings to user
```

## Notes

- The thesis is written in LaTeX and compiled to PDF
- Main compilation output: `/Users/iliasx/Documents/GitHub/box-office-prediction-thesis/latex/main.pdf`
- The thesis uses the `upatras-thesis.sty` style file for formatting
- Images and figures are stored in the `images/` subdirectory

# Interactive Population Policy Explorer

This tool allows you to dynamically explore population policy research directions using AI-Scientist, rather than being limited to the hard-coded `prompt.json`.

## Features

🎯 **Dynamic Prompt Generation**: Customize research focus at runtime  
🤖 **Interactive Model Selection**: Choose from popular AI models  
💡 **Interactive Ideas Configuration**: Set number of experiments interactively  
📋 **Predefined Policy Directions**: Choose from common research themes  
✏️ **Custom Directions**: Specify your own unique policy focus  
🔧 **Original Prompt Option**: Proceed with existing prompt.json unchanged  
🚀 **Automated Execution**: Runs AI-Scientist experiments automatically  
📊 **Plot Generation**: Creates comprehensive analysis visualizations  
🔄 **Automatic Cleanup**: Safely modifies and automatically restores original prompts  

## Quick Start

```bash
# From AI-Scientist root directory
python interactive_population_policy.py
```

## Usage Examples

### Basic Usage
```bash
python interactive_population_policy.py
```

### Custom Model and Multiple Ideas
```bash
python interactive_population_policy.py --model gpt-4o --num-ideas 2
```


## Example Policy Directions

### 1. Use Original Prompt (No Changes)
*Proceed with existing prompt.json without modifications*
- Keep the current research directions unchanged
- Use this for running experiments with the default template focus

### 2. Crisis Solutions with Education & Technology
*"crisis solutions emphasizing education and technology incentives"*
- Focus on innovative educational policies and technology-driven solutions

### 3. Rural Revitalization & Regional Development  
*"regional development focused on rural revitalization and sustainable communities"*
- Emphasize policies to revive rural areas and reduce urban concentration

### 4. Immigration-Centric Integration
*"immigration-focused policies with comprehensive integration and support systems"*
- Explore immigration as a primary solution with strong integration measures

### 5. Work-Life Balance for Families
*"work-life balance improvements specifically targeting young families and parents"*
- Focus on policies that help families balance career and child-rearing

### 6. Economic Incentives & Housing
*"economic and housing affordability policies to reduce barriers to family formation"*
- Address financial barriers that prevent young people from having children

### 7. Comprehensive Multi-Generational Approach
*"comprehensive multi-generational policies addressing elderly care, youth support, and intergenerational solidarity"*
- Holistic approach considering all age groups and generational interactions

### 8. Custom Direction
*Enter your own policy research direction*

## How It Works

1. **Model Selection**: Choose from popular AI models or use default
2. **Ideas Configuration**: Set number of experiments to run (1-10)
3. **Policy Selection**: Choose from predefined directions, custom focus, or original prompt
4. **Dynamic Prompt**: System generates custom `prompt.json` based on your input (if changed)
5. **Experiment Execution**: Runs AI-Scientist with your specified configuration
6. **Analysis Generation**: Creates comprehensive plots and visualizations
7. **Cleanup**: Automatically restores original `prompt.json` (if modified)

## Example Session

```
🧬 INTERACTIVE POPULATION POLICY EXPLORER
============================================================

🤖 MODEL SELECTION
============================================================
1. Claude 3.5 Sonnet (default)
   ID: claude-3-5-sonnet-20241022
   Description: Anthropic's latest model (recommended)

2. GPT-4o
   ID: gpt-4o
   Description: OpenAI's latest multimodal model
...

Select model (1-6, or press Enter for default): 

✓ Using default model: claude-3-5-sonnet-20241022

💡 NUMBER OF IDEAS
============================================================
How many different research ideas should AI-Scientist generate?

• 1 idea: Quick single experiment (~15-30 min)
• 2-3 ideas: Compare different approaches (~30-90 min)
• 4-5 ideas: Comprehensive exploration (~2-4 hours)

Enter number of ideas (1-10, or press Enter for default 1): 2

✓ Selected: 2 idea(s)

📋 PREDEFINED POLICY DIRECTIONS
============================================================

1. Use Original Prompt (No Changes)
   Direction: Proceed with existing prompt.json without modifications

2. Crisis Solutions with Education & Technology
   Direction: "crisis solutions emphasizing education and technology incentives"
...

Select policy direction (1-9): 2

✓ Selected: Crisis Solutions with Education & Technology
Direction: "crisis solutions emphasizing education and technology incentives"

Proceed with this selection? (y/n): y

🔄 Creating dynamic prompt...
✓ Saved dynamic prompt to templates/population_policy/prompt.json

🎯 Ready to run AI-Scientist experiment:
  Model: claude-3-5-sonnet-20241022
  Template: population_policy
  Number of ideas: 2
  Focus: crisis solutions emphasizing education and technology incentives

Proceed with experiment? (y/n): y

🚀 Running AI-Scientist...
```

## Generated Outputs

After running, you'll find:
- **Experimental results** in `templates/population_policy/run_*/`
- **Analysis plots** in `templates/population_policy/`:
  - `population_projections.png`
  - `policy_analysis.png`
  - `step_by_step_changes.png`
  - `demographic_transition.png`
  - `uncertainty_analysis.png`

## Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `claude-3-5-sonnet-20241022` | Default model (can be changed interactively) |
| `--num-ideas` | `1` | Default number of ideas (can be changed interactively) |

**Note**: All options can be overridden during the interactive session. Command line arguments only set the initial defaults.

## Benefits Over Static Prompts

### ✅ **Dynamic Research Directions**
- Explore different policy emphases without editing files
- Quickly test various research hypotheses
- Adapt to current policy debates and priorities

### ✅ **User-Friendly Interface**
- No need to understand prompt engineering
- Guided selection with clear descriptions
- Safe backup and restore functionality

### ✅ **Reproducible Experiments**
- Clear documentation of research focus
- Consistent experimental framework
- Easy to share and replicate studies

### ✅ **Rapid Prototyping**
- Test multiple policy directions quickly
- Compare different research approaches
- Iterate on policy ideas efficiently

## Notes

- Original `prompt.json` is automatically backed up as `prompt_original.json`
- Script must be run from AI-Scientist root directory
- Generated plots require successful experimental runs with valid data
- All experiments use the same mathematical framework but with different policy focus

This tool transforms AI-Scientist from a static research tool into an interactive policy exploration platform!
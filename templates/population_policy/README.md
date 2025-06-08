# Population Policy Analysis Template

## Overview

This template extends AI-Scientist to tackle Japan's population decline challenge through computational demography and policy simulation. It provides a comprehensive framework for modeling population dynamics and evaluating policy interventions.

## Features

### 🧠 **Mathematical Population Models**
- **Classical Leslie Matrix**: Age-structured population dynamics with fertility and survival rates
- **Cohort-Component Method**: Detailed demographic accounting (births, deaths, migrations)
- **Stochastic Simulation**: Monte Carlo population projection with uncertainty quantification
- **Policy Integration**: 10-dimensional policy feature space affecting demographic parameters

### 📊 **Data Integration**
- **Real UN Data**: Uses actual UN World Population Prospects 2024 data for Japan
- **Comprehensive Demographics**: Population by 5-year age groups, fertility rates, life expectancy
- **Current Baselines**: 2024-2025 data providing realistic starting points for projections
- **Synthetic Fallback**: Generates realistic data when real data unavailable

### 🎯 **Policy Simulation**
- **Intervention Scenarios**: Pro-natalist, immigration, work-life balance, regional development policies
- **Multi-objective Optimization**: Balance population growth, economic sustainability, and social equity
- **Uncertainty Quantification**: Monte Carlo simulation for risk assessment

### 📈 **Comprehensive Visualization**
- Population projections and demographic transitions
- Policy effectiveness analysis and feature importance
- Regional migration patterns and age structure evolution
- Uncertainty bounds and scenario comparisons

## Quick Start

### 1. Install Dependencies
Dependencies are managed at the AI-Scientist top level. From the main AI-Scientist directory:
```bash
pip install -r requirements.txt
```

### 2. Configure Environment (Optional)
Copy the example environment file and customize settings:
```bash
cp .env.example .env
# Edit .env with your preferences (API keys, directories, defaults)
```

### 3. Run Basic Experiment
```bash
python experiment.py --model_type leslie_matrix --projection_years 30
```

### 4. Generate Visualizations
```bash
python plot.py --plot_type all
```

## Mathematical Model Architecture

### Leslie Matrix Model
Classical demographic projection matrices:
- Age-specific fertility rates (births by maternal age)
- Age-specific survival rates (1 - mortality rates)
- Migration effects applied as additive terms
- Policy parameters directly modify demographic rates

### Cohort-Component Method
Standard demographic accounting approach:
- Separate tracking of births, deaths, and migrations
- Age progression through cohort advancement
- Policy effects applied to component rates
- More detailed than Leslie matrix for demographic analysis

### Stochastic Leslie Model
Monte Carlo population simulation:
- Random variation in demographic parameters
- Multiple simulation runs for uncertainty bounds
- Policy risk assessment through probabilistic projections
- Confidence intervals for population forecasts

## Policy Feature Space

The framework models 10 key policy dimensions:

1. **Child Allowance Rate** - Monthly support per child
2. **Parental Leave Weeks** - Paid leave duration
3. **Childcare Availability** - Spots per 1000 children
4. **Immigration Rate** - Annual immigration quota
5. **Regional Development** - Rural investment programs
6. **Elder Care Capacity** - Facilities per 1000 elderly
7. **Tax Incentives** - Family tax benefits
8. **Work-Life Balance** - Hour regulations and flexibility
9. **Housing Affordability** - Cost subsidies and support
10. **Education Investment** - System quality and access

## Experiment Ideas

The template includes 3 high-impact research directions focused on Japan's demographic crisis:

- **Multi-Objective Crisis Optimization** - Systematic exploration of policy combinations to achieve population stabilization by 2040 while maintaining economic sustainability
- **Critical Window Interventions** - Analysis of time-sensitive intervention windows to prevent irreversible population collapse, testing emergency vs. delayed response strategies  
- **Cross-Generational Policy Design** - Novel integrated approaches targeting multiple generations simultaneously in Japan's ultra-aging society (30% elderly)

## Data Sources

### Real UN World Population Prospects 2024
- **Japan Population Data**: Complete age structure (21 five-year age groups: 0-4 through 100+)
- **Current Demographics**: 123.8M total population (2024), 30% elderly (65+), 11.2% youth (0-14)
- **Fertility & Mortality**: Total fertility rate 1.22, life expectancy 84.9 years
- **Projection Ready**: 2024-2025 baseline data for mathematical model initialization

### Data Pipeline Features  
- **Automatic Loading**: Reads local UN CSV files from configurable directory
- **Age Group Mapping**: Converts UN 5-year age groups to model-compatible format
- **Demographic Indicators**: Population totals, age structure, fertility rates, life expectancy
- **Quality Validation**: Realistic demographic checks (aging ratio, dependency ratio)
- **Environment Integration**: API keys and data paths configurable via .env file

## Output and Analysis

The framework generates:

### Model Outputs
- Mathematical model parameters and configuration
- Population projection matrices and demographic rates  
- Policy simulation results with uncertainty bounds

### Visualizations
- Population projection charts by scenario
- Demographic transition analysis (dependency ratios, aging)
- Policy feature importance and effectiveness rankings
- Regional migration flow visualizations

### Research Papers
When integrated with AI-Scientist, automatically generates:
- LaTeX formatted academic papers
- Literature review and citation integration
- Experimental methodology and results
- Policy recommendations and implications

## Integration with AI-Scientist

This template is designed to work seamlessly with the AI-Scientist framework:

1. **Idea Generation**: Uses `seed_ideas.json` to bootstrap research directions
2. **Experiment Execution**: `experiment.py` modified by Aider based on generated ideas
3. **Results Analysis**: `plot.py` creates visualizations for paper figures
4. **Paper Writing**: LaTeX template in `latex/` directory for academic output

## Research Impact

This framework enables rapid exploration of:
- Policy intervention effectiveness for demographic challenges
- Economic-demographic feedback mechanisms
- Regional development strategies for population retention
- Cross-national demographic comparison approaches
- Uncertainty quantification for policy risk assessment

The systematic approach allows AI-Scientist to generate dozens of research papers exploring different aspects of population policy, each with rigorous experimental validation and clear policy implications.

## File Structure

```
population_policy/
├── experiment.py          # Main mathematical modeling framework
├── data_pipeline.py       # Real UN data loading and preprocessing
├── plot.py               # Demographic visualization and analysis
├── prompt.json           # AI-Scientist task description
├── seed_ideas.json       # Research idea templates (3 crisis-focused ideas)
├── ideas.json            # Generated ideas (populated by AI-Scientist)
├── notes.txt             # Experiment logging
├── .env.example          # Environment variable template
├── un_population_data/   # Real UN World Population Prospects 2024 data
│   ├── population-by-5-year-age-group.csv
│   ├── total-fertility-rate.csv
│   └── life-expectancy.csv
├── latex/                # LaTeX paper template
│   ├── template.tex
│   └── ...
└── README.md            # This file
```

## Environment Configuration

The template supports configuration via environment variables for flexible deployment:

### API Keys and Data Sources
```bash
# e-Stat API (Japanese government statistics)
ESTAT_API_KEY=your_api_key_here
ESTAT_API_BASE=https://api.e-stat.go.jp/rest/3.0/app/json

# UN Population Data
UN_API_BASE=https://population.un.org/dataportalapi/api/v1
UN_DATA_DIR=un_population_data
```

### Model Defaults
```bash
# Override command-line defaults
DEFAULT_MODEL_TYPE=leslie_matrix
DEFAULT_PROJECTION_YEARS=30
DEFAULT_AGE_GROUPS=21
DEFAULT_POLICY_SENSITIVITY=1.0
```

### Output Settings
```bash
# Directory configuration
CACHE_DIR=data_cache
RESULTS_DIR=results

# Plot configuration
PLOT_DPI=300
PLOT_STYLE=seaborn-v0_8
```

All settings can be overridden by command-line arguments when running experiments.

## Contributing

This template is designed to be extended and modified. Key areas for contribution:
- Additional mathematical demographic models (e.g., multi-regional, economic-demographic coupling)
- New policy intervention types and features
- Enhanced data sources and preprocessing approaches
- Novel visualization and analysis approaches
- Economic and social feedback mechanisms

---

*This template enables AI-Scientist to conduct cutting-edge research on one of Japan's most pressing challenges while advancing the field of computational demography.*
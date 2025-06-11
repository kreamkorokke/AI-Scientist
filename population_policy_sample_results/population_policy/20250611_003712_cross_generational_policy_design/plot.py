import json
import os
import os.path as osp
import pickle

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Set style from environment or default
plot_style = os.getenv("PLOT_STYLE", "seaborn-v0_8")
try:
    plt.style.use(plot_style)
except:
    plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# LOAD FINAL RESULTS:
folders = os.listdir("/")
final_results = {}
all_results = {}
for folder in folders:
    if folder.startswith("run") and osp.isdir(folder):
        try:
            with open(osp.join(folder, "final_info.json"), "r") as f:
                final_results[folder] = json.load(f)
            with open(osp.join(folder, "results.pkl"), "rb") as f:
                all_results[folder] = pickle.load(f)
        except FileNotFoundError:
            print(f"Missing results files in {folder}")
            continue

# CREATE LEGEND -- ADD RUNS HERE THAT WILL BE PLOTTED
labels = {
    "run_1": "Three-Generation Housing",
    "run_2": "Intergenerational Skill Transfer", 
    "run_3": "Economic Partnerships",
    "run_4": "Integrated Cross-Generational Approach"
}

# Color schemes
policy_colors = {
    'baseline': '#1f77b4',
    'three_gen_housing': '#2ecc71',  # Green for housing
    'skill_transfer': '#9b59b6',     # Purple for education/skills
    'economic_partnership': '#e67e22', # Orange for economic
    'integrated_approach': '#e74c3c', # Red for comprehensive
    'comprehensive': '#8c564b',
    'work_life_balance': '#d62728',
    'regional_development': '#9467bd',
}

# Create a programmatic color palette
def generate_color_palette(n):
    cmap = plt.get_cmap("tab20")
    return [mcolors.rgb2hex(cmap(i)) for i in np.linspace(0, 1, n)]

# Get the list of runs and generate the color palette
runs = list(labels.keys())
colors = generate_color_palette(len(runs))

# Plot 1: Population projections by policy scenario
def plot_population_projections():
    """Plot population projections under different policy scenarios."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Process each run
    for i, run in enumerate(runs):
        if run not in all_results:
            continue
            
        results = all_results[run]
        if 'policy_simulations' not in results:
            continue
            
        simulations = results['policy_simulations']
        config = results.get('config', {})
        
        # 1. Total population trajectories
        base_year = 2024
        projection_years = config.get('projection_years', 50)
        
        for scenario_name, data in simulations.items():
            # Handle different data structures from mathematical models
            if 'total_population' in data:
                total_pop = np.array(data['total_population']) / 1000  # Convert to thousands
                is_stochastic = False
            elif 'total_population_mean' in data:  # Stochastic model (old format)
                total_pop = np.array(data['total_population_mean']) / 1000
                is_stochastic = True
            else:
                continue
            
            # Create years array that matches data length
            years = np.arange(base_year, base_year + len(total_pop))
            
            color = policy_colors.get(scenario_name, colors[i % len(colors)])
            
            # Plot main trajectory
            if 'baseline' in scenario_name or 'comprehensive' in scenario_name:
                ax1.plot(years, total_pop, label=f"{labels[run]} - {scenario_name.replace('_', ' ').title()}", 
                        linewidth=3, color=color)
            else:
                ax1.plot(years, total_pop, label=f"{labels[run]} - {scenario_name.replace('_', ' ').title()}", 
                        linewidth=2, alpha=0.8, color=color)
            
            # Add uncertainty bands for stochastic models
            if is_stochastic and 'total_population_p5' in data and 'total_population_p95' in data:
                p5 = np.array(data['total_population_p5']) / 1000
                p95 = np.array(data['total_population_p95']) / 1000
                if len(p5) == len(years) and len(p95) == len(years):
                    ax1.fill_between(years, p5, p95, alpha=0.2, color=color)
            
            # 2. Aging ratio trajectories
            aging_ratio = np.array(data['aging_ratio']) * 100  # Convert to percentage
            if len(aging_ratio) == len(years):
                ax2.plot(years, aging_ratio, label=f"{labels[run]} - {scenario_name.replace('_', ' ').title()}", 
                        linewidth=2, color=color)
    
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Population (thousands)')
    ax1.set_title('Population Projections by Policy Scenario')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Aging Ratio (%)')
    ax2.set_title('Population Aging (65+ %) by Policy Scenario')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # 3. Population change comparison (2050 vs 2024)
    scenario_names = []
    pop_changes = []
    
    for run in runs:
        if run not in all_results:
            continue
        results = all_results[run]
        if 'policy_simulations' not in results:
            continue
        simulations = results['policy_simulations']
        
        for scenario_name, data in simulations.items():
            total_pops = data['total_population']
            if len(total_pops) > 25:  # Ensure we have data for ~2050
                change = ((total_pops[25] - total_pops[0]) / total_pops[0]) * 100
                scenario_names.append(f"{labels[run]} - {scenario_name.replace('_', ' ').title()}")
                pop_changes.append(change)
    
    if scenario_names:
        bar_colors = [policy_colors.get(name.split(' - ')[-1].lower().replace(' ', '_'), 'gray') for name in scenario_names]
        bars = ax3.bar(range(len(scenario_names)), pop_changes, color=bar_colors, alpha=0.8)
        ax3.set_xlabel('Policy Scenario')
        ax3.set_ylabel('Population Change 2024-2050 (%)')
        ax3.set_title('Population Change by 2050')
        ax3.set_xticks(range(len(scenario_names)))
        ax3.set_xticklabels(scenario_names, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add value labels on bars
        for bar, value in zip(bars, pop_changes):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height >= 0 else -1),
                    f'{value:.1f}%', ha='center', va='bottom' if height >= 0 else 'top')
    
    # 4. Age structure evolution (first run with comprehensive scenario)
    age_labels = [f"{i*5}-{i*5+4}" for i in range(20)] + ["100+"]
    for run in runs:
        if run not in all_results:
            continue
        results = all_results[run]
        if 'policy_simulations' not in results:
            continue
        simulations = results['policy_simulations']
        
        if 'comprehensive' in simulations:
            comp_data = simulations['comprehensive']
            trajectory = comp_data['population_trajectory']
            
            # Show age structure at different time points
            time_points = [0, 15, 30]  # 2024, 2039, 2054
            time_labels = ['2024', '2039', '2054']
            
            x = np.arange(len(age_labels))
            width = 0.25
            
            for i, (time_idx, label) in enumerate(zip(time_points, time_labels)):
                if time_idx < len(trajectory):
                    age_dist = trajectory[time_idx] / trajectory[time_idx].sum() * 100
                    ax4.bar(x + i * width, age_dist, width, label=label, alpha=0.8)
            
            ax4.set_xlabel('Age Group')
            ax4.set_ylabel('Population Share (%)')
            ax4.set_title(f'Age Structure Evolution ({labels[run]} - Comprehensive Policy)')
            ax4.set_xticks(x[::2] + width)  # Set every other tick position
            ax4.set_xticklabels(age_labels[::2], rotation=45)  # Show every other label
            ax4.legend()
            ax4.grid(True, alpha=0.3, axis='y')
            break  # Only use first run with comprehensive data
    
    plt.tight_layout()
    plt.savefig("population_projections.png", dpi=300, bbox_inches='tight')
    plt.close()

# Plot 2: Policy feature analysis
def plot_policy_analysis():
    """Analyze and visualize policy features and their impacts."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Process first run with data
    for run in runs:
        if run not in all_results:
            continue
        results = all_results[run]
        if 'policy_simulations' not in results:
            continue
        simulations = results['policy_simulations']
        break
    else:
        print("No policy simulation data found.")
        return
    
    # Extract policy features and outcomes
    policy_names = [
        "Child Allowance", "Parental Leave", "Childcare Avail.", "Immigration", 
        "Regional Dev.", "Elder Care", "Tax Incentives", "Work-Life Balance", 
        "Housing Afford.", "Education Invest."
    ]
    
    scenarios = []
    features_matrix = []
    outcomes = []
    
    for scenario_name, data in simulations.items():
        if 'random' not in scenario_name:  # Exclude random scenarios for clarity
            scenarios.append(scenario_name.replace('_', ' ').title())
            features_matrix.append(data['policy_vector'])
            
            # Calculate outcome: population change by 2050
            total_pops = data['total_population']
            if len(total_pops) > 25:
                outcome = ((total_pops[25] - total_pops[0]) / total_pops[0]) * 100
            else:
                outcome = 0
            outcomes.append(outcome)
    
    features_matrix = np.array(features_matrix)
    
    # 1. Policy features heatmap
    if len(features_matrix) > 0:
        im = ax1.imshow(features_matrix.T, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
        ax1.set_xticks(range(len(scenarios)))
        ax1.set_xticklabels(scenarios, rotation=45, ha='right')
        ax1.set_yticks(range(len(policy_names)))
        ax1.set_yticklabels(policy_names)
        ax1.set_title('Policy Feature Intensities by Scenario')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label('Policy Intensity (0-1)')
        
        # Add text annotations
        for i in range(len(scenarios)):
            for j in range(len(policy_names)):
                text = ax1.text(i, j, f'{features_matrix[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=8)
        
        # 2. Policy effectiveness (outcome vs baseline)
        baseline_outcome = outcomes[0] if scenarios[0] == 'Baseline' else 0
        effectiveness = [outcome - baseline_outcome for outcome in outcomes[1:]]
        scenario_subset = scenarios[1:]
        
        bar_colors = [policy_colors.get(name.lower().replace(' ', '_'), 'gray') for name in scenario_subset]
        bars = ax2.bar(range(len(scenario_subset)), effectiveness, color=bar_colors, alpha=0.8)
        ax2.set_xlabel('Policy Scenario')
        ax2.set_ylabel('Additional Pop. Change vs Baseline (%)')
        ax2.set_title('Policy Effectiveness (vs Baseline)')
        ax2.set_xticks(range(len(scenario_subset)))
        ax2.set_xticklabels(scenario_subset, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add value labels
        for bar, value in zip(bars, effectiveness):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height >= 0 else -0.2),
                    f'{value:.1f}%', ha='center', va='bottom' if height >= 0 else 'top')
        
        # 3. Feature importance (correlation with outcomes)
        correlations = []
        for feature_idx in range(len(policy_names)):
            feature_values = features_matrix[:, feature_idx]
            correlation = np.corrcoef(feature_values, outcomes)[0, 1] if len(feature_values) > 1 else 0
            correlations.append(correlation)
        
        corr_colors = ['green' if corr > 0 else 'red' for corr in correlations]
        bars = ax3.barh(range(len(policy_names)), correlations, color=corr_colors, alpha=0.7)
        ax3.set_yticks(range(len(policy_names)))
        ax3.set_yticklabels(policy_names)
        ax3.set_xlabel('Correlation with Population Outcome')
        ax3.set_title('Policy Feature Importance')
        ax3.grid(True, alpha=0.3, axis='x')
        ax3.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        
        # Add correlation values
        for bar, corr in zip(bars, correlations):
            width = bar.get_width()
            ax3.text(width + (0.02 if width >= 0 else -0.02), bar.get_y() + bar.get_height()/2.,
                    f'{corr:.2f}', ha='left' if width >= 0 else 'right', va='center')
        
        # 4. Scenario comparison radar chart
        # Select top 3 most different scenarios
        if len(scenarios) >= 4:
            selected_scenarios = ['Comprehensive', 'Pro Natalist', 'Immigration Focused']
            selected_indices = [i for i, s in enumerate(scenarios) if s in selected_scenarios]
            
            # Normalize features for radar chart
            angles = np.linspace(0, 2 * np.pi, len(policy_names), endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle
            
            ax4 = plt.subplot(2, 2, 4, projection='polar')
            
            for idx in selected_indices:
                values = features_matrix[idx].tolist()
                values += values[:1]  # Complete the circle
                
                color = policy_colors.get(scenarios[idx].lower().replace(' ', '_'), 'gray')
                ax4.plot(angles, values, 'o-', linewidth=2, label=scenarios[idx], color=color)
                ax4.fill(angles, values, alpha=0.25, color=color)
            
            ax4.set_xticks(angles[:-1])
            ax4.set_xticklabels([name.replace(' ', '\n') for name in policy_names], fontsize=8)
            ax4.set_ylim(0, 1)
            ax4.set_title('Policy Profile Comparison', pad=20)
            ax4.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
            ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig("policy_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

# Plot 3: Step-by-step population changes
def plot_step_by_step_changes():
    """Plot detailed step-by-step population changes over time."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Process each run
    for i, run in enumerate(runs):
        if run not in all_results:
            continue
        results = all_results[run]
        if 'policy_simulations' not in results:
            continue
        simulations = results['policy_simulations']
        break
    else:
        print("No policy simulation data found.")
        return
    
    # 1. Year-over-year population growth rates
    for scenario_name, data in simulations.items():
        if 'random' not in scenario_name and 'population_growth_rates' in data:
            growth_rates = data['population_growth_rates']
            # Create years array that matches growth rates length
            years = np.arange(2025, 2025 + len(growth_rates))  # Growth rates start from year 2
            color = policy_colors.get(scenario_name, 'gray')
            
            ax1.plot(years, growth_rates, label=scenario_name.replace('_', ' ').title(), 
                    linewidth=2, color=color)
    
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Population Growth Rate (%)')
    ax1.set_title('Year-over-Year Population Growth Rates by Policy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # 2. Population composition over time (stacked areas)
    baseline_data = simulations.get('baseline', simulations[list(simulations.keys())[0]])
    
    # Convert to percentages
    total_pop = np.array(baseline_data['total_population'])
    young_pct = np.array(baseline_data['young_population']) / total_pop * 100
    working_pct = np.array(baseline_data['working_age_population']) / total_pop * 100
    elderly_pct = np.array(baseline_data['elderly_population']) / total_pop * 100
    
    # Create years array that matches data length
    years = np.arange(2024, 2024 + len(total_pop))
    
    ax2.fill_between(years, 0, young_pct, alpha=0.7, color='lightblue', label='0-14 years')
    ax2.fill_between(years, young_pct, young_pct + working_pct, alpha=0.7, color='lightgreen', label='15-64 years')
    ax2.fill_between(years, young_pct + working_pct, 100, alpha=0.7, color='lightcoral', label='65+ years')
    
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Population Share (%)')
    ax2.set_title('Population Age Composition Over Time (Baseline)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    # 3. Absolute population changes by age group
    for scenario_name, data in simulations.items():
        if scenario_name == 'baseline':  # Focus on baseline for clarity
            young_pop = np.array(data['young_population'])/1000
            working_pop = np.array(data['working_age_population'])/1000
            elderly_pop = np.array(data['elderly_population'])/1000
            
            # Create years array that matches data length
            years = np.arange(2024, 2024 + len(young_pop))
            
            ax3.plot(years, young_pop, label='0-14 years', 
                    linewidth=2, color='blue', linestyle='-')
            ax3.plot(years, working_pop, label='15-64 years',
                    linewidth=2, color='green', linestyle='-')  
            ax3.plot(years, elderly_pop, label='65+ years',
                    linewidth=2, color='red', linestyle='-')
            break
    
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Population (millions)')
    ax3.set_title('Population by Age Group Over Time (Baseline)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Policy effectiveness over time (cumulative population difference vs baseline)
    if 'baseline' in simulations:
        baseline_pop = np.array(simulations['baseline']['total_population'])
        
        for scenario_name, data in simulations.items():
            if scenario_name != 'baseline' and 'random' not in scenario_name:
                scenario_pop = np.array(data['total_population'])
                # Ensure both arrays have same length
                min_len = min(len(baseline_pop), len(scenario_pop))
                pop_difference = ((scenario_pop[:min_len] - baseline_pop[:min_len]) / baseline_pop[:min_len]) * 100
                color = policy_colors.get(scenario_name, 'gray')
                
                # Create years array that matches data length
                years = np.arange(2024, 2024 + min_len)
                
                ax4.plot(years, pop_difference, label=scenario_name.replace('_', ' ').title(),
                        linewidth=2, color=color)
        
        ax4.set_xlabel('Year') 
        ax4.set_ylabel('Population Difference vs Baseline (%)')
        ax4.set_title('Policy Impact Over Time (vs Baseline)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("step_by_step_changes.png", dpi=300, bbox_inches='tight')
    plt.close()

# Plot 4: Demographic transition analysis
def plot_demographic_transition():
    """Plot demographic transition analysis."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Process first run with data
    for run in runs:
        if run not in all_results:
            continue
        results = all_results[run]
        if 'policy_simulations' not in results:
            continue
        simulations = results['policy_simulations']
        config = results.get('config', {})
        break
    else:
        print("No policy simulation data found.")
        return
    
    base_year = 2024
    
    # 1. Dependency ratios
    for scenario_name, data in simulations.items():
        if 'random' not in scenario_name:
            trajectory = data['population_trajectory']
            dependency_ratios = []
            
            for pop in trajectory:
                if len(pop) >= 15:
                    young = pop[:3].sum()  # 0-14
                    working = pop[3:13].sum()  # 15-64
                    old = pop[13:].sum()  # 65+
                    
                    dependency_ratio = ((young + old) / working) * 100 if working > 0 else 0
                    dependency_ratios.append(dependency_ratio)
            
            # Create years array that matches dependency ratios length
            years = np.arange(base_year, base_year + len(dependency_ratios))
            color = policy_colors.get(scenario_name, 'gray')
            ax1.plot(years, dependency_ratios, 
                    label=scenario_name.replace('_', ' ').title(), linewidth=2, color=color)
    
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Dependency Ratio (%)')
    ax1.set_title('Total Dependency Ratio Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Working age population
    for scenario_name, data in simulations.items():
        if 'random' not in scenario_name:
            trajectory = data['population_trajectory']
            working_age_pop = []
            
            for pop in trajectory:
                if len(pop) >= 15:
                    working = pop[3:13].sum() / 1000  # Convert to thousands
                    working_age_pop.append(working)
            
            # Create years array that matches working age population length
            years = np.arange(base_year, base_year + len(working_age_pop))
            color = policy_colors.get(scenario_name, 'gray')
            ax2.plot(years, working_age_pop, 
                    label=scenario_name.replace('_', ' ').title(), linewidth=2, color=color)
    
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Working Age Population (thousands)')
    ax2.set_title('Working Age Population (15-64) Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Population pyramid comparison (2024 vs 2050)
    if 'baseline' in simulations:
        baseline_data = simulations['baseline']
        trajectory = baseline_data['population_trajectory']
        
        if len(trajectory) > 25:
            pop_2024 = trajectory[0]
            pop_2050 = trajectory[25]
            
            # Aggregate to 5-year groups
            pop_2024_5yr = []
            pop_2050_5yr = []
            
            for i in range(0, min(len(pop_2024), 20), 1):
                pop_2024_5yr.append(pop_2024[i])
                pop_2050_5yr.append(pop_2050[i] if i < len(pop_2050) else 0)
            
            # Assuming roughly equal gender distribution
            pop_2024_male = np.array(pop_2024_5yr) * 0.485 / 1000
            pop_2024_female = np.array(pop_2024_5yr) * 0.515 / 1000
            pop_2050_male = np.array(pop_2050_5yr) * 0.485 / 1000
            pop_2050_female = np.array(pop_2050_5yr) * 0.515 / 1000
            
            y_pos = np.arange(len(pop_2024_male))
            
            # 2024 pyramid
            ax3.barh(y_pos, -pop_2024_male, height=0.4, color='lightblue', alpha=0.7, label='2024 Male')
            ax3.barh(y_pos, pop_2024_female, height=0.4, color='lightcoral', alpha=0.7, label='2024 Female')
            
            # 2050 pyramid (outline)
            ax3.barh(y_pos + 0.4, -pop_2050_male, height=0.4, color='darkblue', alpha=0.7, label='2050 Male')
            ax3.barh(y_pos + 0.4, pop_2050_female, height=0.4, color='darkred', alpha=0.7, label='2050 Female')
            
            ax3.set_yticks(y_pos + 0.2)
            ax3.set_yticklabels([f"{i*5}-{i*5+4}" for i in range(len(y_pos))])
            ax3.set_xlabel('Population (thousands)')
            ax3.set_ylabel('Age Group')
            ax3.set_title('Population Pyramid: 2024 vs 2050 (Baseline)')
            ax3.legend()
            ax3.axvline(x=0, color='black', linestyle='-', alpha=0.5)
            ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. Policy impact summary table
    summary_data = []
    for scenario_name, data in simulations.items():
        if 'random' not in scenario_name:
            total_pops = data['total_population']
            aging_ratios = data['aging_ratio']
            
            if len(total_pops) > 25 and len(aging_ratios) > 25:
                pop_change = ((total_pops[25] - total_pops[0]) / total_pops[0]) * 100
                aging_2050 = aging_ratios[25] * 100
                
                summary_data.append({
                    'Scenario': scenario_name.replace('_', ' ').title(),
                    'Pop Change 2050 (%)': f"{pop_change:.1f}",
                    'Aging Ratio 2050 (%)': f"{aging_2050:.1f}"
                })
    
    # Create table
    ax4.axis('tight')
    ax4.axis('off')
    
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        table = ax4.table(cellText=df_summary.values,
                         colLabels=df_summary.columns,
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.4, 0.3, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Color code rows
        for i in range(len(summary_data)):
            pop_change = float(summary_data[i]['Pop Change 2050 (%)'])
            if pop_change > -5:
                color = 'lightgreen'
            elif pop_change > -15:
                color = 'yellow'
            else:
                color = 'lightcoral'
            
            for j in range(len(df_summary.columns)):
                table[(i+1, j)].set_facecolor(color)
    
    ax4.set_title('Policy Impact Summary', pad=20)
    
    plt.tight_layout()
    plt.savefig("demographic_transition.png", dpi=300, bbox_inches='tight')
    plt.close()

# Plot 5: Uncertainty analysis (if stochastic data available)
def plot_uncertainty_analysis():
    """Plot uncertainty analysis for stochastic models."""
    # Process first run with data
    for run in runs:
        if run not in all_results:
            continue
        results = all_results[run]
        if 'policy_simulations' not in results:
            continue
        simulations = results['policy_simulations']
        config = results.get('config', {})
        break
    else:
        print("No policy simulation data found.")
        return
    
    # Check if we have stochastic data
    has_stochastic = any('total_population_std' in data for data in simulations.values())
    if not has_stochastic:
        print("No stochastic data found. Uncertainty analysis requires stochastic_leslie model.")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    base_year = 2024
    
    # 1. Population uncertainty bands (baseline vs comprehensive)
    key_scenarios = ['baseline', 'comprehensive']
    for scenario_name in key_scenarios:
        if scenario_name in simulations:
            data = simulations[scenario_name]
            color = policy_colors.get(scenario_name, 'gray')
            
            # Mean trajectory
            mean_pop = np.array(data['total_population']) / 1000
            # Create years array that matches data length
            years = np.arange(base_year, base_year + len(mean_pop))
            ax1.plot(years, mean_pop, label=f"{scenario_name.title()} (Mean)", 
                    linewidth=3, color=color)
            
            # Uncertainty bands
            if 'total_population_p5' in data and 'total_population_p95' in data:
                p5 = np.array(data['total_population_p5']) / 1000
                p25 = np.array(data['total_population_p25']) / 1000 if 'total_population_p25' in data else p5
                p75 = np.array(data['total_population_p75']) / 1000 if 'total_population_p75' in data else mean_pop
                p95 = np.array(data['total_population_p95']) / 1000
                
                # Ensure all arrays have same length as years
                min_len = min(len(years), len(p5), len(p25), len(p75), len(p95))
                years_trimmed = years[:min_len]
                
                # 50% confidence interval (dark)
                ax1.fill_between(years_trimmed, p25[:min_len], p75[:min_len], alpha=0.4, color=color, 
                               label=f"{scenario_name.title()} (50% CI)")
                # 90% confidence interval (light)  
                ax1.fill_between(years_trimmed, p5[:min_len], p95[:min_len], alpha=0.2, color=color,
                               label=f"{scenario_name.title()} (90% CI)")
    
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Population (thousands)')
    ax1.set_title('Population Uncertainty Bands (Stochastic Projections)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Aging ratio uncertainty
    for scenario_name in key_scenarios:
        if scenario_name in simulations:
            data = simulations[scenario_name]
            color = policy_colors.get(scenario_name, 'gray')
            
            # Mean aging ratio
            mean_aging = np.array(data['aging_ratio']) * 100
            # Create years array that matches data length
            years = np.arange(base_year, base_year + len(mean_aging))
            ax2.plot(years, mean_aging, label=f"{scenario_name.title()} (Mean)", 
                    linewidth=3, color=color)
            
            # Uncertainty bands for aging ratio
            if 'aging_ratio_p5' in data and 'aging_ratio_p95' in data:
                p5_aging = np.array(data['aging_ratio_p5']) * 100
                p95_aging = np.array(data['aging_ratio_p95']) * 100
                
                # Ensure all arrays have same length
                min_len = min(len(years), len(p5_aging), len(p95_aging))
                years_trimmed = years[:min_len]
                
                ax2.fill_between(years_trimmed, p5_aging[:min_len], p95_aging[:min_len], alpha=0.3, color=color,
                               label=f"{scenario_name.title()} (90% CI)")
    
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Aging Ratio (%)')
    ax2.set_title('Aging Ratio Uncertainty (65+ Population %)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Policy effectiveness with uncertainty
    baseline_data = simulations.get('baseline', {})
    if 'total_population' in baseline_data:
        baseline_mean = np.array(baseline_data['total_population'])
        
        for scenario_name, data in simulations.items():
            if scenario_name != 'baseline' and 'total_population' in data:
                color = policy_colors.get(scenario_name, 'gray')
                
                # Mean effectiveness
                scenario_mean = np.array(data['total_population'])
                # Ensure both arrays have same length
                min_len = min(len(baseline_mean), len(scenario_mean))
                effectiveness_mean = ((scenario_mean[:min_len] - baseline_mean[:min_len]) / baseline_mean[:min_len]) * 100
                
                # Create years array that matches data length
                years = np.arange(base_year, base_year + min_len)
                
                ax3.plot(years, effectiveness_mean, label=f"{scenario_name.replace('_', ' ').title()}", 
                        linewidth=2, color=color)
                
                # Add uncertainty if available
                if ('total_population_p5' in data and 'total_population_p95' in data and 
                    'total_population_p5' in baseline_data and 'total_population_p95' in baseline_data):
                    
                    # Calculate effectiveness bounds
                    scenario_p5 = np.array(data['total_population_p5'])
                    scenario_p95 = np.array(data['total_population_p95'])
                    baseline_p95 = np.array(baseline_data['total_population_p95'])
                    baseline_p5 = np.array(baseline_data['total_population_p5'])
                    
                    # Ensure all arrays have same length
                    min_unc_len = min(len(scenario_p5), len(scenario_p95), len(baseline_p5), len(baseline_p95), min_len)
                    years_unc = years[:min_unc_len]
                    
                    # Conservative uncertainty: worst case for policy effectiveness
                    eff_low = ((scenario_p5[:min_unc_len] - baseline_p95[:min_unc_len]) / baseline_p95[:min_unc_len]) * 100
                    eff_high = ((scenario_p95[:min_unc_len] - baseline_p5[:min_unc_len]) / baseline_p5[:min_unc_len]) * 100
                    
                    ax3.fill_between(years_unc, eff_low, eff_high, alpha=0.2, color=color)
    
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Population Difference vs Baseline (%)')
    ax3.set_title('Policy Effectiveness with Uncertainty')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # 4. Uncertainty magnitude over time
    for scenario_name in ['baseline', 'comprehensive']:
        if scenario_name in simulations:
            data = simulations[scenario_name]
            color = policy_colors.get(scenario_name, 'gray')
            
            if 'total_population_std' in data:
                uncertainty = np.array(data['total_population_std']) / 1000  # Convert to thousands
                # Create years array that matches data length
                years = np.arange(base_year, base_year + len(uncertainty))
                ax4.plot(years, uncertainty, label=f"{scenario_name.title()}", 
                        linewidth=2, color=color)
    
    ax4.set_xlabel('Year')
    ax4.set_ylabel('Population Standard Deviation (thousands)')
    ax4.set_title('Projection Uncertainty Over Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("uncertainty_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Generate all standard plots
    print("Generating population analysis plots...")
    
    plot_population_projections()
    print("✓ Population projections plot saved")
    
    plot_policy_analysis()
    print("✓ Policy analysis plot saved")
    
    plot_step_by_step_changes()
    print("✓ Step-by-step changes plot saved")
    
    plot_demographic_transition()
    print("✓ Demographic transition plot saved")
    
    plot_uncertainty_analysis()
    print("✓ Uncertainty analysis plot saved")

"""
Visualization module for population dynamics and policy simulation results.
Generates comprehensive plots for demographic analysis and policy impact assessment.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pickle
import json
import os
from pathlib import Path
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, continue without it

# Set style from environment or default
plot_style = os.getenv("PLOT_STYLE", "seaborn-v0_8")
try:
    plt.style.use(plot_style)
except:
    plt.style.use('seaborn-v0_8')  # fallback
sns.set_palette("husl")

class PopulationPlotter:
    """
    Comprehensive plotting class for population dynamics and policy analysis.
    """
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results = None
        self.config = None
        self.load_results()
        
        # Color schemes
        self.policy_colors = {
            'baseline': '#1f77b4',
            'pro_natalist': '#ff7f0e', 
            'immigration_focused': '#2ca02c',
            'work_life_balance': '#d62728',
            'regional_development': '#9467bd',
            'comprehensive': '#8c564b',
        }
        
        # Age group labels
        self.age_labels = [f"{i*5}-{i*5+4}" for i in range(20)] + ["100+"]
        
    def load_results(self):
        """Load experimental results and configuration."""
        try:
            with open(self.results_dir / "results.pkl", "rb") as f:
                self.results = pickle.load(f)
            
            with open(self.results_dir / "config.json", "r") as f:
                self.config = json.load(f)
                
            print("Results loaded successfully!")
            print(f"Model type: {self.config.get('model_type', 'Unknown')}")
            
        except FileNotFoundError:
            print("No results found. Run experiment first.")
            self.results = {}
            self.config = {}
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """Plot model training and validation curves."""
        if 'train_losses' not in self.results:
            print("No training data found (mathematical models don't require training).")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Training curves
        epochs = range(len(self.results['train_losses']))
        ax1.plot(epochs, self.results['train_losses'], label='Training Loss', linewidth=2)
        ax1.plot(epochs, self.results['val_losses'], label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MSE Loss')
        ax1.set_title('Model Training Progress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Loss comparison (last 50 epochs)
        recent_epochs = max(50, len(epochs) // 4)
        recent_train = self.results['train_losses'][-recent_epochs:]
        recent_val = self.results['val_losses'][-recent_epochs:]
        
        ax2.plot(range(recent_epochs), recent_train, label='Training Loss', linewidth=2)
        ax2.plot(range(recent_epochs), recent_val, label='Validation Loss', linewidth=2)
        ax2.set_xlabel(f'Last {recent_epochs} Epochs')
        ax2.set_ylabel('MSE Loss')
        ax2.set_title('Recent Training Progress')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            dpi = int(os.getenv("PLOT_DPI", "300"))
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.show()
    
    def plot_population_projections(self, save_path: Optional[str] = None):
        """Plot population projections under different policy scenarios."""
        if 'policy_simulations' not in self.results:
            print("No policy simulation data found.")
            return
        
        simulations = self.results['policy_simulations']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Total population trajectories
        base_year = 2024
        projection_years = self.config.get('projection_years', 50)
        years = np.arange(base_year, base_year + projection_years + 1)
        
        for scenario_name, data in simulations.items():
            # Handle different data structures from mathematical models
            if 'total_population' in data:
                total_pop = np.array(data['total_population']) / 1000  # Convert to thousands
            elif 'total_population_mean' in data:  # Stochastic model
                total_pop = np.array(data['total_population_mean']) / 1000
            else:
                continue
                
            color = self.policy_colors.get(scenario_name, None)
            
            if 'baseline' in scenario_name or 'comprehensive' in scenario_name:
                ax1.plot(years, total_pop, label=scenario_name.replace('_', ' ').title(), 
                        linewidth=3, color=color)
            else:
                ax1.plot(years, total_pop, label=scenario_name.replace('_', ' ').title(), 
                        linewidth=2, alpha=0.8, color=color)
        
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Population (thousands)')
        ax1.set_title('Population Projections by Policy Scenario')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. Aging ratio trajectories
        for scenario_name, data in simulations.items():
            aging_ratio = np.array(data['aging_ratio']) * 100  # Convert to percentage
            color = self.policy_colors.get(scenario_name, None)
            
            ax2.plot(years, aging_ratio, label=scenario_name.replace('_', ' ').title(), 
                    linewidth=2, color=color)
        
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Aging Ratio (%)')
        ax2.set_title('Population Aging (65+ %) by Policy Scenario')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # 3. Population change comparison (2050 vs 2024)
        scenario_names = []
        pop_changes = []
        
        for scenario_name, data in simulations.items():
            total_pops = data['total_population']
            if len(total_pops) > 25:  # Ensure we have data for ~2050
                change = ((total_pops[25] - total_pops[0]) / total_pops[0]) * 100
                scenario_names.append(scenario_name.replace('_', ' ').title())
                pop_changes.append(change)
        
        colors = [self.policy_colors.get(name.lower().replace(' ', '_'), 'gray') for name in scenario_names]
        bars = ax3.bar(range(len(scenario_names)), pop_changes, color=colors, alpha=0.8)
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
        
        # 4. Age structure evolution (comprehensive scenario)
        if 'comprehensive' in simulations:
            comp_data = simulations['comprehensive']
            trajectory = comp_data['population_trajectory']
            
            # Show age structure at different time points
            time_points = [0, 15, 30]  # 2024, 2039, 2054
            time_labels = ['2024', '2039', '2054']
            
            x = np.arange(len(self.age_labels))
            width = 0.25
            
            for i, (time_idx, label) in enumerate(zip(time_points, time_labels)):
                if time_idx < len(trajectory):
                    age_dist = trajectory[time_idx] / trajectory[time_idx].sum() * 100
                    ax4.bar(x + i * width, age_dist, width, label=label, alpha=0.8)
            
            ax4.set_xlabel('Age Group')
            ax4.set_ylabel('Population Share (%)')
            ax4.set_title('Age Structure Evolution (Comprehensive Policy)')
            ax4.set_xticks(x[::2] + width)  # Set every other tick position
            ax4.set_xticklabels(self.age_labels[::2], rotation=45)  # Show every other label
            ax4.legend()
            ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            dpi = int(os.getenv("PLOT_DPI", "300"))
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.show()
    
    def plot_policy_feature_analysis(self, save_path: Optional[str] = None):
        """Analyze and visualize policy features and their impacts."""
        if 'policy_simulations' not in self.results:
            print("No policy simulation data found.")
            return
        
        simulations = self.results['policy_simulations']
        
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
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Policy features heatmap
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
        
        colors = [self.policy_colors.get(name.lower().replace(' ', '_'), 'gray') for name in scenario_subset]
        bars = ax2.bar(range(len(scenario_subset)), effectiveness, color=colors, alpha=0.8)
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
            correlation = np.corrcoef(feature_values, outcomes)[0, 1]
            correlations.append(correlation)
        
        colors = ['green' if corr > 0 else 'red' for corr in correlations]
        bars = ax3.barh(range(len(policy_names)), correlations, color=colors, alpha=0.7)
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
                
                color = self.policy_colors.get(scenarios[idx].lower().replace(' ', '_'), 'gray')
                ax4.plot(angles, values, 'o-', linewidth=2, label=scenarios[idx], color=color)
                ax4.fill(angles, values, alpha=0.25, color=color)
            
            ax4.set_xticks(angles[:-1])
            ax4.set_xticklabels([name.replace(' ', '\n') for name in policy_names], fontsize=8)
            ax4.set_ylim(0, 1)
            ax4.set_title('Policy Profile Comparison', pad=20)
            ax4.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
            ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            dpi = int(os.getenv("PLOT_DPI", "300"))
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.show()
    
    def plot_demographic_transition(self, save_path: Optional[str] = None):
        """Plot demographic transition analysis."""
        if 'policy_simulations' not in self.results:
            print("No policy simulation data found.")
            return
        
        simulations = self.results['policy_simulations']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        base_year = 2024
        years = np.arange(base_year, base_year + self.config.get('prediction_horizon', 50) + 1)
        
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
                
                color = self.policy_colors.get(scenario_name, 'gray')
                ax1.plot(years[:len(dependency_ratios)], dependency_ratios, 
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
                
                color = self.policy_colors.get(scenario_name, 'gray')
                ax2.plot(years[:len(working_age_pop)], working_age_pop, 
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
                
                # Create population pyramid
                age_groups_5yr = np.arange(0, 100, 5)
                
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
        
        if save_path:
            dpi = int(os.getenv("PLOT_DPI", "300"))
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.show()
    
    def create_summary_report(self, save_dir: Optional[str] = None):
        """Create a comprehensive summary report with all visualizations."""
        if save_dir is None:
            save_dir = self.results_dir
        else:
            save_dir = Path(save_dir)
        
        save_dir.mkdir(exist_ok=True)
        
        print("Generating comprehensive population analysis report...")
        
        # Generate all plots
        self.plot_training_curves(save_dir / "training_curves.png")
        self.plot_population_projections(save_dir / "population_projections.png")
        self.plot_policy_feature_analysis(save_dir / "policy_analysis.png")
        self.plot_demographic_transition(save_dir / "demographic_transition.png")
        
        print(f"Report saved to {save_dir}/")
        
        # Generate summary statistics
        if 'policy_simulations' in self.results:
            with open(save_dir / "summary_stats.txt", "w") as f:
                f.write("POPULATION DYNAMICS MODEL - SUMMARY REPORT\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"Model Configuration:\n")
                f.write(f"- Model Type: {self.config.get('model_type', 'Unknown')}\n")
                f.write(f"- Hidden Dimension: {self.config.get('hidden_dim', 'Unknown')}\n")
                f.write(f"- Training Epochs: {self.config.get('num_epochs', 'Unknown')}\n")
                f.write(f"- Prediction Horizon: {self.config.get('prediction_horizon', 'Unknown')} years\n\n")
                
                f.write("Policy Scenario Results (Population Change by 2050):\n")
                f.write("-" * 50 + "\n")
                
                simulations = self.results['policy_simulations']
                for scenario_name, data in simulations.items():
                    if 'random' not in scenario_name:
                        total_pops = data['total_population']
                        if len(total_pops) > 25:
                            change = ((total_pops[25] - total_pops[0]) / total_pops[0]) * 100
                            f.write(f"{scenario_name.replace('_', ' ').title():.<30} {change:>6.1f}%\n")
                
                f.write("\nNote: Negative values indicate population decline.\n")
                f.write("Values are relative to 2024 baseline population.\n")

def main():
    """Main plotting function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate population analysis plots")
    parser.add_argument("--plot_type", type=str, default="all",
                       choices=["training", "projections", "policy", "transition", "all"],
                       help="Type of plot to generate")
    
    args = parser.parse_args()
    
    # Find all run directories (following AI-Scientist pattern)
    folders = os.listdir("./")
    run_folders = [folder for folder in folders if folder.startswith("run") and os.path.isdir(folder)]
    
    if not run_folders:
        print("No run directories found. Looking for directories starting with 'run'.")
        return
    
    print(f"Found run directories: {run_folders}")
    
    # Process each run directory
    for run_dir in sorted(run_folders):
        print(f"\nProcessing {run_dir}...")
        
        try:
            plotter = PopulationPlotter(run_dir)
            
            if args.plot_type == "training":
                plotter.plot_training_curves(f"{run_dir}/training_curves.png")
            elif args.plot_type == "projections":
                plotter.plot_population_projections(f"{run_dir}/population_projections.png")
            elif args.plot_type == "policy":
                plotter.plot_policy_feature_analysis(f"{run_dir}/policy_analysis.png")
            elif args.plot_type == "transition":
                plotter.plot_demographic_transition(f"{run_dir}/demographic_transition.png")
            elif args.plot_type == "all":
                plotter.create_summary_report(run_dir)
                
            print(f"✓ Completed plots for {run_dir}")
            
        except Exception as e:
            print(f"✗ Error processing {run_dir}: {e}")
            continue

if __name__ == "__main__":
    main()
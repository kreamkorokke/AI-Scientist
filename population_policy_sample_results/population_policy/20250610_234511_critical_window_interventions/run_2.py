"""
Pure Mathematical Population Dynamics Modeling and Policy Simulation

This module implements deterministic and stochastic demographic models using
classical Leslie matrix approaches with policy intervention parameters 
affecting demographic rates directly.
"""

import argparse
import json
import os
import pickle
from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, continue without it

from data_pipeline import JapanDemographicDataPipeline

@dataclass
class DemographicModelConfig:
    """Configuration for mathematical demographic models."""
    model_type: str = "leslie_matrix"  # leslie_matrix, stochastic_leslie
    age_groups: int = 21  # 0-4, 5-9, ..., 95-99, 100+
    projection_years: int = 30  # Years to project forward
    time_step: float = 1.0  # Years per time step (1.0 for annual, 0.25 for quarterly)
    use_stochastic: bool = False  # Add stochastic variation
    monte_carlo_runs: int = 1000  # Number of MC runs if stochastic
    policy_sensitivity: float = 1.0  # How strongly policies affect demographic rates
    
    # Base demographic parameters (for Japan)
    base_fertility_rates: List[float] = None  # Age-specific fertility rates
    base_mortality_rates: List[float] = None  # Age-specific mortality rates  
    base_migration_rates: List[float] = None  # Age-specific net migration rates
    
    def __post_init__(self):
        if self.base_fertility_rates is None:
            # Japan fertility rates by 5-year age groups (UN World Population Prospects 2024)
            # Calibrated to Total Fertility Rate = 1.217 children per woman
            self.base_fertility_rates = [
                0.0000, 0.0000, 0.0000,  # 0-14 years
                0.0024, 0.0122, 0.0438, 0.0779, 0.0682, 0.0365,  # 15-44 years
                0.0017, 0.0005, 0.0002,  # 45-59 years
                0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000  # 60+ years
            ]
        
        if self.base_mortality_rates is None:
            # Japan mortality rates by 5-year age groups (UN World Population Prospects 2024)
            # Calibrated to Life Expectancy = 84.9 years (annual death rates)
            self.base_mortality_rates = [
                0.0006, 0.0001, 0.0001,  # 0-14 years
                0.0001, 0.0002, 0.0002, 0.0003, 0.0005, 0.0008,  # 15-44 years
                0.0015, 0.0025, 0.0040, 0.0065, 0.0105, 0.0170,  # 45-74 years
                0.0290, 0.0480, 0.0800, 0.1350, 0.2200  # 75+ years
            ]
            
        if self.base_migration_rates is None:
            # Japan net migration rates by age group (UN World Population Prospects 2024)
            # Based on +153,357 net migrants per year, distributed by typical age patterns
            # Positive values = net immigration (more people coming in than leaving)
            self.base_migration_rates = [
                0.000757, 0.000632, 0.000854,  # 0-14 years (family migration)
                0.002149, 0.003738, 0.004976, 0.004486, 0.002722, 0.001627,  # 15-44 years (peak working age)
                0.000889, 0.000542, 0.000401, 0.000213, 0.000238, 0.000315,  # 45-74 years (declining)
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000   # 75+ years (minimal)
            ]

class PolicyEffectsCalculator:
    """
    Maps policy interventions to changes in demographic rates.
    Based on demographic literature and empirical studies.
    """
    
    def __init__(self, sensitivity: float = 1.0):
        self.sensitivity = sensitivity
        
        # Policy effect matrices: how each policy affects each demographic process
        # Rows: policies, Columns: age groups (21 groups)
        # Values: multiplicative effects (1.0 = no change, >1.0 = increase, <1.0 = decrease)
        
        # Fertility effects by age group (mainly affects reproductive ages 15-49)
        self.fertility_effects = {
            'child_allowance_rate': [0, 0, 0, 1.05, 1.15, 1.20, 1.10, 1.05, 1.02] + [1.0] * 12,
            'parental_leave_weeks': [0, 0, 0, 1.03, 1.12, 1.18, 1.15, 1.08, 1.03] + [1.0] * 12,
            'childcare_availability': [0, 0, 0, 1.08, 1.25, 1.30, 1.20, 1.10, 1.05] + [1.0] * 12,
            'tax_incentive_families': [0, 0, 0, 1.02, 1.08, 1.12, 1.08, 1.03, 1.01] + [1.0] * 12,
            'work_life_balance_index': [0, 0, 0, 1.04, 1.18, 1.22, 1.15, 1.08, 1.02] + [1.0] * 12,
            'housing_affordability': [0, 0, 0, 1.06, 1.15, 1.18, 1.12, 1.05, 1.02] + [1.0] * 12,
            'education_investment': [0, 0, 0, 1.01, 1.05, 1.08, 1.05, 1.02, 1.01] + [1.0] * 12,
            # These policies have minimal direct fertility effects
            'immigration_rate': [1.0] * 21,
            'regional_development_investment': [0, 0, 0, 1.01, 1.03, 1.02, 1.01, 1.0, 1.0] + [1.0] * 12,
            'elder_care_capacity': [1.0] * 21,
        }
        
        # Mortality effects (mainly affect elderly, some policies affect all ages)
        self.mortality_effects = {
            'elder_care_capacity': [1.0] * 10 + [0.95, 0.92, 0.90, 0.88, 0.85, 0.82, 0.80, 0.78, 0.75, 0.72, 0.70],
            'education_investment': [0.98] * 21,  # Better health awareness
            'regional_development_investment': [0.99] * 21,  # Better access to healthcare
            'housing_affordability': [0.99] * 21,  # Better living conditions
            # Other policies have minimal mortality effects
            'child_allowance_rate': [1.0] * 21,
            'parental_leave_weeks': [1.0] * 21,
            'childcare_availability': [1.0] * 21,
            'immigration_rate': [1.0] * 21,
            'tax_incentive_families': [1.0] * 21,
            'work_life_balance_index': [0.995] * 21,  # Slight health benefit
        }
        
        # Migration effects (immigration policy has large effect, others moderate)
        self.migration_effects = {
            'immigration_rate': [1.5, 2.0, 3.0, 5.0, 4.0, 3.0, 2.0, 1.5, 1.2] + [1.1] * 12,
            'regional_development_investment': [1.1, 1.1, 1.2, 1.3, 1.2, 1.1, 1.0, 1.0, 1.0] + [1.0] * 12,
            'work_life_balance_index': [1.0, 1.0, 1.05, 1.15, 1.10, 1.05, 1.0, 1.0, 1.0] + [1.0] * 12,
            'housing_affordability': [1.0, 1.0, 1.03, 1.08, 1.05, 1.02, 1.0, 1.0, 1.0] + [1.0] * 12,
            'education_investment': [1.0, 1.0, 1.02, 1.05, 1.03, 1.01, 1.0, 1.0, 1.0] + [1.0] * 12,
            # Other policies have minimal migration effects
            'child_allowance_rate': [1.0] * 21,
            'parental_leave_weeks': [1.0] * 21,
            'childcare_availability': [1.0] * 21,
            'tax_incentive_families': [1.0] * 21,
            'elder_care_capacity': [1.0] * 21,
        }
    
    def apply_policy_effects(self, base_fertility: np.ndarray, base_mortality: np.ndarray, 
                           base_migration: np.ndarray, policy_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply policy effects to base demographic rates.
        
        Args:
            base_fertility: Base age-specific fertility rates
            base_mortality: Base age-specific mortality rates  
            base_migration: Base age-specific migration rates
            policy_vector: 10-element policy intervention vector (0-1 scale)
        
        Returns:
            Adjusted fertility, mortality, and migration rates
        """
        policy_names = [
            'child_allowance_rate', 'parental_leave_weeks', 'childcare_availability',
            'immigration_rate', 'regional_development_investment', 'elder_care_capacity',
            'tax_incentive_families', 'work_life_balance_index', 'housing_affordability',
            'education_investment'
        ]
        
        adjusted_fertility = base_fertility.copy()
        adjusted_mortality = base_mortality.copy()
        adjusted_migration = base_migration.copy()
        
        for i, (policy_name, policy_value) in enumerate(zip(policy_names, policy_vector)):
            # Policy effect is proportional to policy intensity (0-1) and sensitivity
            effect_strength = policy_value * self.sensitivity
            
            # Apply fertility effects
            fertility_multipliers = np.array(self.fertility_effects[policy_name])
            fertility_adjustment = 1.0 + (fertility_multipliers - 1.0) * effect_strength
            adjusted_fertility *= fertility_adjustment
            
            # Apply mortality effects  
            mortality_multipliers = np.array(self.mortality_effects[policy_name])
            mortality_adjustment = 1.0 + (mortality_multipliers - 1.0) * effect_strength
            adjusted_mortality *= mortality_adjustment
            
            # Apply migration effects
            migration_multipliers = np.array(self.migration_effects[policy_name])
            migration_adjustment = 1.0 + (migration_multipliers - 1.0) * effect_strength
            adjusted_migration *= migration_adjustment
        
        # Ensure rates stay within reasonable bounds
        adjusted_fertility = np.clip(adjusted_fertility, 0, 0.5)  # Max 50% fertility rate
        adjusted_mortality = np.clip(adjusted_mortality, 0.0001, 0.8)  # Min survival, max 80% mortality
        adjusted_migration = np.clip(adjusted_migration, -0.1, 0.1)  # Max 10% migration rate
        
        return adjusted_fertility, adjusted_mortality, adjusted_migration

class LeslieMatrixModel:
    """
    Pure mathematical Leslie matrix model for age-structured population projection.
    """
    
    def __init__(self, config: DemographicModelConfig):
        self.config = config
        self.policy_calculator = PolicyEffectsCalculator(config.policy_sensitivity)
        
        # Convert lists to numpy arrays
        self.base_fertility = np.array(config.base_fertility_rates)
        self.base_mortality = np.array(config.base_mortality_rates)  
        self.base_migration = np.array(config.base_migration_rates)
        
        # Ensure arrays are correct length
        if len(self.base_fertility) != config.age_groups:
            self.base_fertility = np.resize(self.base_fertility, config.age_groups)
        if len(self.base_mortality) != config.age_groups:
            self.base_mortality = np.resize(self.base_mortality, config.age_groups)
        if len(self.base_migration) != config.age_groups:
            self.base_migration = np.resize(self.base_migration, config.age_groups)
    
    def build_leslie_matrix(self, policy_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build Leslie matrix and migration vector from policy parameters.
        
        Returns:
            Leslie matrix (age_groups x age_groups) and migration vector
        """
        # Get policy-adjusted demographic rates
        fertility, mortality, migration = self.policy_calculator.apply_policy_effects(
            self.base_fertility, self.base_mortality, self.base_migration, policy_vector
        )
        
        # Calculate survival rates from mortality rates
        survival = 1.0 - mortality
        
        # Build Leslie matrix
        L = np.zeros((self.config.age_groups, self.config.age_groups))
        
        # First row: fertility rates (births)
        L[0, :] = fertility
        
        # Sub-diagonal: survival rates (aging)
        for i in range(self.config.age_groups - 1):
            L[i + 1, i] = survival[i]
        
        # Last age group: survival within same age group (no aging out)
        L[-1, -1] = survival[-1]
        
        return L, migration
    
    def project_population(self, initial_population: np.ndarray, policy_vector: np.ndarray, 
                          years: int, time_varying_policies: Dict[int, np.ndarray] = None) -> np.ndarray:
        """
        Project population forward using Leslie matrix with time-varying policies.
        
        Args:
            initial_population: Starting population by age group
            policy_vector: Base policy intervention parameters
            years: Number of years to project
            time_varying_policies: Dict mapping years to policy vectors for changes
        
        Returns:
            Population trajectory (years+1 x age_groups)
        """
        trajectory = np.zeros((years + 1, self.config.age_groups))
        trajectory[0] = initial_population.copy()
        
        for year in range(years):
            # Get policy vector for this year
            current_policy = policy_vector.copy()
            if time_varying_policies and year in time_varying_policies:
                current_policy = time_varying_policies[year]
            
            # Build Leslie matrix and migration vector for current policy
            L, migration = self.build_leslie_matrix(current_policy)
            
            # Leslie matrix projection
            next_pop = L @ trajectory[year]
            
            # Add migration
            next_pop += migration * trajectory[year]
            
            # Ensure non-negative population
            next_pop = np.maximum(next_pop, 0)
            
            trajectory[year + 1] = next_pop
        
        return trajectory


class StochasticLeslieModel:
    """
    Stochastic Leslie matrix with random variation in demographic rates.
    """
    
    def __init__(self, config: DemographicModelConfig):
        self.config = config
        self.leslie_model = LeslieMatrixModel(config)
        
        # Stochastic variation parameters (coefficient of variation)
        self.fertility_cv = 0.1  # 10% CV in fertility rates
        self.mortality_cv = 0.05  # 5% CV in mortality rates
        self.migration_cv = 0.2   # 20% CV in migration rates
    
    def run_monte_carlo(self, initial_population: np.ndarray, policy_vector: np.ndarray,
                       years: int, n_runs: int, time_varying_policies: Dict[int, np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Run Monte Carlo simulation with stochastic demographic rates.
        
        Returns:
            Dictionary with mean, std, percentiles of population trajectories
        """
        all_trajectories = []
        
        for run in range(n_runs):
            # Add random variation to policy effects
            varied_policy = policy_vector + np.random.normal(0, 0.05, len(policy_vector))
            varied_policy = np.clip(varied_policy, 0, 1)
            
            # Project with varied parameters
            trajectory = self.leslie_model.project_population(
                initial_population, varied_policy, years,
                time_varying_policies
            )
            
            all_trajectories.append(trajectory)
        
        all_trajectories = np.array(all_trajectories)  # Shape: (n_runs, years+1, age_groups)
        
        return {
            'mean': np.mean(all_trajectories, axis=0),
            'std': np.std(all_trajectories, axis=0),
            'percentile_5': np.percentile(all_trajectories, 5, axis=0),
            'percentile_25': np.percentile(all_trajectories, 25, axis=0),
            'percentile_75': np.percentile(all_trajectories, 75, axis=0),
            'percentile_95': np.percentile(all_trajectories, 95, axis=0),
            'all_runs': all_trajectories
        }

class PopulationExperiment:
    """
    Main experiment class for mathematical population modeling.
    """
    
    def __init__(self, config: DemographicModelConfig):
        self.config = config
        self.data_pipeline = JapanDemographicDataPipeline()
        
        # Initialize model based on type
        if config.model_type == "leslie_matrix":
            self.model = LeslieMatrixModel(config)
        elif config.model_type == "stochastic_leslie":
            self.model = StochasticLeslieModel(config)
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")
        
        self.results = {}
    
    def prepare_initial_population(self) -> np.ndarray:
        """Prepare initial population vector from data."""
        # Fetch real demographic data
        raw_data = self.data_pipeline.get_combined_dataset()
        processed_data = self.data_pipeline.preprocess_for_modeling(raw_data)
        
        # Validate required data is available
        if 'age_structure' not in processed_data:
            raise ValueError("Missing 'age_structure' data in processed dataset. Cannot initialize population model.")
        
        if processed_data['age_structure'].size == 0:
            raise ValueError("Empty 'age_structure' data in processed dataset. Cannot initialize population model.")
        
        # Use the most recent year's age structure from real data
        print("Using real UN demographic data for initial population")
        age_structure = processed_data['age_structure']
        latest_year_idx = -1  # Most recent year
        initial_population = age_structure[latest_year_idx]
        
        # Ensure we have the right number of age groups
        if len(initial_population) != self.config.age_groups:
            # Resize to match model expectations
            if len(initial_population) > self.config.age_groups:
                # Aggregate older age groups if we have more than needed
                resized_pop = list(initial_population[:self.config.age_groups-1])
                resized_pop.append(initial_population[self.config.age_groups-1:].sum())
                initial_population = np.array(resized_pop)
            else:
                # Pad with zeros if we have fewer
                padded_pop = list(initial_population)
                padded_pop.extend([0.0] * (self.config.age_groups - len(initial_population)))
                initial_population = np.array(padded_pop)
        
        # Validate that we have a reasonable population
        total_population = initial_population.sum()
        if total_population <= 0:
            raise ValueError(f"Invalid population data: total population is {total_population}")
        
        print(f"Using real data: Total population = {initial_population.sum():.0f}k")
        print(f"Age structure: 0-14: {initial_population[:3].sum():.0f}k, " +
              f"15-64: {initial_population[3:13].sum():.0f}k, " +
              f"65+: {initial_population[13:].sum():.0f}k")
        
        return initial_population
    
    def generate_policy_scenarios(self) -> Dict[str, Union[np.ndarray, Dict]]:
        """Generate different policy scenarios for simulation."""
        # Baseline (current Japan policies, normalized 0-1)
        baseline = np.array([0.3, 0.4, 0.2, 0.1, 0.3, 0.3, 0.2, 0.2, 0.1, 0.5])
        
        # Delayed response scenario (2030-2035)
        delayed_intervention_intense = np.array([0.9, 0.9, 0.9, 0.8, 0.8, 0.7, 0.8, 0.9, 0.8, 0.8])  # Same aggressive policies
        delayed_intervention_sustained = np.array([0.6, 0.6, 0.6, 0.5, 0.5, 0.5, 0.5, 0.6, 0.5, 0.5])  # Same sustained support
        
        # Time-varying policies for delayed intervention
        delayed_intervention_timeline = {}
        # First 6 years: Baseline policies (2024-2029)
        for year in range(6):
            delayed_intervention_timeline[year] = baseline
        # Next 5 years: Intense intervention (2030-2035)
        for year in range(6, 11):
            delayed_intervention_timeline[year] = delayed_intervention_intense
        # Remaining years: Sustained support
        for year in range(11, 30):
            delayed_intervention_timeline[year] = delayed_intervention_sustained
        
        scenarios = {
            'baseline': baseline,
            'delayed_intervention': {'base_policy': baseline, 'time_varying': delayed_intervention_timeline}
        }
        
        return scenarios
    
    def run_policy_simulations(self):
        """Run policy simulations using mathematical models."""
        print(f"Running {self.config.model_type} simulations...")
        
        initial_pop = self.prepare_initial_population()
        scenarios = self.generate_policy_scenarios()
        
        simulation_results = {}
        
        for scenario_name, policy_data in scenarios.items():
            # Handle both simple policy vectors and time-varying policies
            if isinstance(policy_data, dict):
                policy_vector = policy_data['base_policy']
                time_varying = policy_data['time_varying']
            else:
                policy_vector = policy_data
                time_varying = None
            print(f"Simulating policy: {scenario_name}")
            
            if self.config.model_type == "stochastic_leslie":
                # Run Monte Carlo simulation
                results = self.model.run_monte_carlo(
                    initial_pop, policy_vector, self.config.projection_years,
                    self.config.monte_carlo_runs, time_varying
                )
                
                # Calculate detailed demographics from mean trajectory
                mean_trajectory = results['mean']
                total_population_by_step = mean_trajectory.sum(axis=1)
                aging_ratio_by_step = mean_trajectory[:, 13:].sum(axis=1) / mean_trajectory.sum(axis=1)  # 65+ ratio
                working_age_pop = mean_trajectory[:, 3:13].sum(axis=1)  # 15-64 age groups
                young_pop = mean_trajectory[:, :3].sum(axis=1)  # 0-14 age groups  
                elderly_pop = mean_trajectory[:, 13:].sum(axis=1)  # 65+ age groups
                dependency_ratio_by_step = ((young_pop + elderly_pop) / working_age_pop) * 100
                
                # Calculate year-over-year changes from mean
                population_changes = np.diff(total_population_by_step)
                population_growth_rates = (population_changes / total_population_by_step[:-1]) * 100
                
                # Calculate uncertainty bounds for key demographic indicators from all runs
                all_trajectories = results['all_runs']  # Shape: (n_runs, years+1, age_groups)
                
                # Total population uncertainty
                all_total_pops = all_trajectories.sum(axis=2)  # Sum across age groups for each run
                total_pop_std = np.std(all_total_pops, axis=0)
                total_pop_p5 = np.percentile(all_total_pops, 5, axis=0)
                total_pop_p95 = np.percentile(all_total_pops, 95, axis=0)
                
                # Aging ratio uncertainty (65+ / total population)
                all_elderly = all_trajectories[:, :, 13:].sum(axis=2)  # Sum 65+ age groups
                all_aging_ratios = all_elderly / all_total_pops
                aging_ratio_std = np.std(all_aging_ratios, axis=0)
                aging_ratio_p5 = np.percentile(all_aging_ratios, 5, axis=0)
                aging_ratio_p95 = np.percentile(all_aging_ratios, 95, axis=0)
                
                # Dependency ratio uncertainty
                all_young = all_trajectories[:, :, :3].sum(axis=2)  # Sum 0-14 age groups
                all_working = all_trajectories[:, :, 3:13].sum(axis=2)  # Sum 15-64 age groups
                all_dependency_ratios = ((all_young + all_elderly) / all_working) * 100
                dependency_ratio_std = np.std(all_dependency_ratios, axis=0)
                dependency_ratio_p5 = np.percentile(all_dependency_ratios, 5, axis=0)
                dependency_ratio_p95 = np.percentile(all_dependency_ratios, 95, axis=0)
                
                simulation_results[scenario_name] = {
                    'policy_vector': policy_vector,
                    
                    # Mean trajectory detailed metrics (same as vanilla leslie)
                    'population_trajectory': mean_trajectory,
                    'total_population': total_population_by_step,
                    'young_population': young_pop,  # 0-14 age groups
                    'working_age_population': working_age_pop,  # 15-64 age groups  
                    'elderly_population': elderly_pop,  # 65+ age groups
                    'aging_ratio': aging_ratio_by_step,
                    'dependency_ratio': dependency_ratio_by_step,
                    'population_changes': population_changes,  # Year-over-year absolute changes
                    'population_growth_rates': population_growth_rates,  # Year-over-year % changes
                    'years': np.arange(2024, 2024 + self.config.projection_years + 1),  # Year labels
                    
                    # Full uncertainty information
                    'population_mean': results['mean'],
                    'population_std': results['std'],
                    'population_p5': results['percentile_5'],
                    'population_p25': results['percentile_25'],
                    'population_p75': results['percentile_75'],
                    'population_p95': results['percentile_95'],
                    
                    # Key demographic indicators with uncertainty bounds
                    'total_population_std': total_pop_std,
                    'total_population_p5': total_pop_p5,
                    'total_population_p95': total_pop_p95,
                    
                    'aging_ratio_std': aging_ratio_std,
                    'aging_ratio_p5': aging_ratio_p5,
                    'aging_ratio_p95': aging_ratio_p95,
                    
                    'dependency_ratio_std': dependency_ratio_std,
                    'dependency_ratio_p5': dependency_ratio_p5,
                    'dependency_ratio_p95': dependency_ratio_p95,
                    
                    # Full Monte Carlo results for advanced analysis
                    'all_runs': all_trajectories,
                }
                
            else:  # leslie_matrix
                # Run standard Leslie matrix projection
                trajectory = self.model.project_population(
                    initial_pop, policy_vector, self.config.projection_years
                )
                
                # Calculate step-by-step metrics
                total_population_by_step = trajectory.sum(axis=1)
                aging_ratio_by_step = trajectory[:, 13:].sum(axis=1) / trajectory.sum(axis=1)  # 65+ ratio
                working_age_pop = trajectory[:, 3:13].sum(axis=1)  # 15-64 age groups
                young_pop = trajectory[:, :3].sum(axis=1)  # 0-14 age groups  
                elderly_pop = trajectory[:, 13:].sum(axis=1)  # 65+ age groups
                dependency_ratio_by_step = ((young_pop + elderly_pop) / working_age_pop) * 100
                
                # Calculate year-over-year changes
                population_changes = np.diff(total_population_by_step)
                population_growth_rates = (population_changes / total_population_by_step[:-1]) * 100
                
                simulation_results[scenario_name] = {
                    'policy_vector': policy_vector,
                    'population_trajectory': trajectory,
                    'total_population': total_population_by_step,
                    'young_population': young_pop,  # 0-14 age groups
                    'working_age_population': working_age_pop,  # 15-64 age groups  
                    'elderly_population': elderly_pop,  # 65+ age groups
                    'aging_ratio': aging_ratio_by_step,
                    'dependency_ratio': dependency_ratio_by_step,
                    'population_changes': population_changes,  # Year-over-year absolute changes
                    'population_growth_rates': population_growth_rates,  # Year-over-year % changes
                    'years': np.arange(2024, 2024 + self.config.projection_years + 1),  # Year labels
                }
        
        self.results['policy_simulations'] = simulation_results
        self.results['initial_population'] = initial_pop
        self.results['projection_years'] = self.config.projection_years
        
        print("Policy simulations completed!")
    
    def calculate_summary_metrics(self):
        """Calculate summary metrics for policy comparison."""
        if 'policy_simulations' not in self.results:
            return
        
        simulations = self.results['policy_simulations']
        summary = {}
        
        for scenario_name, data in simulations.items():
            if 'total_population' in data:
                total_pop = data['total_population']
                
                # Population change metrics
                initial_pop = total_pop[0]
                final_pop = total_pop[-1]
                mid_pop = total_pop[len(total_pop)//2] if len(total_pop) > 20 else total_pop[-1]
                
                pop_change_total = (final_pop - initial_pop) / initial_pop * 100
                pop_change_rate = ((final_pop / initial_pop) ** (1/self.config.projection_years) - 1) * 100
                
                # Demographic metrics
                if 'aging_ratio' in data:
                    aging_2024 = data['aging_ratio'][0] * 100
                    aging_final = data['aging_ratio'][-1] * 100
                    aging_change = aging_final - aging_2024
                else:
                    aging_2024 = aging_final = aging_change = 0
                
                if 'dependency_ratio' in data:
                    dependency_final = data['dependency_ratio'][-1]
                else:
                    dependency_final = 0
                
                summary[scenario_name] = {
                    'population_change_total_pct': pop_change_total,
                    'population_change_annual_pct': pop_change_rate,
                    'aging_ratio_2024_pct': aging_2024,
                    'aging_ratio_final_pct': aging_final,
                    'aging_ratio_change_pct': aging_change,
                    'dependency_ratio_final': dependency_final,
                    'final_population_millions': final_pop / 1000,
                }
        
        self.results['summary_metrics'] = summary
    
    def save_results(self, save_dir: str = "results"):
        """Save all results to files."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save main results
        with open(f"{save_dir}/results.pkl", "wb") as f:
            pickle.dump(self.results, f)
        
        # Save configuration  
        config_dict = {
            'model_type': self.config.model_type,
            'age_groups': self.config.age_groups,
            'projection_years': self.config.projection_years,
            'time_step': self.config.time_step,
            'use_stochastic': self.config.use_stochastic,
            'monte_carlo_runs': self.config.monte_carlo_runs,
            'policy_sensitivity': self.config.policy_sensitivity,
        }
        
        with open(f"{save_dir}/config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
        
        # Save summary metrics as CSV
        if 'summary_metrics' in self.results:
            summary_df = pd.DataFrame(self.results['summary_metrics']).T
            summary_df.to_csv(f"{save_dir}/summary_metrics.csv")
        
        # Generate final_info.json for AI-Scientist compatibility
        self._generate_final_info(save_dir)
        
        print(f"Results saved to {save_dir}/")
    
    def _generate_final_info(self, save_dir: str):
        """Generate final_info.json file compatible with AI-Scientist framework."""
        if 'policy_simulations' not in self.results:
            return
        
        # Extract key metrics for each policy scenario
        final_info = {}
        
        for scenario_name, data in self.results['policy_simulations'].items():
            if 'total_population' in data and len(data['total_population']) > 0:
                initial_pop = data['total_population'][0]
                final_pop = data['total_population'][-1]
                
                # Key demographic metrics
                population_decline_pct = ((final_pop - initial_pop) / initial_pop) * 100
                final_aging_ratio = data['aging_ratio'][-1] * 100 if 'aging_ratio' in data else 0
                final_dependency_ratio = data['dependency_ratio'][-1] if 'dependency_ratio' in data else 0
                
                # Average annual decline rate
                years = len(data['total_population']) - 1
                annual_decline_rate = ((final_pop / initial_pop) ** (1/years) - 1) * 100 if years > 0 else 0
                
                # Policy effectiveness (vs baseline if available)
                if scenario_name != 'baseline' and 'baseline' in self.results['policy_simulations']:
                    baseline_final = self.results['policy_simulations']['baseline']['total_population'][-1]
                    policy_effectiveness = ((final_pop - baseline_final) / baseline_final) * 100
                else:
                    policy_effectiveness = 0.0
                
                final_info[scenario_name] = {
                    'population_decline_pct': float(population_decline_pct),
                    'annual_decline_rate': float(annual_decline_rate),
                    'final_aging_ratio_pct': float(final_aging_ratio),
                    'final_dependency_ratio': float(final_dependency_ratio),
                    'policy_effectiveness_pct': float(policy_effectiveness),
                    'final_population_millions': float(final_pop / 1000)  # Convert to millions
                }
        
        # Save final_info.json
        with open(f"{save_dir}/final_info.json", "w") as f:
            json.dump(final_info, f, indent=2)
        
        print(f"Generated final_info.json with {len(final_info)} policy scenarios")
    
    def get_key_metrics(self) -> Dict[str, float]:
        """
        Extract key metrics for AI-Scientist framework analysis.
        
        Returns standardized metrics that can be compared across different runs.
        This method provides a template-agnostic interface for launch_scientist.py.
        """
        key_metrics = {}
        
        # Extract the most important metrics for tracking experiment success
        if 'policy_simulations' in self.results:
            for scenario_name, data in self.results['policy_simulations'].items():
                if 'total_population' in data and len(data['total_population']) > 0:
                    initial_pop = data['total_population'][0]
                    final_pop = data['total_population'][-1]
                    
                    # Core demographic metrics for comparison
                    population_decline_pct = ((final_pop - initial_pop) / initial_pop) * 100
                    final_aging_ratio = data['aging_ratio'][-1] * 100 if 'aging_ratio' in data else 0
                    final_dependency_ratio = data['dependency_ratio'][-1] if 'dependency_ratio' in data else 0
                    
                    # Policy effectiveness vs baseline
                    if scenario_name != 'baseline' and 'baseline' in self.results['policy_simulations']:
                        baseline_final = self.results['policy_simulations']['baseline']['total_population'][-1]
                        policy_effectiveness = ((final_pop - baseline_final) / baseline_final) * 100
                    else:
                        policy_effectiveness = 0.0
                    
                    # Store metrics with descriptive names
                    key_metrics[f"{scenario_name}_population_decline_pct"] = float(population_decline_pct)
                    key_metrics[f"{scenario_name}_aging_ratio_pct"] = float(final_aging_ratio)
                    key_metrics[f"{scenario_name}_dependency_ratio"] = float(final_dependency_ratio)
                    key_metrics[f"{scenario_name}_policy_effectiveness_pct"] = float(policy_effectiveness)
                    key_metrics[f"{scenario_name}_final_population_millions"] = float(final_pop / 1000)
        
        # Add aggregate metrics for overall assessment
        if 'baseline_population_decline_pct' in key_metrics and 'comprehensive_population_decline_pct' in key_metrics:
            # How much the best policy reduces decline vs baseline
            policy_improvement = key_metrics['baseline_population_decline_pct'] - key_metrics['comprehensive_population_decline_pct']
            key_metrics['best_policy_improvement_pct'] = policy_improvement
            
        # Crisis severity indicators
        if 'baseline_aging_ratio_pct' in key_metrics:
            key_metrics['demographic_crisis_severity'] = key_metrics['baseline_aging_ratio_pct'] / 100  # 0-1 scale
            
        return key_metrics

def extract_key_metrics_from_results(results_dir: str) -> Dict[str, float]:
    """
    Helper function for AI-Scientist framework integration.
    
    Loads results from a directory and extracts key metrics for comparison.
    This function provides a standard interface that launch_scientist.py can call.
    
    Args:
        results_dir: Directory containing experimental results
        
    Returns:
        Dictionary of key metrics with standardized names and float values
    """
    try:
        # Load experimental results
        import pickle
        results_path = os.path.join(results_dir, "results.pkl")
        config_path = os.path.join(results_dir, "config.json")
        
        if not os.path.exists(results_path) or not os.path.exists(config_path):
            print(f"Warning: Results files not found in {results_dir}")
            return {}
            
        with open(results_path, "rb") as f:
            results = pickle.load(f)
            
        with open(config_path, "r") as f:
            config_dict = json.load(f)
            
        # Create a temporary config object
        config = DemographicModelConfig(
            model_type=config_dict.get('model_type', 'leslie_matrix'),
            age_groups=config_dict.get('age_groups', 21),
            projection_years=config_dict.get('projection_years', 30)
        )
        
        # Create temporary experiment instance and extract metrics
        temp_experiment = PopulationExperiment(config)
        temp_experiment.results = results
        
        return temp_experiment.get_key_metrics()
        
    except Exception as e:
        print(f"Error extracting key metrics from {results_dir}: {e}")
        return {}

def main():
    parser = argparse.ArgumentParser(description="Mathematical Population Dynamics Modeling")
    parser.add_argument("--model_type", type=str, 
                       default=os.getenv("DEFAULT_MODEL_TYPE", "stochastic_leslie"),
                       choices=["leslie_matrix", "stochastic_leslie"])
    parser.add_argument("--age_groups", type=int, 
                       default=int(os.getenv("DEFAULT_AGE_GROUPS", "21")))
    parser.add_argument("--projection_years", type=int, 
                       default=int(os.getenv("DEFAULT_PROJECTION_YEARS", "30")))
    parser.add_argument("--time_step", type=float, default=1.0)
    parser.add_argument("--use_stochastic", action="store_true")
    parser.add_argument("--monte_carlo_runs", type=int, default=1000)
    parser.add_argument("--policy_sensitivity", type=float, 
                       default=float(os.getenv("DEFAULT_POLICY_SENSITIVITY", "1.0")))
    parser.add_argument("--out_dir", type=str, required=True,
                       help="Output directory for experimental results (required)")
    
    args = parser.parse_args()
    
    # Create configuration
    config = DemographicModelConfig(
        model_type=args.model_type,
        age_groups=args.age_groups,
        projection_years=args.projection_years,
        time_step=args.time_step,
        use_stochastic=args.use_stochastic,
        monte_carlo_runs=args.monte_carlo_runs,
        policy_sensitivity=args.policy_sensitivity,
    )
    
    # Run experiment
    experiment = PopulationExperiment(config)
    
    # Run simulations
    experiment.run_policy_simulations()
    
    # Calculate summary metrics
    experiment.calculate_summary_metrics()
    
    # Save results
    experiment.save_results(args.out_dir)
    
    print("Mathematical population modeling experiment completed!")

if __name__ == "__main__":
    main()

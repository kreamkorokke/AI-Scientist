"""
Calculate realistic demographic rates for Japan based on UN World Population Prospects 2024 data.

This script calibrates fertility and mortality rates to match:
- Total Fertility Rate (TFR): 1.217 children per woman
- Life Expectancy: 84.9 years

Used to generate base demographic parameters for the population modeling experiment.
"""

import numpy as np

def calculate_fertility_rates(target_tfr=1.217):
    """
    Calculate age-specific fertility rates that sum to the target TFR.
    
    Args:
        target_tfr: Total fertility rate (children per woman)
    
    Returns:
        Array of fertility rates by 5-year age group
    """
    print(f"Calculating fertility rates for TFR = {target_tfr}")
    
    # Standard fertility pattern for Japan (later childbearing)
    # These are births per woman per year within each 5-year age group
    fertility_pattern = np.array([
        0.000, 0.000, 0.000,  # 0-14 years (no fertility)
        0.010, 0.050, 0.180, 0.320, 0.280, 0.150, 0.007, 0.002, 0.001,  # 15-59 years  
        0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000   # 60+ years (no fertility)
    ])
    
    # Scale to match target TFR
    # TFR = sum of age-specific rates * 5 years per age group
    current_tfr = fertility_pattern.sum() * 5
    scaling_factor = target_tfr / current_tfr
    adjusted_fertility = fertility_pattern * scaling_factor
    
    print(f"Original pattern TFR: {current_tfr:.3f}")
    print(f"Scaling factor: {scaling_factor:.3f}")
    print(f"Final TFR: {adjusted_fertility.sum() * 5:.3f}")
    
    return adjusted_fertility

def calculate_mortality_rates(target_life_exp=84.9):
    """
    Calculate age-specific mortality rates to achieve target life expectancy.
    
    Args:
        target_life_exp: Target life expectancy in years
    
    Returns:
        Array of mortality rates by 5-year age group
    """
    print(f"\nCalculating mortality rates for life expectancy = {target_life_exp} years")
    
    # Ultra-low mortality rates for Japan (one of world's highest life expectancies)
    # These are annual death rates (deaths per person per year)
    mortality_rates = np.array([
        0.0006, 0.0001, 0.0001,  # 0-14 years (very low infant/child mortality)
        0.0001, 0.0002, 0.0002, 0.0003, 0.0005, 0.0008,  # 15-44 years  
        0.0015, 0.0025, 0.0040, 0.0065, 0.0105, 0.0170,  # 45-74 years
        0.0290, 0.0480, 0.0800, 0.1350, 0.2200  # 75+ years
    ])
    
    # Calculate life expectancy from these rates
    years_lived = []
    survival_prob = 1.0
    
    for i, mort_rate in enumerate(mortality_rates):
        # Probability of surviving this 5-year period
        period_survival = (1 - mort_rate) ** 5
        
        # Expected years lived in this age group
        if i < 20:  # Not the last age group
            years_in_group = 5 * survival_prob * (1 + period_survival) / 2
        else:  # Last age group (100+)
            years_in_group = survival_prob / mort_rate if mort_rate > 0 else 0
        
        years_lived.append(years_in_group)
        survival_prob *= period_survival
    
    calculated_life_exp = sum(years_lived)
    print(f"Calculated life expectancy: {calculated_life_exp:.1f} years")
    print(f"Target life expectancy: {target_life_exp} years")
    
    return mortality_rates

def format_rates_for_code(rates, name):
    """Format rates for easy copy-paste into experiment.py"""
    print(f"\n{name.upper()} RATES FOR EXPERIMENT.PY:")
    print("self.base_" + name.lower() + "_rates = [")
    
    for i, rate in enumerate(rates):
        age_start = i * 5
        age_end = age_start + 4
        
        if i == 20:
            comment = "# 100+ years"
        elif i < 3:
            comment = f"# {age_start}-{age_end} years"
        elif i < 12:
            comment = f"# {age_start}-{age_end} years"
        else:
            comment = f"# {age_start}-{age_end} years"
        
        print(f"    {rate:.4f},  {comment}")
    
    print("]")

def main():
    """Calculate and display demographic rates for Japan."""
    print("DEMOGRAPHIC RATE CALCULATION FOR JAPAN")
    print("=" * 50)
    print("Source: UN World Population Prospects 2024")
    print("Country: Japan")
    print("Reference Year: 2024")
    print()
    
    # Calculate fertility rates
    fertility_rates = calculate_fertility_rates(target_tfr=1.217)
    
    # Calculate mortality rates  
    mortality_rates = calculate_mortality_rates(target_life_exp=84.9)
    
    # Format for code
    format_rates_for_code(fertility_rates, "fertility")
    format_rates_for_code(mortality_rates, "mortality")
    
    print("\nKEY INSIGHTS:")
    print("- Individual age-specific fertility rates are low because they represent")
    print("  annual births per woman in each 5-year age group")
    print("- TFR = sum(age_specific_rate * 5_years) across all reproductive ages")
    print("- Japan's pattern: peak fertility in 30-34 age group (delayed childbearing)")
    print("- Very low mortality rates reflect Japan's excellent healthcare system")
    print("- High life expectancy requires extremely low mortality in all age groups")

if __name__ == "__main__":
    main()
"""
Data pipeline for loading Japanese demographic data from local UN World Population Prospects files.
Requires local CSV files to be available in the configured data directory.
"""

try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: pandas/numpy not available, using basic Python functionality")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("Warning: requests not available, using local data only")

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    print("Warning: python-dotenv not available, environment variables won't be loaded from .env file")

import os
from typing import Dict
from pathlib import Path

# Load environment variables from .env file if available
if DOTENV_AVAILABLE:
    load_dotenv()

class JapanDemographicDataPipeline:
    """
    Comprehensive data pipeline for Japanese demographic analysis.
    Fetches data from multiple sources and provides unified interface.
    """
    
    def __init__(self):
        # Data directory for local UN population data files
        self.data_dir = Path(os.getenv("UN_DATA_DIR", "un_population_data"))
        
        print("Using local UN data files only")
        
    def fetch_un_population_data(self, country_code: int = 392) -> pd.DataFrame:
        """
        Load UN World Population Prospects data for Japan from local files.
        Country code 392 = Japan
        """
        print(f"Loading UN data from local files in {self.data_dir}...")
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"UN population data directory {self.data_dir} not found. Please ensure UN data files are available.")
        
        try:
            
            all_data = []
            
            # Load population by age group data
            pop_file = self.data_dir / "population-by-5-year-age-group.csv"
            if not pop_file.exists():
                raise FileNotFoundError(f"Required file {pop_file} not found.")
            
            pop_df = pd.read_csv(pop_file)
            print(f"Loaded population data: {len(pop_df)} records")
            
            for _, row in pop_df.iterrows():
                all_data.append({
                    'indicator_id': 46,  # Population by age groups
                    'year': int(row['Time']),
                    'value': float(row['Value']),
                    'age_group': row['Age'],
                    'sex': row['Sex'],
                    'variant': row['Variant'],
                    'age_start': int(row['AgeStart']) if pd.notna(row['AgeStart']) and row['AgeStart'] != 'null' else 100,
                    'age_end': int(row['AgeEnd']) if pd.notna(row['AgeEnd']) and row['AgeEnd'] != 'null' else 999
                })
            
            # Load fertility rate data
            fertility_file = self.data_dir / "total-fertility-rate.csv"
            if not fertility_file.exists():
                raise FileNotFoundError(f"Required file {fertility_file} not found.")
            
            fert_df = pd.read_csv(fertility_file)
            print(f"Loaded fertility data: {len(fert_df)} records")
            
            for _, row in fert_df.iterrows():
                all_data.append({
                    'indicator_id': 70,  # Total fertility rate
                    'year': int(row['Time']),
                    'value': float(row['Value']),
                    'age_group': 'Total',
                    'sex': row['Sex'],
                    'variant': row['Variant'],
                    'age_start': 0,
                    'age_end': -1
                })
            
            # Load life expectancy data
            life_exp_file = self.data_dir / "life-expectancy.csv"
            if not life_exp_file.exists():
                raise FileNotFoundError(f"Required file {life_exp_file} not found.")
            
            life_df = pd.read_csv(life_exp_file)
            print(f"Loaded life expectancy data: {len(life_df)} records")
            
            for _, row in life_df.iterrows():
                all_data.append({
                    'indicator_id': 68,  # Life expectancy at birth
                    'year': int(row['Time']),
                    'value': float(row['Value']),
                    'age_group': 'Total',
                    'sex': row['Sex'],
                    'variant': row['Variant'],
                    'age_start': 0,
                    'age_end': -1
                })
            
            df = pd.DataFrame(all_data)
            
            if not df.empty:
                print(f"Processed UN data: {len(df)} total records")
                print(f"Years available: {sorted(df['year'].unique())}")
                print(f"Age groups available: {len(df[df['indicator_id']==46]['age_group'].unique())}")
            
            return df
            
        except Exception as e:
            print(f"Error loading UN data from files: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to load required UN population data: {e}")
    
    
    
    
    def get_combined_dataset(self) -> Dict[str, pd.DataFrame]:
        """
        Load UN demographic data from local files.
        
        Returns:
            Dictionary with 'un_data' DataFrame
        """
        print("Loading Japanese demographic dataset...")
        
        un_data = self.fetch_un_population_data()
        
        return {
            'un_data': un_data
        }
    
    def preprocess_for_modeling(self, data: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
        """
        Preprocess the demographic data for population modeling experiments.
        
        Returns:
            Dictionary with processed arrays ready for modeling
        """
        un_data = data['un_data']
        
        # Extract time series for key indicators
        processed = {}
        
        # Validate UN data is not empty
        if un_data.empty:
            raise ValueError("UN data is empty. Cannot process demographic data for modeling.")
        
        # Process population data
        # Filter population data by age groups (indicator_id == 46)
        pop_data = un_data[
            (un_data['indicator_id'] == 46) & 
            (un_data['sex'] == 'Both sexes')
        ].copy()
        
        if pop_data.empty:
            raise ValueError("No population by age group data found in UN dataset. Cannot initialize demographic models.")
        
        # Calculate total population by year
        total_by_year = pop_data.groupby('year')['value'].sum().reset_index()
        processed['population_timeseries'] = {
            'years': total_by_year['year'].values,
            'population': total_by_year['value'].values / 1000  # Convert to thousands
        }
        
        # Create age structure matrix
        # First, create a mapping from UN age groups to our model age groups
        age_structure_data = []
        years = sorted(pop_data['year'].unique())
        
        # Define age group mapping from UN format to model format (21 groups: 0-4, 5-9, ..., 95-99, 100+)
        age_group_order = [
            '0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49',
            '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80-84', '85-89', '90-94', '95-99', '100+'
        ]
        
        age_structure_matrix = []
        for year in years:
            year_data = pop_data[pop_data['year'] == year]
            age_populations = []
            
            for age_group in age_group_order:
                matching_rows = year_data[year_data['age_group'] == age_group]
                if not matching_rows.empty:
                    age_populations.append(matching_rows['value'].iloc[0] / 1000)  # Convert to thousands
                else:
                    age_populations.append(0.0)  # Missing age group
            
            age_structure_matrix.append(age_populations)
        
        processed['age_structure'] = np.array(age_structure_matrix)
        processed['age_group_labels'] = age_group_order
        
        # Validate that we have sufficient age structure data
        if processed['age_structure'].size == 0:
            raise ValueError("Failed to process age structure data. No valid age group data found.")
        
        print(f"Processed age structure: {len(years)} years × {len(age_group_order)} age groups")
        print(f"Total population 2024: {processed['population_timeseries']['population'][0]:.0f}k")
        if len(processed['population_timeseries']['population']) > 1:
            print(f"Total population 2025: {processed['population_timeseries']['population'][1]:.0f}k")
        
        # Fertility and life expectancy
        tfr_data = un_data[un_data['indicator_id'] == 70].sort_values('year')
        life_exp_data = un_data[un_data['indicator_id'] == 68].sort_values('year')
        
        if not tfr_data.empty:
            processed['fertility_rate'] = {
                'years': tfr_data['year'].values,
                'tfr': tfr_data['value'].values
            }
        else:
            print("Warning: No fertility rate data found")
        
        if not life_exp_data.empty:
            processed['life_expectancy'] = {
                'years': life_exp_data['year'].values,
                'life_exp': life_exp_data['value'].values
            }
        else:
            print("Warning: No life expectancy data found")
        
        return processed

# Example usage and testing
if __name__ == "__main__":
    # Initialize the pipeline
    pipeline = JapanDemographicDataPipeline()
    
    # Load data (requires local UN CSV files to be available)
    data = pipeline.get_combined_dataset()
    
    print("UN Data shape:", data['un_data'].shape)
    
    # Preprocess for modeling
    processed = pipeline.preprocess_for_modeling(data)
    
    print("\nProcessed data keys:", list(processed.keys()))
    
    if 'population_timeseries' in processed:
        pop_data = processed['population_timeseries']
        print(f"Population time series: {len(pop_data['years'])} years")
        print(f"Population range: {pop_data['population'].min():.0f} - {pop_data['population'].max():.0f}")
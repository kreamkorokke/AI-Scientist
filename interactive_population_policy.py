#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any


"""
Interactive Population Policy Explorer for AI-Scientist

This script allows users to dynamically specify population policy research directions
and automatically runs AI-Scientist experiments with custom prompts.

Usage:
    python interactive_population_policy.py

Example policy directions:
- "Crisis solutions emphasizing education and technology incentives"
- "Regional development focused on rural revitalization" 
- "Immigration-centric policies with integration support"
- "Work-life balance improvements for young families"
"""


class InteractivePopulationPolicy:
    """Interactive interface for dynamic population policy research with AI-Scientist."""
    
    def __init__(self):
        self.template_dir = Path("templates/population_policy")
        self.prompt_file = self.template_dir / "prompt.json"
        self.backup_file = self.template_dir / "prompt_original.json"
        
        # Base system prompt (constant)
        self.base_system = ("You are an ambitious AI PhD student specializing in computational "
                           "demography and policy analysis, looking to publish impactful research that "
                           "provides novel solutions to Japan's severe population decline crisis.")
        
        # Base task framework (constant)
        self.base_framework = ("You are given a rigorous mathematical population dynamics modeling "
                              "framework using real UN World Population Prospects 2024 data for Japan. "
                              "The framework includes: (1) Classical Leslie Matrix models for deterministic "
                              "age-structured population projection, (2) Stochastic Leslie Matrix models "
                              "with Monte Carlo simulation for uncertainty quantification (1000 runs with "
                              "confidence intervals), (3) Policy intervention effects mapping (10-dimensional "
                              "policy space: child allowance, parental leave, childcare, immigration, "
                              "regional development, elder care, tax incentives, work-life balance, housing, "
                              "education), and (4) Real Japanese demographic baselines (123.8M population, "
                              "30% elderly, 1.217 fertility rate, 30-year projections to 2054).")
        
        self.base_conclusion = ("Your experiments should validate policy effectiveness using mathematical "
                               "population projections with uncertainty bounds, provide concrete evidence "
                               "for policy recommendations, and demonstrate measurable demographic improvements.")
    
    def backup_original_prompt(self):
        """Create backup of original prompt.json if it doesn't exist."""
        if not self.backup_file.exists() and self.prompt_file.exists():
            with open(self.prompt_file, 'r') as f:
                original = json.load(f)
            with open(self.backup_file, 'w') as f:
                json.dump(original, f, indent=4)
            print(f"✓ Backed up original prompt to {self.backup_file}")
    
    def restore_original_prompt(self):
        """Restore original prompt.json from backup and clean up backup file."""
        if self.backup_file.exists():
            with open(self.backup_file, 'r') as f:
                original = json.load(f)
            with open(self.prompt_file, 'w') as f:
                json.dump(original, f, indent=4)
            # Delete the backup file after restoration
            self.backup_file.unlink()
            print("✓ Restored original prompt.json and cleaned up backup")
        else:
            print("⚠ No backup found. Cannot restore original prompt.")
    
    def get_policy_examples(self) -> Dict[str, str]:
        """Predefined policy direction examples for user convenience."""
        return {
            "1": {
                "name": "Use Original Prompt (No Changes)",
                "direction": None,
                "description": "Proceed with the existing prompt.json without modifications"
            },
            "2": {
                "name": "Crisis Solutions with Education & Technology",
                "direction": "crisis solutions emphasizing education and technology incentives",
                "description": "Focus on innovative educational policies and technology-driven solutions"
            },
            "3": {
                "name": "Rural Revitalization & Regional Development", 
                "direction": "regional development focused on rural revitalization and sustainable communities",
                "description": "Emphasize policies to revive rural areas and reduce urban concentration"
            },
            "4": {
                "name": "Immigration-Centric Integration",
                "direction": "immigration-focused policies with comprehensive integration and support systems",
                "description": "Explore immigration as a primary solution with strong integration measures"
            },
            "5": {
                "name": "Work-Life Balance for Families",
                "direction": "work-life balance improvements specifically targeting young families and parents",
                "description": "Focus on policies that help families balance career and child-rearing"
            },
            "6": {
                "name": "Economic Incentives & Housing",
                "direction": "economic and housing affordability policies to reduce barriers to family formation",
                "description": "Address financial barriers that prevent young people from having children"
            },
            "7": {
                "name": "Comprehensive Multi-Generational Approach",
                "direction": "comprehensive multi-generational policies addressing elderly care, youth support, and intergenerational solidarity",
                "description": "Holistic approach considering all age groups and generational interactions"
            }
        }
    
    def display_policy_options(self):
        """Display predefined policy direction options."""
        examples = self.get_policy_examples()
        
        print("\n" + "="*80)
        print("📋 PREDEFINED POLICY DIRECTIONS")
        print("="*80)
        
        for key, value in examples.items():
            print(f"\n{key}. {value['name']}")
            print(f"   Direction: \"{value['direction']}\"")
            print(f"   Focus: {value['description']}")
        
        print(f"\n{len(examples)+1}. Custom Direction")
        print("   Specify your own policy research direction")
        
        print("\n" + "="*80)
    
    def get_user_policy_direction(self) -> str:
        """Get policy direction from user input. Returns None for original prompt."""
        examples = self.get_policy_examples()
        
        while True:
            try:
                choice = input(f"\nSelect policy direction (1-{len(examples)+1}): ").strip()
                
                if choice in examples:
                    selected = examples[choice]
                    print(f"\n✓ Selected: {selected['name']}")
                    if selected['direction'] is None:
                        print("Direction: Keep existing prompt.json unchanged")
                    else:
                        print(f"Direction: \"{selected['direction']}\"")
                    
                    confirm = self._get_confirmation("Proceed with this selection?")
                    if confirm:
                        return selected['direction']
                    
                elif choice == str(len(examples) + 1):
                    print("\n📝 Enter your custom policy direction:")
                    print("Examples:")
                    print("- 'crisis solutions emphasizing education and technology incentives'")
                    print("- 'regional development focused on rural sustainability'")
                    print("- 'innovative childcare policies with AI and automation support'")
                    
                    custom_direction = input("\nYour direction: ").strip()
                    if custom_direction:
                        print(f"\n✓ Custom direction: \"{custom_direction}\"")
                        confirm = self._get_confirmation("Proceed with this direction?")
                        if confirm:
                            return custom_direction
                    else:
                        print("⚠ Direction cannot be empty.")
                
                else:
                    print(f"⚠ Invalid choice. Please enter 1-{len(examples)+1}")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                sys.exit(0)
            except EOFError:
                print("\n\n👋 Goodbye!")
                sys.exit(0)
    
    def get_model_selection(self, default_model: str) -> str:
        """Get model selection from user input."""
        models = {
            "1": {"name": "Claude 3.5 Sonnet", "id": "claude-3-5-sonnet-20241022", "description": "Anthropic's latest model (recommended)"},
            "2": {"name": "GPT-4o", "id": "gpt-4o", "description": "OpenAI's latest multimodal model"},
            "3": {"name": "GPT-4o Mini", "id": "gpt-4o-mini", "description": "OpenAI's cost-effective model"},
            "4": {"name": "Claude 3.5 Haiku", "id": "claude-3-5-haiku-20241022", "description": "Anthropic's fast and efficient model"},
            "5": {"name": "GPT-4 Turbo", "id": "gpt-4-turbo", "description": "OpenAI's previous flagship model"},
        }
        
        print("\n" + "="*60)
        print("🤖 MODEL SELECTION")
        print("="*60)
        
        for key, value in models.items():
            marker = " (default)" if value["id"] == default_model else ""
            print(f"{key}. {value['name']}{marker}")
            print(f"   ID: {value['id']}")
            print(f"   Description: {value['description']}")
            print()
        
        print(f"{len(models)+1}. Custom Model")
        print("   Specify your own model ID")
        print("\n" + "="*60)
        
        while True:
            try:
                choice = input(f"\nSelect model (1-{len(models)+1}, or press Enter for default): ").strip()
                
                # Default selection
                if not choice:
                    print(f"✓ Using default model: {default_model}")
                    return default_model
                
                if choice in models:
                    selected = models[choice]
                    print(f"\n✓ Selected: {selected['name']}")
                    print(f"Model ID: {selected['id']}")
                    return selected['id']
                    
                elif choice == str(len(models) + 1):
                    custom_model = input("\nEnter custom model ID: ").strip()
                    if custom_model:
                        print(f"\n✓ Custom model: {custom_model}")
                        return custom_model
                    else:
                        print("⚠ Model ID cannot be empty.")
                
                else:
                    print(f"⚠ Invalid choice. Please enter 1-{len(models)+1} or press Enter for default")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                sys.exit(0)
            except EOFError:
                print("\n\n👋 Goodbye!")
                sys.exit(0)
    
    def get_num_ideas_selection(self, default_num: int) -> int:
        """Get number of ideas from user input."""
        print("\n" + "="*60)
        print("💡 NUMBER OF IDEAS")
        print("="*60)
        print("How many different research ideas should AI-Scientist generate?")
        print("More ideas = more diverse approaches but longer runtime")
        print()
        print("Recommendations:")
        print("• 1 idea: Quick single experiment (~15-30 min)")
        print("• 2-3 ideas: Compare different approaches (~30-90 min)")
        print("• 4-5 ideas: Comprehensive exploration (~2-4 hours)")
        print("="*60)
        
        while True:
            try:
                choice = input(f"\nEnter number of ideas (1-10, or press Enter for default {default_num}): ").strip()
                
                # Default selection
                if not choice:
                    print(f"✓ Using default: {default_num} idea(s)")
                    return default_num
                
                try:
                    num_ideas = int(choice)
                    if 1 <= num_ideas <= 10:
                        print(f"✓ Selected: {num_ideas} idea(s)")
                        return num_ideas
                    else:
                        print("⚠ Please enter a number between 1 and 10")
                except ValueError:
                    print("⚠ Please enter a valid number")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                sys.exit(0)
            except EOFError:
                print("\n\n👋 Goodbye!")
                sys.exit(0)
    
    def create_dynamic_prompt(self, policy_direction: str) -> Dict[str, Any]:
        """Create dynamic prompt based on user's policy direction."""
        
        # Dynamic mission statement based on user input
        mission = (f"Your mission is to develop novel, creative, and scientifically rigorous approaches "
                  f"focusing on {policy_direction}. Think beyond conventional approaches - Japan needs "
                  f"breakthrough solutions to one of the world's most severe demographic challenges. "
                  f"Focus on generating actionable insights that could realistically help Japan reverse "
                  f"population decline while maintaining economic sustainability and social equity.")
        
        # Combine all parts
        task_description = f"{self.base_framework} {mission} {self.base_conclusion}"
        
        return {
            "system": self.base_system,
            "task_description": task_description
        }
    
    def save_prompt(self, prompt_data: Dict[str, Any]):
        """Save the dynamic prompt to prompt.json."""
        with open(self.prompt_file, 'w') as f:
            json.dump(prompt_data, f, indent=4)
        print(f"✓ Saved dynamic prompt to {self.prompt_file}")
    
    def run_ai_scientist(self, model: str = "claude-3-5-sonnet-20241022", num_ideas: int = 1):
        """Run AI-Scientist with the dynamic prompt."""
        print(f"\n🚀 Running AI-Scientist with {model}...")
        print(f"Template: population_policy")
        print(f"Number of ideas: {num_ideas}")
        print("="*60)
        
        # Construct command
        cmd = [
            sys.executable, "launch_scientist.py",
            "--model", model,
            "--experiment", "population_policy", 
            "--num-ideas", str(num_ideas)
        ]
        
        try:
            # Run from the AI-Scientist root directory
            ai_scientist_root = Path(__file__).parent
            print(f"📝 AI-Scientist output will be displayed in real-time below...")
            print(f"💡 You can also check detailed logs in the generated run directories")
            print("="*60)
            
            result = subprocess.run(cmd, cwd=ai_scientist_root, check=True)
            print("="*60)
            print(f"✅ AI-Scientist completed successfully!")
            return True
            
        except subprocess.CalledProcessError as e:
            print("="*60)
            print(f"❌ AI-Scientist failed with error code {e.returncode}")
            print("💡 Check the output above for error details.")
            return False
        except FileNotFoundError:
            print(f"❌ Could not find launch_scientist.py")
            print("Make sure you're running this from the AI-Scientist root directory.")
            return False
    
    def _display_welcome_message(self):
        """Display welcome message and instructions."""
        print("🧬 INTERACTIVE POPULATION POLICY EXPLORER")
        print("=" * 60)
        print("This tool allows you to dynamically explore population policy")
        print("research directions using AI-Scientist.")
        print()
        print("The system will:")
        print("1. Let you configure model and number of ideas")  
        print("2. Let you specify a policy research direction")  
        print("3. Generate a custom prompt for AI-Scientist")
        print("4. Run experiments with your specified focus")
        print("5. Generate comprehensive analysis plots")
    
    def _cleanup_prompt_if_needed(self, policy_direction):
        """Clean up prompt modifications if they were made."""
        if policy_direction is not None:
            print("\n🔄 Restoring original prompt.json...")
            self.restore_original_prompt()
        else:
            print("\n💡 No prompt modifications were made.")
    
    def _get_confirmation(self, message: str) -> bool:
        """Get confirmation from user with consistent error handling."""
        try:
            response = input(f"\n{message} (y/n): ").strip().lower()
            return response in ['y', 'yes']
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Goodbye!")
            sys.exit(0)

    def run_interactive_session(self, default_model: str = "claude-3-5-sonnet-20241022", 
                               default_num_ideas: int = 1):
        """Run complete interactive session."""
        self._display_welcome_message()
        
        # Backup original prompt
        self.backup_original_prompt()
        
        try:
            # Get interactive selections
            model = self.get_model_selection(default_model)
            num_ideas = self.get_num_ideas_selection(default_num_ideas)
            
            # Get policy direction from user
            self.display_policy_options()
            policy_direction = self.get_user_policy_direction()
            
            # Handle prompt modification
            if policy_direction is None:
                print(f"\n✓ Using original prompt.json (no modifications)")
                task_focus = "original research directions"
            else:
                # Create and save dynamic prompt
                print(f"\n🔄 Creating dynamic prompt...")
                prompt_data = self.create_dynamic_prompt(policy_direction)
                self.save_prompt(prompt_data)
                task_focus = policy_direction
                
                print(f"\n📋 Generated Task Focus:")
                print(f"'{policy_direction}'")
            
            # Confirm before running
            print(f"\n🎯 Ready to run AI-Scientist experiment:")
            print(f"  Model: {model}")
            print(f"  Template: population_policy") 
            print(f"  Number of ideas: {num_ideas}")
            print(f"  Focus: {task_focus}")
            
            if not self._get_confirmation("Proceed with experiment?"):
                print("❌ Experiment cancelled.")
                return
            
            # Run AI-Scientist
            success = self.run_ai_scientist(model, num_ideas)

        except KeyboardInterrupt:
            print(f"\n\n🛑 Session interrupted by user.")
        finally:
            self._cleanup_prompt_if_needed(locals().get('policy_direction'))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Interactive Population Policy Explorer for AI-Scientist",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python interactive_population_policy.py
  python interactive_population_policy.py --model gpt-4o --num-ideas 2
        """
    )
    
    parser.add_argument(
        "--model", 
        default="claude-3-5-sonnet-20241022",
        help="Default model for AI-Scientist (can be changed interactively)"
    )
    
    parser.add_argument(
        "--num-ideas",
        type=int, 
        default=1,
        help="Default number of ideas to generate (can be changed interactively)"
    )
    
    
    args = parser.parse_args()
    
    # Check if we're in the right directory
    if not Path("launch_scientist.py").exists():
        print("❌ Error: launch_scientist.py not found.")
        print("Please run this script from the AI-Scientist root directory.")
        sys.exit(1)
    
    if not Path("templates/population_policy").exists():
        print(f"❌ Error: Template directory not found: templates/population_policy")
        sys.exit(1)
    
    # Create and run interactive session
    explorer = InteractivePopulationPolicy()
    explorer.run_interactive_session(
        default_model=args.model,
        default_num_ideas=args.num_ideas
    )


if __name__ == "__main__":
    main()
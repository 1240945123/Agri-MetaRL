#!/usr/bin/env python3
"""
Basic visualization script for GreenLight-Gym2 results
Generates simple plots using available data
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set matplotlib parameters
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (12, 8)

def load_data(filepath):
    """Load CSV data"""
    try:
        # Try relative path first
        if os.path.exists(filepath):
            return pd.read_csv(filepath)
        # Try absolute path from project root
        abs_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), filepath)
        if os.path.exists(abs_path):
            return pd.read_csv(abs_path)
        print(f"File not found: {filepath}")
        print(f"Also tried: {abs_path}")
        return None
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def plot_trajectories():
    """Plot state trajectories comparison"""
    print("Generating trajectory plots...")
    
    # Load data
    ppo_file = "data/AgriControl/deterministic/ppo/dummy-cg4axdls-201059-Amsterdam.csv"
    rb_file = "data/AgriControl/deterministic/rb_baseline/rb_baseline-201059-Amsterdam.csv"
    
    ppo_data = load_data(ppo_file)
    rb_data = load_data(rb_file)
    
    if ppo_data is None or rb_data is None:
        print("Cannot generate trajectory plots - data files not found")
        return
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('State Trajectories: PPO vs Rule-Based Controller', fontsize=16)
    
    # Plot temperature
    axes[0,0].plot(ppo_data['temp_air'], label='PPO', alpha=0.8)
    axes[0,0].plot(rb_data['temp_air'], label='Rule-Based', alpha=0.8)
    axes[0,0].set_title('Air Temperature (°C)')
    axes[0,0].set_ylabel('Temperature (°C)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot humidity
    axes[0,1].plot(ppo_data['rh_air'], label='PPO', alpha=0.8)
    axes[0,1].plot(rb_data['rh_air'], label='Rule-Based', alpha=0.8)
    axes[0,1].set_title('Relative Humidity (%)')
    axes[0,1].set_ylabel('Humidity (%)')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot CO2
    axes[1,0].plot(ppo_data['co2_air'], label='PPO', alpha=0.8)
    axes[1,0].plot(rb_data['co2_air'], label='Rule-Based', alpha=0.8)
    axes[1,0].set_title('CO2 Concentration (ppm)')
    axes[1,0].set_ylabel('CO2 (ppm)')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot cumulative rewards
    axes[1,1].plot(ppo_data['Rewards'].cumsum(), label='PPO', alpha=0.8)
    axes[1,1].plot(rb_data['Rewards'].cumsum(), label='Rule-Based', alpha=0.8)
    axes[1,1].set_title('Cumulative Rewards')
    axes[1,1].set_ylabel('Cumulative Reward')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/trajectory_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: plots/trajectory_comparison.png")

def plot_performance_metrics():
    """Plot performance metrics comparison"""
    print("Generating performance metrics plots...")
    
    # Load data
    ppo_file = "data/AgriControl/deterministic/ppo/dummy-cg4axdls-201059-Amsterdam.csv"
    rb_file = "data/AgriControl/deterministic/rb_baseline/rb_baseline-201059-Amsterdam.csv"
    
    ppo_data = load_data(ppo_file)
    rb_data = load_data(rb_file)
    
    if ppo_data is None or rb_data is None:
        print("Cannot generate performance plots - data files not found")
        return
    
    # Calculate final metrics
    ppo_metrics = {
        'Total Reward': ppo_data['Rewards'].sum(),
        'Total Revenue': ppo_data['Revenue'].sum(),
        'Total Heat Cost': ppo_data['Heat costs'].sum(),
        'Total CO2 Cost': ppo_data['CO2 costs'].sum(),
        'Total Elec Cost': ppo_data['Elec costs'].sum(),
        'Temp Violations': ppo_data['temp_violation'].sum(),
        'CO2 Violations': ppo_data['co2_violation'].sum(),
        'RH Violations': ppo_data['rh_violation'].sum()
    }
    
    rb_metrics = {
        'Total Reward': rb_data['Rewards'].sum(),
        'Total Revenue': rb_data['Revenue'].sum(),
        'Total Heat Cost': rb_data['Heat costs'].sum(),
        'Total CO2 Cost': rb_data['CO2 costs'].sum(),
        'Total Elec Cost': rb_data['Elec costs'].sum(),
        'Temp Violations': rb_data['temp_violation'].sum(),
        'CO2 Violations': rb_data['co2_violation'].sum(),
        'RH Violations': rb_data['rh_violation'].sum()
    }
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Performance Metrics: PPO vs Rule-Based Controller', fontsize=16)
    
    # Economic metrics
    economic_metrics = ['Total Reward', 'Total Revenue', 'Total Heat Cost', 'Total CO2 Cost', 'Total Elec Cost']
    ppo_economic = [ppo_metrics[m] for m in economic_metrics]
    rb_economic = [rb_metrics[m] for m in economic_metrics]
    
    x = np.arange(len(economic_metrics))
    width = 0.35
    
    axes[0,0].bar(x - width/2, ppo_economic, width, label='PPO', alpha=0.8)
    axes[0,0].bar(x + width/2, rb_economic, width, label='Rule-Based', alpha=0.8)
    axes[0,0].set_title('Economic Metrics')
    axes[0,0].set_ylabel('Value')
    axes[0,0].set_xticks(x)
    axes[0,0].set_xticklabels(economic_metrics, rotation=45, ha='right')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Constraint violations
    violation_metrics = ['Temp Violations', 'CO2 Violations', 'RH Violations']
    ppo_violations = [ppo_metrics[m] for m in violation_metrics]
    rb_violations = [rb_metrics[m] for m in violation_metrics]
    
    x = np.arange(len(violation_metrics))
    axes[0,1].bar(x - width/2, ppo_violations, width, label='PPO', alpha=0.8)
    axes[0,1].bar(x + width/2, rb_violations, width, label='Rule-Based', alpha=0.8)
    axes[0,1].set_title('Constraint Violations')
    axes[0,1].set_ylabel('Violation Count')
    axes[0,1].set_xticks(x)
    axes[0,1].set_xticklabels(violation_metrics, rotation=45, ha='right')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Performance summary
    summary_data = {
        'PPO': [ppo_metrics['Total Reward'], ppo_metrics['Total Revenue'], 
                ppo_metrics['Temp Violations'] + ppo_metrics['CO2 Violations'] + ppo_metrics['RH Violations']],
        'Rule-Based': [rb_metrics['Total Reward'], rb_metrics['Total Revenue'],
                      rb_metrics['Temp Violations'] + rb_metrics['CO2 Violations'] + rb_metrics['RH Violations']]
    }
    
    summary_labels = ['Total Reward', 'Total Revenue', 'Total Violations']
    x = np.arange(len(summary_labels))
    
    axes[1,0].bar(x - width/2, summary_data['PPO'], width, label='PPO', alpha=0.8)
    axes[1,0].bar(x + width/2, summary_data['Rule-Based'], width, label='Rule-Based', alpha=0.8)
    axes[1,0].set_title('Performance Summary')
    axes[1,0].set_ylabel('Value')
    axes[1,0].set_xticks(x)
    axes[1,0].set_xticklabels(summary_labels, rotation=45, ha='right')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Improvement percentage
    improvements = []
    for metric in ['Total Reward', 'Total Revenue']:
        if rb_metrics[metric] != 0:
            improvement = ((ppo_metrics[metric] - rb_metrics[metric]) / abs(rb_metrics[metric])) * 100
            improvements.append(improvement)
        else:
            improvements.append(0)
    
    axes[1,1].bar(['Total Reward', 'Total Revenue'], improvements, alpha=0.8, color='green')
    axes[1,1].set_title('PPO Improvement over Rule-Based (%)')
    axes[1,1].set_ylabel('Improvement (%)')
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: plots/performance_comparison.png")

def plot_uncertainty_analysis():
    """Plot uncertainty analysis for PPO"""
    print("Generating uncertainty analysis plots...")
    
    uncertainty_scales = ['0.0', '0.05', '0.1', '0.15', '0.2', '0.25', '0.3']
    results = {}
    
    # Load data for each uncertainty scale
    for scale in uncertainty_scales:
        filepath = f"data/AgriControl/stochastic/ppo/{scale}/dummy-nbw883du-201059-Amsterdam.csv"
        data = load_data(filepath)
        if data is not None:
            results[scale] = {
                'Total Reward': data['Rewards'].sum(),
                'Total Revenue': data['Revenue'].sum(),
                'Total Violations': data['temp_violation'].sum() + data['co2_violation'].sum() + data['rh_violation'].sum()
            }
    
    if not results:
        print("Cannot generate uncertainty analysis - no stochastic data found")
        return
    
    # Create uncertainty analysis plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('PPO Performance under Parameter Uncertainty', fontsize=16)
    
    scales = list(results.keys())
    scales_float = [float(s) for s in scales]
    
    # Total Reward
    rewards = [results[s]['Total Reward'] for s in scales]
    axes[0].plot(scales_float, rewards, 'o-', linewidth=2, markersize=8)
    axes[0].set_title('Total Reward vs Uncertainty')
    axes[0].set_xlabel('Uncertainty Scale')
    axes[0].set_ylabel('Total Reward')
    axes[0].grid(True, alpha=0.3)
    
    # Total Revenue
    revenues = [results[s]['Total Revenue'] for s in scales]
    axes[1].plot(scales_float, revenues, 'o-', linewidth=2, markersize=8, color='green')
    axes[1].set_title('Total Revenue vs Uncertainty')
    axes[1].set_xlabel('Uncertainty Scale')
    axes[1].set_ylabel('Total Revenue')
    axes[1].grid(True, alpha=0.3)
    
    # Total Violations
    violations = [results[s]['Total Violations'] for s in scales]
    axes[2].plot(scales_float, violations, 'o-', linewidth=2, markersize=8, color='red')
    axes[2].set_title('Total Violations vs Uncertainty')
    axes[2].set_xlabel('Uncertainty Scale')
    axes[2].set_ylabel('Total Violations')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/uncertainty_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: plots/uncertainty_analysis.png")

def main():
    """Main function to generate all plots"""
    print("=" * 50)
    print("Generating GreenLight-Gym2 Visualization Results")
    print("=" * 50)
    
    # Get the project root directory
    project_root = os.path.dirname(os.path.dirname(__file__))
    os.chdir(project_root)
    
    # Create output directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)
    
    # Generate plots
    plot_trajectories()
    plot_performance_metrics()
    plot_uncertainty_analysis()
    
    print("\n" + "=" * 50)
    print("Visualization generation completed!")
    print("=" * 50)
    print("Generated plots:")
    print("- trajectory_comparison.png")
    print("- performance_comparison.png") 
    print("- uncertainty_analysis.png")
    print(f"\nAll plots saved in: {os.path.join(project_root, 'plots')}")

if __name__ == "__main__":
    main()

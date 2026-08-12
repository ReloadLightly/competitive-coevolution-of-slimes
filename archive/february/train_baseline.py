"""
train_baseline.py - Paper-correct baseline training
Fixes:
  1. Action threshold 0.5 (steepened sigmoid outputs [0,1])
  2. Calls _update_stagnation() (was missing)
  3. Fitness rewards activity, not passivity
  4. Uses paper-correct config (pop=150, 200 gens)
"""

import numpy as np
import os
import time
import pickle
from datetime import datetime
from typing import List
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import gym
import slimevolleygym

from neat.genome import NEATGenome
from neat.population import NEATPopulation
from neat.config import NEATConfig


def genome_to_action(genome: NEATGenome, obs: np.ndarray) -> List[int]:
    """Convert network output to discrete actions.
    
    With steepened sigmoid (outputs [0,1]), threshold at 0.5.
    With tanh (outputs [-1,1]), threshold at 0.
    """
    outputs = genome.forward(obs)
    # Steepened sigmoid outputs [0, 1], so threshold at 0.5
    return [1 if o > 0.5 else 0 for o in outputs]


def evaluate_game(genome, env, max_steps=1500):
    """Evaluate one game vs the internal AI.
    
    Fitness design principles:
    - Primary signal: scoring points and winning (sparse but meaningful)
    - Shaping: reward MOVING and ENGAGING, not standing still
    - No survival bonus (was rewarding passivity)
    """
    env.policy = None  # Use internal AI
    obs = env.reset()
    
    fitness = 0.0
    my_score = 0
    opp_score = 0
    total_movement = 0
    steps = 0
    
    for step in range(max_steps):
        action = genome_to_action(genome, obs)
        obs, reward, done, _ = env.step(action)
        steps += 1
        
        # === Primary rewards: scoring ===
        if reward > 0:
            my_score += 1
            fitness += 10.0
        elif reward < 0:
            opp_score += 1
            fitness -= 3.0
        
        # === Shaping: reward engagement, not passivity ===
        # Track if agent is actually doing things (any non-zero action)
        if sum(action) > 0:
            total_movement += 1
        
        # Ball on opponent's side is good (but small)
        ball_x = obs[4]
        if ball_x > 0:
            fitness += 0.003
        
        if done:
            break
    
    # === Win bonus (big) ===
    if my_score > opp_score:
        fitness += 20.0
    
    # === Activity bonus: reward genomes that actually move ===
    # This prevents evolution from favoring "stand still" strategies
    activity_rate = total_movement / max(steps, 1)
    fitness += activity_rate * 2.0  # Up to +2 for being active
    
    # === Penalty for pure passivity ===
    if total_movement < steps * 0.1:
        fitness -= 5.0  # Penalize agents that barely move
    
    return fitness, my_score, opp_score


def evaluate_genome(genome, env, num_games=3):
    """Evaluate genome over multiple games vs internal AI."""
    total_fitness = 0.0
    total_wins = 0
    total_pf = 0
    total_pa = 0
    
    for _ in range(num_games):
        fit, my_score, opp_score = evaluate_game(genome, env)
        total_fitness += fit
        total_pf += my_score
        total_pa += opp_score
        if my_score > opp_score:
            total_wins += 1
    
    return total_fitness / num_games, total_wins, total_pf, total_pa


def debug_genome(genome, env):
    """Detailed debug of best genome's behavior."""
    env.policy = None
    obs = env.reset()
    actions = []
    my_score = 0
    opp_score = 0
    
    for _ in range(1500):
        out = genome.forward(obs)
        action = [1 if o > 0.5 else 0 for o in out]  # Match threshold
        actions.append(action)
        obs, reward, done, _ = env.step(action)
        if reward > 0:
            my_score += 1
        elif reward < 0:
            opp_score += 1
        if done:
            break
    
    actions = np.array(actions)
    hidden = len([n for n in genome.nodes.values() if n.node_type == 'hidden'])
    
    print(f"     Net: {genome.get_num_connections_enabled()} conns, {hidden} hidden")
    print(f"     vs Internal AI: {my_score}-{opp_score} | "
          f"L={actions[:,0].sum()} R={actions[:,1].sum()} J={actions[:,2].sum()} "
          f"({len(actions)} steps)")


def plot_progress(history, filepath):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0,0].plot(history['best_fitness'], 'b-', linewidth=2)
    axes[0,0].set_title('Best Fitness')
    axes[0,0].set_xlabel('Generation')
    axes[0,0].grid(True, alpha=0.3)
    
    axes[0,1].plot(history['avg_fitness'], 'g-', linewidth=2)
    axes[0,1].set_title('Average Fitness')
    axes[0,1].set_xlabel('Generation')
    axes[0,1].grid(True, alpha=0.3)
    
    axes[1,0].plot(history['species'], 'r-', linewidth=2)
    axes[1,0].set_title('Species Count')
    axes[1,0].set_xlabel('Generation')
    axes[1,0].grid(True, alpha=0.3)
    
    axes[1,1].plot(history['win_rate'], 'm-', linewidth=2)
    axes[1,1].set_title('Win Rate vs Internal AI (%)')
    axes[1,1].set_xlabel('Generation')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()


def train():
    config = NEATConfig
    
    print("=" * 60)
    print("BASELINE TRAINING - Paper-Correct Settings")
    print("=" * 60)
    print(f"Population: {config.POPULATION_SIZE}")
    print(f"Generations: {config.MAX_GENERATIONS}")
    print(f"Games per eval: {config.GAMES_PER_EVALUATION}")
    print(f"Activation: {config.ACTIVATION_FUNCTION}")
    print(f"Strategy: Evolve vs internal AI, no curriculum")
    print("=" * 60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(config.SAVE_DIR, f"baseline_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    print(f"\nSaving to: {exp_dir}\n")
    
    env = gym.make("SlimeVolley-v0")
    
    print("Creating population...")
    pop = NEATPopulation(
        num_inputs=config.NUM_INPUTS,
        num_outputs=config.NUM_OUTPUTS,
        population_size=config.POPULATION_SIZE,
        compatibility_threshold=config.COMPATIBILITY_THRESHOLD,
        mutation_add_node_prob=config.MUTATION_ADD_NODE_PROB,
        mutation_add_connection_prob=config.MUTATION_ADD_CONNECTION_PROB,
        mutation_weight_prob=config.MUTATION_WEIGHT_PROB,
    )
    
    history = {
        'best_fitness': [], 'avg_fitness': [],
        'species': [], 'win_rate': [],
        'points_for': [], 'points_against': []
    }
    
    print("\n" + "-" * 60)
    start = time.time()
    
    for gen in range(config.MAX_GENERATIONS):
        t0 = time.time()
        
        total_wins = 0
        total_games = 0
        total_pf = 0
        total_pa = 0
        
        # === Evaluate all genomes ===
        for genome in pop.genomes:
            genome.fitness, wins, pf, pa = evaluate_genome(
                genome, env, config.GAMES_PER_EVALUATION
            )
            total_wins += wins
            total_games += config.GAMES_PER_EVALUATION
            total_pf += pf
            total_pa += pa
        
        win_rate = total_wins / total_games * 100
        
        best_genome = max(pop.genomes, key=lambda g: g.fitness)
        best_fit = best_genome.fitness
        avg_fit = np.mean([g.fitness for g in pop.genomes])
        
        # === CRITICAL: Use proper NEAT lifecycle ===
        # Speciate
        pop._speciate()
        
        # Adjusted fitness
        for sp in pop.species_list:
            sp.calculate_adjusted_fitness()
        num_species = len(pop.species_list)
        
        # Stagnation check (WAS MISSING - this enables the 20-gen refocus)
        pop._update_stagnation()
        
        # Reproduce
        pop.genomes = pop._reproduce()
        pop.generation += 1
        
        # === Track history ===
        history['best_fitness'].append(best_fit)
        history['avg_fitness'].append(avg_fit)
        history['species'].append(num_species)
        history['win_rate'].append(win_rate)
        history['points_for'].append(total_pf)
        history['points_against'].append(total_pa)
        
        elapsed = time.time() - t0
        total_time = (time.time() - start) / 60
        
        print(f"Gen {gen+1:3d} | best={best_fit:+.2f} avg={avg_fit:+.2f} | "
              f"wins={win_rate:.0f}% ({total_pf}-{total_pa}) | "
              f"sp={num_species} | {elapsed:.0f}s ({total_time:.1f}m)")
        
        # Debug best genome periodically
        if (gen + 1) % 10 == 0 or gen == 0:
            debug_genome(best_genome, env)
        
        # Checkpoints
        if (gen + 1) % config.SAVE_BEST_EVERY == 0:
            ckpt = os.path.join(exp_dir, f"gen_{gen+1}")
            os.makedirs(ckpt, exist_ok=True)
            with open(os.path.join(ckpt, "best_genome.pkl"), "wb") as f:
                pickle.dump(best_genome, f)
            with open(os.path.join(ckpt, "population.pkl"), "wb") as f:
                pickle.dump(pop, f)
            # Save intermediate plot
            plot_progress(history, os.path.join(exp_dir, "progress.png"))
            print(f"       | Saved checkpoint + plot")
    
    total_time = (time.time() - start) / 60
    print("-" * 60)
    print(f"COMPLETE | Time: {total_time:.1f}m | Best: {history['best_fitness'][-1]:+.2f}")
    print(f"         | Final win rate: {history['win_rate'][-1]:.0f}%")
    print("-" * 60)
    
    # Save final results
    final = os.path.join(exp_dir, "final")
    os.makedirs(final, exist_ok=True)
    with open(os.path.join(final, "best_genome.pkl"), "wb") as f:
        pickle.dump(best_genome, f)
    with open(os.path.join(final, "population.pkl"), "wb") as f:
        pickle.dump(pop, f)
    with open(os.path.join(final, "history.pkl"), "wb") as f:
        pickle.dump(history, f)
    
    plot_progress(history, os.path.join(exp_dir, "progress.png"))
    print(f"\nPlot: {exp_dir}/progress.png")
    
    env.close()


if __name__ == "__main__":
    train()
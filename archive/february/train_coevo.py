"""
train_coevo.py - Coevolutionary NEAT Training for SlimeVolley
Uses the REAL internal AI as baseline opponent
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
    outputs = genome.forward(obs)
    return [1 if o > 0 else 0 for o in outputs]


class GenomePolicy:
    def __init__(self, genome: NEATGenome):
        self.genome = genome
    
    def predict(self, obs: np.ndarray) -> List[int]:
        return genome_to_action(self.genome, obs)


def evaluate_game(genome, opponent_policy, env, max_steps=1500):
    env.policy = opponent_policy
    obs = env.reset()
    
    fitness = 0.0
    my_score = 0
    opp_score = 0
    
    for step in range(max_steps):
        action = genome_to_action(genome, obs)
        obs, reward, done, _ = env.step(action)
        
        if reward > 0:
            my_score += 1
        elif reward < 0:
            opp_score += 1
        
        if obs[4] > 0:
            fitness += 0.001
        fitness += 0.0001
        
        if done:
            break
    
    if my_score > opp_score:
        fitness += 5.0
    elif my_score < opp_score:
        fitness -= 5.0
    
    return fitness, my_score, opp_score


def evaluate_genome(genome, opponents, env, games=2, baseline_weight=0.5):
    total = 0.0
    count = 0
    wins_vs_internal = 0
    games_vs_internal = 0
    
    if baseline_weight < 1.0 and opponents:
        for opp in opponents:
            for _ in range(games):
                fit, _, _ = evaluate_game(genome, GenomePolicy(opp), env)
                total += fit * (1 - baseline_weight)
                count += 1
    
    if baseline_weight > 0:
        num_baseline = max(1, len(opponents) if opponents else 1) * games
        for _ in range(num_baseline):
            fit, my_score, opp_score = evaluate_game(genome, None, env)
            total += fit * baseline_weight
            count += 1
            games_vs_internal += 1
            if my_score > opp_score:
                wins_vs_internal += 1
    
    return total / max(1, count), wins_vs_internal, games_vs_internal


def debug_genome(genome, env):
    env.policy = None
    obs = env.reset()
    actions = []
    my_score = 0
    opp_score = 0
    
    for _ in range(1500):
        out = genome.forward(obs)
        action = [1 if o > 0 else 0 for o in out]
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
    print(f"     vs Internal AI: {my_score}-{opp_score} | L={actions[:,0].sum()} R={actions[:,1].sum()} J={actions[:,2].sum()}")


def plot_progress(history, filepath):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0,0].plot(history['left_best'], 'b-', label='Left')
    axes[0,0].plot(history['right_best'], 'r-', label='Right')
    axes[0,0].set_title('Best Fitness')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    axes[0,1].plot(history['left_avg'], 'b-', label='Left')
    axes[0,1].plot(history['right_avg'], 'r-', label='Right')
    axes[0,1].set_title('Average Fitness')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    axes[1,0].plot(history['left_species'], 'b-', label='Left')
    axes[1,0].plot(history['right_species'], 'r-', label='Right')
    axes[1,0].set_title('Species Count')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    axes[1,1].plot(history['win_rate'], 'g-', linewidth=2)
    axes[1,1].set_title('Win Rate vs Internal AI')
    axes[1,1].set_ylabel('Win %')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()


def train():
    config = NEATConfig
    config.display()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(config.SAVE_DIR, f"coevo_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    print(f"\n[Saving to: {exp_dir}]")
    print("[Training against INTERNAL AI]\n")
    
    env = gym.make("SlimeVolley-v0")
    
    print("Creating populations...")
    left_pop = NEATPopulation(
        num_inputs=config.NUM_INPUTS, num_outputs=config.NUM_OUTPUTS,
        population_size=config.POPULATION_SIZE,
        compatibility_threshold=config.COMPATIBILITY_THRESHOLD,
        mutation_add_node_prob=config.MUTATION_ADD_NODE_PROB,
        mutation_add_connection_prob=config.MUTATION_ADD_CONNECTION_PROB,
        mutation_weight_prob=config.MUTATION_WEIGHT_PROB,
    )
    right_pop = NEATPopulation(
        num_inputs=config.NUM_INPUTS, num_outputs=config.NUM_OUTPUTS,
        population_size=config.POPULATION_SIZE,
        compatibility_threshold=config.COMPATIBILITY_THRESHOLD,
        mutation_add_node_prob=config.MUTATION_ADD_NODE_PROB,
        mutation_add_connection_prob=config.MUTATION_ADD_CONNECTION_PROB,
        mutation_weight_prob=config.MUTATION_WEIGHT_PROB,
    )
    
    history = {
        'left_best': [], 'left_avg': [], 'left_species': [],
        'right_best': [], 'right_avg': [], 'right_species': [],
        'baseline_weight': [], 'win_rate': []
    }
    
    print("\n" + "=" * 75)
    print("TRAINING START")
    print("=" * 75)
    
    start = time.time()
    
    for gen in range(config.MAX_GENERATIONS):
        t0 = time.time()
        baseline_weight = max(0.3, 0.7 - gen * 0.02)
        
        total_wins = 0
        total_games = 0
        
        opps = np.random.choice(right_pop.genomes, min(config.COEVO_SAMPLE_SIZE, len(right_pop.genomes)), replace=False).tolist()
        for g in left_pop.genomes:
            g.fitness, wins, games = evaluate_genome(g, opps, env, config.GAMES_PER_EVALUATION, baseline_weight)
            total_wins += wins
            total_games += games
        
        opps = np.random.choice(left_pop.genomes, min(config.COEVO_SAMPLE_SIZE, len(left_pop.genomes)), replace=False).tolist()
        for g in right_pop.genomes:
            g.fitness, wins, games = evaluate_genome(g, opps, env, config.GAMES_PER_EVALUATION, baseline_weight)
            total_wins += wins
            total_games += games
        
        win_rate = total_wins / max(1, total_games) * 100
        
        left_best_genome = max(left_pop.genomes, key=lambda g: g.fitness)
        right_best_genome = max(right_pop.genomes, key=lambda g: g.fitness)
        
        left_best = left_best_genome.fitness
        right_best = right_best_genome.fitness
        left_avg = np.mean([g.fitness for g in left_pop.genomes])
        right_avg = np.mean([g.fitness for g in right_pop.genomes])
        
        left_pop._speciate()
        for sp in left_pop.species_list:
            sp.calculate_adjusted_fitness()
        left_species = len(left_pop.species_list)
        left_pop.genomes = left_pop._reproduce()
        left_pop.generation += 1
        
        right_pop._speciate()
        for sp in right_pop.species_list:
            sp.calculate_adjusted_fitness()
        right_species = len(right_pop.species_list)
        right_pop.genomes = right_pop._reproduce()
        right_pop.generation += 1
        
        history['left_best'].append(left_best)
        history['left_avg'].append(left_avg)
        history['left_species'].append(left_species)
        history['right_best'].append(right_best)
        history['right_avg'].append(right_avg)
        history['right_species'].append(right_species)
        history['baseline_weight'].append(baseline_weight)
        history['win_rate'].append(win_rate)
        
        elapsed = time.time() - t0
        total = (time.time() - start) / 60
        
        print(f"Gen {gen+1:2d} | L: best={left_best:+.2f} avg={left_avg:+.2f} sp={left_species}")
        print(f"       | R: best={right_best:+.2f} avg={right_avg:+.2f} sp={right_species}")
        print(f"       | Win vs AI: {win_rate:.1f}% | {elapsed:.0f}s ({total:.1f}m)")
        
        if (gen + 1) % 5 == 0 or gen == 0:
            debug_genome(left_best_genome, env)
        
        print("-" * 75)
        
        if (gen + 1) % config.SAVE_BEST_EVERY == 0:
            ckpt = os.path.join(exp_dir, f"gen_{gen+1}")
            os.makedirs(ckpt, exist_ok=True)
            with open(os.path.join(ckpt, "best_left.pkl"), "wb") as f:
                pickle.dump(left_best_genome, f)
            with open(os.path.join(ckpt, "best_right.pkl"), "wb") as f:
                pickle.dump(right_best_genome, f)
            print(f"       | Checkpoint saved")
            print("-" * 75)
    
    total = (time.time() - start) / 60
    print("=" * 75)
    print(f"TRAINING COMPLETE")
    print(f"Time: {total:.1f} minutes")
    print(f"Left Best:  {history['left_best'][-1]:+.2f}")
    print(f"Right Best: {history['right_best'][-1]:+.2f}")
    print(f"Final Win Rate: {history['win_rate'][-1]:.1f}%")
    print("=" * 75)
    
    final = os.path.join(exp_dir, "final")
    os.makedirs(final, exist_ok=True)
    with open(os.path.join(final, "best_left.pkl"), "wb") as f:
        pickle.dump(left_best_genome, f)
    with open(os.path.join(final, "best_right.pkl"), "wb") as f:
        pickle.dump(right_best_genome, f)
    with open(os.path.join(final, "history.pkl"), "wb") as f:
        pickle.dump(history, f)
    
    plot_progress(history, os.path.join(exp_dir, "progress.png"))
    print(f"\nPlot saved: {exp_dir}/progress.png")
    
    env.close()
    print("\nDone!\n")


if __name__ == "__main__":
    train()
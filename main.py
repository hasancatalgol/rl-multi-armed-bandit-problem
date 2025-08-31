import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import random, seed
from random import randint
import csv

# ---- Config ----
BANDITS = 3
EPISODES = 5000
true_probs = [0.5, 0.6, 0.4]
optimal_reward = max(true_probs)

N_RUNS = 50
SEED = 42  # set to None for randomness
if SEED is not None:
    seed(SEED); np.random.seed(SEED)

os.makedirs("docs", exist_ok=True)

# ---- Bandit ----
class Bandit:
    def __init__(self, probability):
        self.q = 0
        self.k = 0
        self.probability = probability
        self.alpha = 1
        self.beta = 1
    def get_reward(self):
        return 1 if random() < self.probability else 0

# ---- Algorithms ----
class EpsilonGreedy:
    def __init__(self, epsilon, true_probs):
        self.epsilon = epsilon
        self.bandits = [Bandit(p) for p in true_probs]
        self.rewards = []
    def run(self):
        for _ in range(EPISODES):
            if random() < self.epsilon:
                bandit = self.bandits[randint(0, BANDITS-1)]
            else:
                bandit = max(self.bandits, key=lambda b: b.q)
            reward = bandit.get_reward()
            bandit.k += 1; bandit.q += (1/(1+bandit.k))*(reward-bandit.q)
            self.rewards.append(reward)

class DecayedEpsilonGreedy:
    def __init__(self, epsilon, decay, true_probs):
        self.epsilon0 = epsilon; self.decay = decay
        self.bandits = [Bandit(p) for p in true_probs]
        self.rewards = []
    def run(self):
        for t in range(EPISODES):
            eps_t = self.epsilon0 * (self.decay ** t)
            if random() < eps_t:
                bandit = self.bandits[randint(0, BANDITS-1)]
            else:
                bandit = max(self.bandits, key=lambda b: b.q)
            reward = bandit.get_reward()
            bandit.k += 1; bandit.q += (1/(1+bandit.k))*(reward-bandit.q)
            self.rewards.append(reward)

class Softmax:
    def __init__(self, tau, true_probs):
        self.tau = tau
        self.bandits = [Bandit(p) for p in true_probs]
        self.rewards = []
    def run(self):
        for _ in range(EPISODES):
            qs = np.array([b.q for b in self.bandits])
            exp_qs = np.exp((qs - np.max(qs)) / self.tau)
            probs = exp_qs / exp_qs.sum()
            choice = np.random.choice(len(self.bandits), p=probs)
            bandit = self.bandits[choice]
            reward = bandit.get_reward()
            bandit.k += 1; bandit.q += (1/(1+bandit.k))*(reward-bandit.q)
            self.rewards.append(reward)

class AnnealedSoftmax:
    def __init__(self, tau0, decay, true_probs):
        self.tau0 = tau0; self.decay = decay
        self.bandits = [Bandit(p) for p in true_probs]
        self.rewards = []
    def run(self):
        for t in range(EPISODES):
            tau_t = max(0.01, self.tau0 * (self.decay ** t))  # avoid divide by 0
            qs = np.array([b.q for b in self.bandits])
            exp_qs = np.exp((qs - np.max(qs)) / tau_t)
            probs = exp_qs / exp_qs.sum()
            choice = np.random.choice(len(self.bandits), p=probs)
            bandit = self.bandits[choice]
            reward = bandit.get_reward()
            bandit.k += 1; bandit.q += (1/(1+bandit.k))*(reward-bandit.q)
            self.rewards.append(reward)

class UCB1:
    def __init__(self, true_probs):
        self.bandits = [Bandit(p) for p in true_probs]; self.rewards = []
    def run(self):
        for t in range(1, EPISODES+1):
            if t <= BANDITS: bandit = self.bandits[t-1]
            else:
                ucb = [b.q + np.sqrt(2*np.log(t)/(b.k+1e-6)) for b in self.bandits]
                bandit = self.bandits[np.argmax(ucb)]
            reward = bandit.get_reward()
            bandit.k += 1; bandit.q += (1/(1+bandit.k))*(reward-bandit.q)
            self.rewards.append(reward)

class ThompsonSampling:
    def __init__(self, true_probs):
        self.bandits = [Bandit(p) for p in true_probs]; self.rewards = []
    def run(self):
        for _ in range(EPISODES):
            samples = [np.random.beta(b.alpha,b.beta) for b in self.bandits]
            bandit = self.bandits[np.argmax(samples)]
            reward = bandit.get_reward()
            if reward: bandit.alpha += 1
            else: bandit.beta += 1
            bandit.k += 1; bandit.q += (1/(1+bandit.k))*(reward-bandit.q)
            self.rewards.append(reward)

# ---- Averaging ----
def run_avg(strategy_class, **kwargs):
    all_rewards, all_regrets, all_counts = [], [], []
    for _ in range(N_RUNS):
        algo = strategy_class(**kwargs, true_probs=true_probs) if kwargs else strategy_class(true_probs)
        algo.run()
        all_rewards.append(np.cumsum(algo.rewards)/(np.arange(len(algo.rewards))+1))
        all_regrets.append(np.cumsum(optimal_reward - np.array(algo.rewards)))
        all_counts.append([b.k for b in algo.bandits])
    return {
        "avg_rewards": np.mean(all_rewards, axis=0),
        "regret": np.mean(all_regrets, axis=0),
        "counts": np.mean(all_counts, axis=0)
    }

# ---- Collect Results for Main Strategies ----
strategies = {
    "epsilon_greedy": (EpsilonGreedy, {"epsilon":0.1}, "ε-Greedy (0.1)","blue"),
    "decayed_epsilon": (DecayedEpsilonGreedy, {"epsilon":0.5,"decay":0.999}, "Decayed ε","orange"),
    "softmax": (Softmax, {"tau":0.1}, "Softmax (τ=0.1)","green"),
    "annealed_softmax": (AnnealedSoftmax, {"tau0":1.0,"decay":0.999}, "Annealed Softmax","cyan"),
    "ucb1": (UCB1, {}, "UCB1","purple"),
    "thompson": (ThompsonSampling, {}, "Thompson Sampling","brown")
}

# ---- CSV Export with Sweeps ----
results_summary = {}

# Main strategies
for key, (cls, kwargs, label, color) in strategies.items():
    res = run_avg(cls, **kwargs)
    results_summary[label] = (res["avg_rewards"][-1], res["regret"][-1])

# ε sweep
epsilons = [0, 0.1, 0.5, 1]
for eps in epsilons:
    res = run_avg(EpsilonGreedy, epsilon=eps)
    results_summary[f"ε-Greedy (ε={eps})"] = (res["avg_rewards"][-1], res["regret"][-1])

# τ sweep
taus = [0.01, 0.1, 0.5, 1, 2]
for tau in taus:
    res = run_avg(Softmax, tau=tau)
    results_summary[f"Softmax (τ={tau})"] = (res["avg_rewards"][-1], res["regret"][-1])

# Decayed ε sweep
decay_rates = [0.99, 0.999, 0.9995, 0.9999]
for d in decay_rates:
    res = run_avg(DecayedEpsilonGreedy, epsilon=0.5, decay=d)
    results_summary[f"Decayed ε (decay={d})"] = (res["avg_rewards"][-1], res["regret"][-1])

# ---- Print Leaderboard ----
print("\n📊 Strategy Performance Comparison (averaged over runs)")
print("-" * 70)
print(f"{'Strategy':<25}{'Final Avg Reward':<20}{'Total Regret':<20}")
for label, (reward, regret) in results_summary.items():
    print(f"{label:<25}{reward:<20.4f}{regret:<20.4f}")

# Leaderboard by reward
print("\n🏆 Leaderboard: Final Avg Reward (higher is better)")
sorted_reward = sorted(results_summary.items(), key=lambda x: x[1][0], reverse=True)
for rank, (label, (reward, regret)) in enumerate(sorted_reward, 1):
    print(f"{rank}. {label:<25} Reward={reward:.4f}  Regret={regret:.4f}")

# Leaderboard by regret
print("\n🏆 Leaderboard: Total Regret (lower is better)")
sorted_regret = sorted(results_summary.items(), key=lambda x: x[1][1])
for rank, (label, (reward, regret)) in enumerate(sorted_regret, 1):
    print(f"{rank}. {label:<25} Regret={regret:.4f}  Reward={reward:.4f}")

# ---- Save to CSV ----
csv_path = "docs/leaderboard.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Strategy", "Final Avg Reward", "Total Regret"])
    for label, (reward, regret) in results_summary.items():
        writer.writerow([label, f"{reward:.4f}", f"{regret:.4f}"])

print(f"\n📂 Leaderboard with all sweeps saved to {csv_path}")


# ---- Plotting helper ----
def plot_strategy(res, title, filename, color="blue"):
    fig, axes = plt.subplots(1,3,figsize=(20,6))

    # Average reward
    axes[0].plot(res["avg_rewards"], color=color)
    axes[0].axhline(optimal_reward, ls="--", c="red", label="Optimal")
    axes[0].set_title("Average Reward"); axes[0].set_xlabel("Episode")

    # Bandit selection counts
    axes[1].bar(range(BANDITS), res["counts"], color=color)
    axes[1].set_xticks(range(BANDITS))
    axes[1].set_xticklabels([f"Bandit {i+1}\nP={true_probs[i]:.2f}" for i in range(BANDITS)])
    axes[1].set_title("Selection Counts")

    # Regret
    axes[2].plot(res["regret"], color=color)
    axes[2].set_title("Cumulative Regret"); axes[2].set_xlabel("Episode")

    plt.suptitle(title, fontsize=16, fontweight="bold"); plt.tight_layout()
    out = f"docs/{filename}.jpeg"; plt.savefig(out, format="jpeg"); plt.close()
    print(f"✅ Saved {out}")

# ---- Individual plots for main strategies ----
for key,(cls,kwargs,label,color) in strategies.items():
    res = run_avg(cls, **kwargs)
    plot_strategy(res, label, key, color)

# ---- Combined overview plot ----
fig, axs = plt.subplots(1,2,figsize=(16,6))
for key,(cls,kwargs,label,color) in strategies.items():
    res = run_avg(cls,**kwargs)
    axs[0].plot(res["avg_rewards"], label=label, color=color)
    axs[1].plot(res["regret"], label=label, color=color)

axs[0].axhline(optimal_reward, ls="--", c="red", label="Optimal")
axs[0].set_title("Average Reward (All)")
axs[1].set_title("Cumulative Regret (All)")
for ax in axs: ax.set_xlabel("Episode"); ax.legend()
plt.suptitle("All Strategies Averaged", fontsize=16, fontweight="bold")
plt.savefig("docs/all_strategies.jpeg", format="jpeg"); plt.close()
print("✅ Saved docs/all_strategies.jpeg")

# ---- Softmax τ sweep ----
fig, axes = plt.subplots(1,2,figsize=(16,6))
for tau,color in zip(taus,["blue","orange","green","purple","brown"]):
    res = run_avg(Softmax, tau=tau)
    axes[0].plot(res["avg_rewards"], label=f"τ={tau}", color=color)
    axes[1].plot(res["regret"], label=f"τ={tau}", color=color)
axes[0].axhline(optimal_reward, ls="--", c="red", label="Optimal")
axes[0].set_title("Softmax: Avg Reward"); axes[1].set_title("Softmax: Regret")
for ax in axes: ax.set_xlabel("Episode"); ax.legend()
plt.suptitle("Softmax Exploration with Different τ values", fontsize=16, fontweight="bold")
plt.savefig("docs/softmax_tau_sweep.jpeg", format="jpeg"); plt.close()
print("✅ Saved docs/softmax_tau_sweep.jpeg")

# ---- Decayed ε sweep ----
fig, axes = plt.subplots(1,2,figsize=(16,6))
for d,color in zip(decay_rates,["blue","orange","green","purple"]):
    res = run_avg(DecayedEpsilonGreedy, epsilon=0.5, decay=d)
    axes[0].plot(res["avg_rewards"], label=f"decay={d}", color=color)
    axes[1].plot(res["regret"], label=f"decay={d}", color=color)
axes[0].axhline(optimal_reward, ls="--", c="red", label="Optimal")
axes[0].set_title("Decayed ε-Greedy: Avg Reward"); axes[1].set_title("Decayed ε-Greedy: Regret")
for ax in axes: ax.set_xlabel("Episode"); ax.legend()
plt.suptitle("Decayed ε-Greedy with Different Decay Rates", fontsize=16, fontweight="bold")
plt.savefig("docs/decayed_epsilon_sweep.jpeg", format="jpeg"); plt.close()
print("✅ Saved docs/decayed_epsilon_sweep.jpeg")

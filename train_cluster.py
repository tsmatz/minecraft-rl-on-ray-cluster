import os
import argparse
import ray
import ray.tune as tune

# Function for stopping a learner when successful training
def stop_check(trial_id, result):
    return result["episode_reward_mean"] >= 85

# Main
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers",
        type=int,
        required=False,
        default=1,
        help="number of ray workers")
    parser.add_argument("--num_gpus",
        type=int,
        required=False,
        default=0,
        help="number of gpus")
    parser.add_argument("--num_cpus_per_worker",
        type=int,
        required=False,
        default=1,
        help="number of cores per worker")
    args = parser.parse_args()

    ray.init(address='auto')

    ray.tune.run(
        "IMPALA",
        config={
            "log_level": "WARN",
            "env": "custom_malmo_env:MalmoMazeEnv-v0",
            "num_workers": args.num_workers,
            "num_gpus": args.num_gpus,
            "num_cpus_per_worker": args.num_cpus_per_worker,
            "explore": True,
            "learner_queue_timeout": 600,
            "exploration_config": {
                "type": "EpsilonGreedy",
                "initial_epsilon": 1.0,
                "final_epsilon": 0.02,
                "epsilon_timesteps": 500000
            }
        },
        stop=stop_check,
        checkpoint_at_end=True,
        checkpoint_freq=2,
        local_dir='./logs'
    )

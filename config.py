import argparse
from ast import parse
from datetime import datetime


def get_config():
    # Initialize the argument parser for configuration settings
    parser = argparse.ArgumentParser()

    # Path to the pkl dataset containing demonstrations
    parser.add_argument('--dataset_path', type=str, default='./dataset.pkl')

    # Training arguments
    # Set standard random seed for reproducibility
    parser.add_argument('--seed', type=int, default=3)
    # Allows switching between standard MLP and diffusion policies
    parser.add_argument('--policy', type=str, default='diffusion', choices=['mlp', 'diffusion'])
    # Exponential Moving Average (EMA) flag
    parser.add_argument('--use_ema_model', action='store_true')
    # Decay power for EMA moving rate
    parser.add_argument('--ema_power', type=float, default=0.75)
    # Initial burn-in steps to skip before accumulating EMA weights
    parser.add_argument('--ema_update_after_step', type=int, default=0)
    # Inverse gamma schedules for dynamic EMA calculation progression
    parser.add_argument('--ema_inv_gamma', type=float, default=1)
    # EMA bounds to heavily restrict sudden drops or peaks
    parser.add_argument('--ema_min_value', type=float, default=0.)
    parser.add_argument('--ema_max_value', type=float, default=0.999)
    # If using sequence based temporal predictions
    parser.add_argument('--sequential', action='store_true')
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[1024, 1024, 1024])
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'mps', 'cuda'])
    # Sequence length window indicating how many past states/observations the agent can see 
    parser.add_argument('--obs_horizon', type=int, default=4)
    # Sequence length indicating chunk size limits on projected future actions
    parser.add_argument('--pred_horizon', type=int, default=8)
    # embedding feature-vector dimension for transformer
    parser.add_argument('--emb_dim', type=int, default=128)
    # Transformer attention head total
    parser.add_argument('--n_heads', type=int, default=4)
    # Stacking depth count for model networks Transformer blocks
    parser.add_argument('--n_layers', type=int, default=3)
    # diffusion timestep noise scheduling loops
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--saveloc', type=str, default='./results/')
    default_name = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    parser.add_argument('--savename', type=str, default=default_name)

    # Evaluating arguments
    # file path for checkpoint
    parser.add_argument('--loadloc', type=str, default=None)
    parser.add_argument('--num_evals', type=int, default=10)
    # time horizon for each evaluation run
    parser.add_argument('--time_horizon', type=int, default=300)
    # number of denoising steps at inference time
    parser.add_argument('--inference_steps', type=int, default=25)
    # Action chunks executed consecutively before querying states again
    parser.add_argument('--n_rollout_actions', type=int, default=1)
    
    args = parser.parse_args()

    if args.policy == 'diffusion':
        # Diffusion relies directly on chained sequences over singular steps 
        args.sequential = True
        # Often EMA dramatically improves sample quality via model averaging
        args.use_ema_model = True
    return args

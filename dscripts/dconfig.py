import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import argparse
from dscripts.dutils import wrap_single, wrap_single_nested, wrap_double


def get_argument_parser():
    """
    Creates the argument parser.
    """
    
    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser(
        description="dconfig", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--runs_on",
        required=True,
        type=str,
        help="Comma-separated IP addresses of machines",
    )
    parser.add_argument(
        "--config_script",
        required=True,
        type=str,
        help="Path to bash script with configuration",
    )
    parser.add_argument(
        "--ssh-cmd",
        required=False,
        type=str,
        default="ssh -f {addr}",
        help="SSH command. Useful to utilize a pre-shared key with 'ssh -i path/to/mykey.pem -f ubuntu@{addr}'. "
        "The option `-f` should be used, since we want to a non-interactive session",
    )
    parser.add_argument(
        "--distribute_public_rsa_key",
        dest="distribute_public_rsa_key",
        action="store_true",
        required=False,
        help="if you pass the `--distribute_public_rsa_key` flag, the manager node's public key will be added to the "
        "authorized keys of all wokrers (this is necessary in default-configured EC2 instances to use "
        "`scripts/dmain.py`)",
    )
    parser.set_defaults(distribute_public_rsa_key=False)
    parser.add_argument(
        "--experiment",
        required=True,
        type=str,
        help="Experiment Name"
    )
    
    return parser


def get_args():
    """
    Creates the argument parser and parses any input arguments.
    """
    parser = get_argument_parser()
    args = parser.parse_args()
    
    return args


if __name__ == "__main__":
    args = get_args()
    
    all_addresses = args.runs_on.split(",")
    print(f"Running on addresses {all_addresses}")
    
    remote_config_scripts = f"{os.path.split(args.config_script)[-1]}.distributed"
    
    for it, addr in enumerate(all_addresses):
        
        if args.distribute_public_rsa_key:
            key_command = (
                f"{args.ssh_cmd.format(addr=addr)} "
                f"{wrap_double('echo $(cat ~/.ssh/id_rsa.pub) >> ~/.ssh/authorized_keys')}"
            )
            # print(f"Key command: {key_command}")
            os.system(f"{key_command}")
            
        scp_cmd = (
            f'{args.ssh_cmd.replace("ssh ", "scp ").replace("-f", args.config_script).format(addr=addr)}'
            f":~/research/{remote_config_scripts}"
        )
        # print(f"SCP command: {scp_cmd}")
        os.system(f"{scp_cmd}")
        
        tmux_name = f"{args.experiment}-{addr}"
        bash_cmd = wrap_double(
            f"source ~/research/{remote_config_scripts} &>> dconfig.log"
        )
        tmux_cmd = wrap_single(
            f"tmux new-session -s {tmux_name} -d && tmux send-keys -t {tmux_name} {bash_cmd} C-m"
        )
        
        ssh_command = f"{args.ssh_cmd.format(addr=addr)} {tmux_cmd}"
        # print(f"SSH command: {ssh_command}")
        os.system(f"{ssh_command}")
        # print(f"{addr} => {tmux_name}")
        
    print("DONE")
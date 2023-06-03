"""Entry point to multi-node (distributed) training for a user given experiment
name."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import argparse
import random
import string
import subprocess
import time
import re
import socket
from pathlib import Path
from typing import Optional

from dscripts.dutils import wrap_single, wrap_single_nested, wrap_double
from allenact.main import get_argument_parser as get_main_argument_parser
from allenact.utils.system import init_logging, get_logger, find_free_port
from rearrange_constants import ABS_PATH_OF_REARRANGE_TOP_LEVEL_DIR
from dscripts.sshutils import read_openssh_config


def get_argument_parser():
    """Creates the argument parser."""
    
    parser = get_main_argument_parser()
    parser.description = f"distributed {parser.description}"
    
    parser.add_argument(
        "--runs_on",
        required=True,
        type=str,
        help="Comma-separated IP addresses of machines",
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
        "--conda_env",
        required=True,
        type=str,
        help="Name of the conda environment. It must be the same across all machines",
    )
    parser.add_argument(
        "--code_base_dir",
        required=False,
        type=str,
        default="~/research/rearrange2022",
        help="Path to code base directory. It must be the same across all machines",
    )
    parser.add_argument(
        "--experiment_name",
        required=True,
        type=str,
        help="Experiment Name"
    )
    
    # Required distributed_ip_and_port
    idx = [a.dest for a in parser._actions].index("distributed_ip_and_port")
    parser._actions[idx].required = True

    return parser


def get_args():
    """
    Creates the argument parser and parses any input arguments.
    """
    parser = get_argument_parser()
    args = parser.parse_args()
    
    return args


def get_raw_args():
    raw_args = sys.argv[1:]
    filtered_args = []
    remove: Optional[str] = None
    enclose_in_quotes: Optional[str] = None
    for arg in raw_args:
        if remove is not None:
            remove = None
        elif enclose_in_quotes is not None:
            # Within backslash expansion: close former single, open double, create single, close double, reopen single
            inner_quote = r"\'\"\'\"\'"
            # Convert double quotes into backslash double for later expansion
            filtered_args.append(
                inner_quote + arg.replace('"', r"\"").replace("'", r"\"") + inner_quote
            )
            enclose_in_quotes = None
        elif arg in [
            "--runs_on",
            "--ssh_cmd",
            "--extra_tag",
            "--machine_id",
            "--conda_env",
            "--code_base_dir",
            "--experiment_name",
        ]:
            remove = arg
        elif arg == "--config_kwargs":
            enclose_in_quotes = arg
            filtered_args.append(arg)
        else:
            filtered_args.append(arg)
    return filtered_args


def id_generator(size: int = 4, chars: str = (string.ascii_uppercase + string.digits)):
    return "".join(random.choice(chars) for _ in range(size))


if __name__ == "__main__":
    cwd = os.path.abspath(os.getcwd())
    assert cwd == ABS_PATH_OF_REARRANGE_TOP_LEVEL_DIR, (
        f"`dmain.py` called from {cwd}."
        f"\nIt should be called from {ABS_PATH_OF_REARRANGE_TOP_LEVEL_DIR}."
    )
    
    args = get_args()
    
    init_logging(args.log_level)
    
    raw_args = get_raw_args()
    
    if args.seed is None:
        seed = random.randint(0, 2 ** 31 - 1)
        raw_args.extend(["-s", f"{seed}"])
        get_logger().info(f"Using random seed {seed} in all workers (none was given)")
        
    all_addresses = args.runs_on.split(",")
    all_ip_addresses = [read_openssh_config(addr)[0] for addr in all_addresses]
    
    get_logger().info(f"Running on IP addresses {[f'{addr} ({ip})' for addr, ip in zip(all_addresses, all_ip_addresses)]}")
    
    assert (
        args.distributed_ip_and_port.split(":")[0] in all_addresses
        or args.distributed_ip_and_port.split(":")[0] in all_ip_addresses
    ), (
        f"Missing listener IP address {args.distributed_ip_and_port.split(':')[0]}"
        f" in list of worker addresses ({[f'{addr} ({ip})' for addr, ip in zip(all_addresses, all_ip_addresses)]})"
    )
    
    ip_regex = re.compile(r'^\d{1,3}\.\d{1,3}\.\d{1,3}.\d{1,3}')
    if re.match(ip_regex, args.distributed_ip_and_port.split(":")[0]):
        distributed_ip = args.distributed_ip_and_port.split(":")[0]
    else:
        distributed_ip, _, _, _ = read_openssh_config(args.distributed_ip_and_port.split(":")[0])
        assert re.match(ip_regex, distributed_ip)
    
    assert len(args.distributed_ip_and_port.split(":")) == 2
    port = args.distributed_ip_and_port.split(":")[1]
    idx = raw_args.index("--distributed_ip_and_port")
    raw_args[idx + 1] = f"{distributed_ip}:{port}"
    
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
    
    global_job_id = id_generator()
    killfilename = os.path.join(
        os.path.expanduser("~"), ".allenact", f"{time_str}_{global_job_id}.killfile"
    )
    os.makedirs(os.path.dirname(killfilename), exist_ok=True)
    
    code_src = "."
    
    with open(killfilename, "w") as killfile:
        for it, (addr, addr_ip) in enumerate(zip(all_addresses, all_ip_addresses)):
            job_id = id_generator()
            
            command = " ".join(
                ["allenact"]
                + raw_args
                + [
                    "--extra_tag",
                    f"{args.extra_tag}{'__' if len(args.extra_tag) > 0 else ''}machine{it}",
                ]
                + ["--machine_id", f"{it}"]
            )
            
            logfile = (
                f"{args.output_dir}/log_{time_str}_{global_job_id}_{job_id}_machine{it}"
            )
            
            env_and_command = wrap_single_nested(
                f"for NCCL_SOCKET_IFNAME in $(route | grep default) ; do : ; done && export NCCL_SOCKET_IFNAME"
                f" && cd {args.code_base_dir}"
                f" && export PYTHONPATH=$PYTHONPATH:$PWD"
                f" && mkdir -p {args.output_dir}"
                f" && conda activate {args.conda_env} &>> {logfile}"
                f" && echo pwd=$(pwd) &>> {logfile}"
                f" && echo output_dir={args.output_dir} &>> {logfile}"
                f" && echo python_version=$(python --version) &>> {logfile}"
                f" && echo python_path=$(which python) &>> {logfile}"
                f" && set | grep NCCL_SOCKET_IFNAME &>> {logfile}"
                f" && echo allenact_path=$(which allenact) &>> {logfile}"
                f" && echo &>> {logfile}"
                f" && {command} &>> {logfile}"
            )
            
            tmux_name = f"{args.experiment_name}_{time_str}_{global_job_id}_{job_id}_machine{it}"
            tmux_new_command = wrap_single(
                f"tmux new-session -s {tmux_name} -d && tmux send-keys -t {tmux_name} {env_and_command} C-m"
            )
            ssh_command = f"{args.ssh_cmd.format(addr=addr)} {tmux_new_command}"
            get_logger().debug(f"SSH command {ssh_command}")
            subprocess.run(ssh_command, shell=True, executable="/bin/bash")
            get_logger().info(f"{addr} {tmux_name}")

            killfile.write(f"{addr} {tmux_name}\n")

    get_logger().info("")
    get_logger().info(f"Running tmux ids saved to {killfilename}")
    get_logger().info("")
    
    get_logger().info("DONE")

import os
from paramiko import SSHConfig

def read_openssh_config(host, config_file=None):
    _ssh_config_file = config_file if config_file else \
        os.path.sep.join([os.path.expanduser("~"), '.ssh', 'config'])
    if not os.path.isfile(_ssh_config_file):
        print(f'Wrong config file assigned...')
        return

    ssh_config = SSHConfig()
    ssh_config.parse(open(_ssh_config_file))
    host_config = ssh_config.lookup(host)
    host = (
        host_config['hostname'] 
        if 'hostname' in host_config else host
    )
    user = (
        host_config['user']
        if 'user' in host_config else None
    )
    port = int(
        host_config['port']
        if 'port' in host_config else 22
    )
    pkey = None
    
    if 'identityfile' in host_config:
        if isinstance(host_config['identityfile'], str):
            pkey = os.path.expanduser(host_config['identityfile']).strip()
        elif isinstance(host_config['identityfile'], list) and len(host_config['identityfile']) > 0:
            pkey = os.path.expanduser(host_config['identityfile'][0]).strip()
    
    return host, user, port, pkey
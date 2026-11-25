# Board setup

- `setup.sh`: one-time install on the KV260 (links the repo to /opt, enables the two services)
- `pcb-inspect.service`: starts the station 20 s after boot
- `pcb-ip.sh`, `pcb-ip.service`: fixed address 192.168.50.1 on eth0 and a small DHCP server when no router is present, so a laptop on a direct cable gets an address without any internet

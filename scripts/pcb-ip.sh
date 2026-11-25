#!/bin/bash
# Direct laptop link without a router or internet.
# Keeps 192.168.50.1 on eth0. If eth0 gets no other address for 30 s (no router on the cable),
# it starts a small DHCP server so the laptop gets 192.168.50.x automatically.
# No gateway or DNS is handed out, so the laptop keeps using its Wi-Fi for internet.
IF=eth0; IP=192.168.50.1/24; alone=0
while true; do
    ip addr show $IF | grep -q "192.168.50.1/" || ip addr add $IP dev $IF 2>/dev/null
    other=$(ip -4 addr show $IF | grep inet | grep -v "192.168.50.1/")
    if [ -z "$other" ]; then alone=$((alone + 5)); else alone=0; fi
    if [ $alone -ge 30 ] && ! [ -f /run/pcb-dhcp.pid ]; then
        /usr/sbin/dnsmasq --conf-file=/dev/null --port=0 --interface=$IF --bind-interfaces \
            --dhcp-range=192.168.50.10,192.168.50.50,255.255.255.0,12h --dhcp-option=3 --dhcp-option=6 \
            --dhcp-leasefile=/run/pcb-dhcp.leases --pid-file=/run/pcb-dhcp.pid
    fi
    if [ -n "$other" ] && [ -f /run/pcb-dhcp.pid ]; then
        kill "$(cat /run/pcb-dhcp.pid)" 2>/dev/null; rm -f /run/pcb-dhcp.pid
    fi
    sleep 5
done

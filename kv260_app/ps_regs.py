#!/usr/bin/env python3
# Loading dpu.bit through PYNQ rewrites some PS-side AXI width registers for the DPU design.
# The camera design expects the boot-time values, so we save them on the first camera start
# after boot and put them back on every later start.
import os, mmap, struct

REGS = [0xFD615000, 0xFF419000] + [b + o for b in (0xFD360000, 0xFD370000, 0xFD380000, 0xFD390000,
                                                   0xFD3A0000, 0xFD3B0000, 0xFF9B0000) for o in (0x0, 0x14)]
SAVE = "/run/pcb_psregs.saved"          # /run is cleared on every boot

f = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)


def rw(addr, val=None):
    m = mmap.mmap(f, 0x1000, offset=addr & ~0xFFF)
    o = addr & 0xFFF
    if val is not None:
        m[o:o + 4] = struct.pack("<I", val)
    return struct.unpack("<I", m[o:o + 4])[0]


now = [rw(a) for a in REGS]
if not os.path.exists(SAVE):
    open(SAVE, "w").write(" ".join(map(hex, now)))
    print("PS regs saved (boot state)")
else:
    boot = [int(x, 16) for x in open(SAVE).read().split()]
    for a, b, n in zip(REGS, boot, now):
        if b != n:
            rw(a, b)
            print(f"PS reg {a:#x}: {n:#x} -> restored {b:#x}")

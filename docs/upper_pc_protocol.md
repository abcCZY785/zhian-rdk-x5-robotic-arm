# Upper PC Communication Protocol (v7)

> This document describes the serial communication protocol between the host PC and the lower controller (STM32/Feetech).

## Overview

- Baud rate: 115200
- Text commands: LF-terminated ASCII
- RL binary delta command: `0x55 0xBB` + 6x float32 + checksum (27 bytes)
- RL binary report frame: 58 bytes (current firmware), 65-byte compatibility parsing

## Text Commands

> TODO: Migrate from original `upper_pc_protocol.md`.

## Binary Protocol

### RL Delta Command (Host → Controller)

```
Header:  0x55 0xBB
Payload: 6x float32 (joint deltas in degrees)
Checksum: sum of payload bytes & 0xFF
Total:   27 bytes
```

### RL Report Frame (Controller → Host)

```
Header:  0x55 0xAA
Payload: 6x float32 joint positions + status bytes
Total:   58 bytes (current) / 65 bytes (compat)
```

## Gripper Control

> TODO: Document closed-loop gripper control (added in v7).

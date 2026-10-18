#!/usr/bin/env python3
"""Generate model opcode identifiers from the project's canonical ISA schema."""
import json
from pathlib import Path
import sys

schema = json.loads(Path(sys.argv[1]).read_text())
lines = ['#pragma once', '#include <cstdint>', 'namespace isa {', 'enum Opcode : std::uint16_t {']
for ins in schema['instructions']:
    name = ins['mnemonic'].replace('.', '_')
    lines.append(f'  {name} = {(ins["major_code"] << 8) | ins["func"]},')
lines.extend(['};', '}'])
Path(sys.argv[2]).write_text('\n'.join(lines)+'\n')

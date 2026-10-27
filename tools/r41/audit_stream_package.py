#!/usr/bin/env python3
"""Audit actual packaged stream-CU metadata without opening a board."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import xml.etree.ElementTree as ET
from generate_stream_binding import FIELDS


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(component, kernel_xml):
    root = ET.fromstring(component)
    ns = {'s': root.tag.split('}')[0][1:]}
    buses = {b.findtext('s:name', namespaces=ns): b for b in root.findall('s:busInterfaces/s:busInterface', ns)}
    expected = {f'm_axi_pc{p}' for p in range(32)} | {'s_axi_tnca'}
    require(set(buses) == expected | {'ap_clk','ap_rst_n'}, 'packaged bus interface set differs')
    def parameters(bus):
        return {p.findtext('s:name', namespaces=ns): p.findtext('s:value', namespaces=ns)
                for p in buses[bus].findall('s:parameters/s:parameter', ns)}
    clock = parameters('ap_clk')
    require(set(clock.get('ASSOCIATED_BUSIF','').split(':')) == expected, 'clock association misses interfaces')
    require(clock.get('FREQ_HZ') == '100000000', 'stream clock metadata is not 100 MHz')
    require(clock.get('ASSOCIATED_RESET') == 'ap_rst_n', 'stream reset association differs')
    require(parameters('ap_rst_n').get('POLARITY') == 'ACTIVE_LOW', 'stream reset is not active-low')
    ports = {p.findtext('s:name', namespaces=ns): p for p in root.findall('s:model/s:ports/s:port', ns)}
    for pc in range(32):
        mappings = {p.findtext('s:logicalPort/s:name', namespaces=ns):p.findtext('s:physicalPort/s:name', namespaces=ns)
                    for p in buses[f'm_axi_pc{pc}'].findall('s:portMaps/s:portMap', ns)}
        for field, direction, width in FIELDS:
            name = f'm_axi_pc{pc}_{field}'
            require(mappings.get(field.upper()) == name, f'logical/physical mapping differs: {name}')
            require(name in ports, f'packaged physical port missing: {name}')
            port = ports[name]
            require(port.findtext('s:wire/s:direction', namespaces=ns) == ('out' if direction=='output' else 'in'),
                    f'packaged port direction differs: {name}')
            vector = port.find('s:wire/s:vector', ns)
            actual_width = 1 if vector is None else abs(int(vector.findtext('s:left', namespaces=ns))-
                                                        int(vector.findtext('s:right', namespaces=ns)))+1
            require(actual_width == width, f'packaged port width differs: {name}')
    kernel = ET.fromstring(kernel_xml).find('kernel')
    require(kernel is not None and kernel.get('name') == 'tnca_stream', 'kernel name differs')
    args = kernel.findall('args/arg')
    require(len(args) == 32, 'kernel must expose 32 memory arguments')
    masters = {p.get('name'):p for p in kernel.findall('ports/port') if p.get('mode')=='master'}
    require(set(masters) == expected-{'s_axi_tnca'}, 'kernel master names differ')
    for pc, arg in enumerate(args):
        require(arg.get('id') == str(pc) and arg.get('name') == f'pc{pc}' and arg.get('port') == f'm_axi_pc{pc}',
                'kernel memory argument mapping differs')
        require(masters[f'm_axi_pc{pc}'].get('dataWidth') == '512', 'kernel memory data width differs')
    return dict(status='pass', masters=32, data_width=512, clock_hz=100000000, reset='active-low',
                component_sha256=hashlib.sha256(component).hexdigest(), kernel_sha256=hashlib.sha256(kernel_xml).hexdigest(),
                qualification='packaged-metadata-only-not-linked-image')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('metadata', type=Path)
    args = parser.parse_args()
    with tarfile.open(args.metadata) as archive:
        component = next(n for n in archive.getnames() if n.endswith('/component.xml'))
        result = audit(archive.extractfile(component).read(), archive.extractfile('kernel.xml').read())
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

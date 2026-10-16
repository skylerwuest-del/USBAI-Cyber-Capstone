"""Bounded structural GGUF validation; backend compatibility is preparation-tested."""
import struct
import math


def validate_structure(path):
    with path.open('rb') as handle:
        size = path.stat().st_size

        def number(fmt):
            raw = handle.read(struct.calcsize(fmt))
            if len(raw) != struct.calcsize(fmt):
                raise ValueError('Truncated GGUF table')
            return struct.unpack(fmt, raw)[0]

        def skip(count):
            if count > size - handle.tell():
                raise ValueError('Truncated GGUF value')
            handle.seek(count, 1)

        def string():
            length = number('<Q')
            if length > 1024 * 1024:
                raise ValueError('Excessive GGUF key length')
            raw = handle.read(length)
            if len(raw) != length:
                raise ValueError('Truncated GGUF key')
            return raw.decode('utf-8')

        def value(kind):
            sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
            if kind in sizes:
                skip(sizes[kind])
            elif kind == 8:
                skip(number('<Q'))
            elif kind == 9:
                subtype, count = number('<I'), number('<Q')
                if subtype in sizes:
                    skip(count * sizes[subtype])
                elif subtype == 8 and count <= size // 8:
                    for _ in range(count):
                        skip(number('<Q'))
                else:
                    raise ValueError('Invalid GGUF array')
            else:
                raise ValueError('Invalid GGUF metadata type')

        if handle.read(4) != b'GGUF' or number('<I') not in (2, 3):
            raise ValueError('Invalid GGUF header')
        tensors, metadata = number('<Q'), number('<Q')
        if not 0 < tensors <= min(size // 32, 1_000_000) or metadata > min(size // 12, 1_000_000):
            raise ValueError('Invalid GGUF table counts')
        alignment = 32
        scalars = {}
        for _ in range(metadata):
            key, kind = string(), number('<I')
            if key == 'general.alignment':
                if kind != 4:
                    raise ValueError('Invalid GGUF alignment type')
                alignment = number('<I')
                if not alignment or alignment > 1024 * 1024 or alignment & (alignment - 1):
                    raise ValueError('Invalid GGUF alignment')
            elif kind in (4, 10):
                scalars[key] = number('<I' if kind == 4 else '<Q')
            elif kind == 8 and key.startswith('general.'):
                scalars[key] = string()
            else:
                value(kind)
        offsets = []
        names = set()
        parameters = 0
        for _ in range(tensors):
            name, dimensions = string(), number('<I')
            if not name or name in names or not 1 <= dimensions <= 4:
                raise ValueError('Invalid GGUF tensor description')
            names.add(name)
            shape = []
            for _ in range(dimensions):
                dimension = number('<Q')
                if dimension == 0:
                    raise ValueError('Empty GGUF tensor')
                shape.append(dimension)
            parameters += math.prod(shape)
            number('<I')  # Tensor type is checked by the prepared llama.cpp backend.
            offset = number('<Q')
            if offset % alignment:
                raise ValueError('Unaligned GGUF tensor')
            offsets.append(offset)
        data_start = (handle.tell() + alignment - 1) // alignment * alignment
        if any(data_start + offset >= size for offset in offsets):
            raise ValueError('Missing GGUF tensor data')
        return scalars, parameters

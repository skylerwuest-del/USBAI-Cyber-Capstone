"""Read a GGUF's native text context without loading its tensor weights."""
import struct


SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def context_length(path):
    with path.open('rb') as handle:
        size = path.stat().st_size
        def number(fmt):
            data = handle.read(struct.calcsize(fmt))
            if len(data) != struct.calcsize(fmt):
                raise ValueError('Truncated GGUF metadata')
            return struct.unpack(fmt, data)[0]
        def skip(count):
            if count < 0 or handle.tell() + count > size:
                raise ValueError('Truncated GGUF metadata')
            handle.seek(count, 1)
        def skip_value(kind):
            if kind in SIZES:
                skip(SIZES[kind])
            elif kind == 8:
                skip(number('<Q'))
            elif kind == 9:
                subtype, count = number('<I'), number('<Q')
                if subtype in SIZES:
                    skip(count * SIZES[subtype])
                elif subtype == 8:
                    for _ in range(count):
                        skip(number('<Q'))
                else:
                    raise ValueError('Unsupported GGUF array metadata')
            else:
                raise ValueError('Unsupported GGUF metadata type')
        if handle.read(4) != b'GGUF' or number('<I') not in (2, 3):
            raise ValueError('Choose a GGUF v2/v3 model')
        number('<Q')  # Tensor count; no weights are read.
        count = number('<Q')
        for _ in range(count):
            length = number('<Q')
            if length > 1024 * 1024 or handle.tell() + length > size:
                raise ValueError('Invalid GGUF metadata key')
            key = handle.read(length).decode('utf-8')
            kind = number('<I')
            if key.endswith('.context_length') and not key.startswith(('clip.', 'vision.')):
                if kind not in (4, 10):
                    raise ValueError('Invalid native context in GGUF')
                value = number('<I' if kind == 4 else '<Q')
                if value <= 0:
                    raise ValueError('Invalid native context in GGUF')
                return value
            skip_value(kind)
    return None

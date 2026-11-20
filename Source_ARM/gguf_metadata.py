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


def read_metadata(path):
    """Bounded scalar metadata/tensor-shape reader; never load token arrays or weights."""
    formats = {0: '<B', 1: '<b', 2: '<H', 3: '<h', 4: '<I', 5: '<i',
               6: '<f', 7: '<?', 10: '<Q', 11: '<q', 12: '<d'}
    with path.open('rb') as handle:
        size = path.stat().st_size
        def read(n):
            if n < 0 or n > size - handle.tell():
                raise ValueError('Truncated GGUF metadata')
            value = handle.read(n)
            if len(value) != n:
                raise ValueError('Truncated GGUF metadata')
            return value
        def number(fmt):
            return struct.unpack(fmt, read(struct.calcsize(fmt)))[0]
        def skip(n):
            if n < 0 or n > size - handle.tell():
                raise ValueError('Truncated GGUF metadata')
            handle.seek(n, 1)
        def string(keep=True):
            n = number('<Q')
            if keep:
                if n > 1024 * 1024:
                    raise ValueError('Oversized GGUF string')
                return read(n).decode('utf-8')
            skip(n)
        if read(4) != b'GGUF' or number('<I') not in (2, 3):
            raise ValueError('Expected GGUF v2/v3')
        tensors, count = number('<Q'), number('<Q')
        if count > 100000 or tensors > 1000000:
            raise ValueError('Unreasonable GGUF metadata/tensor count')
        result = {}
        for _ in range(count):
            key, kind = string(), number('<I')
            # Do not retain tokenizer content or prompt templates in diagnostics.
            keep = not key.startswith('tokenizer.')
            if kind in formats:
                value = number(formats[kind])
                if keep:
                    result[key] = value
            elif kind == 8:
                if key == 'tokenizer.chat_template':
                    result['has_chat_template'] = True
                value = string(keep)
                if keep:
                    result[key] = value
            elif kind == 9:
                subtype, length = number('<I'), number('<Q')
                if length > 10000000:
                    raise ValueError('Oversized GGUF array')
                if subtype in formats:
                    skip(length * struct.calcsize(formats[subtype]))
                elif subtype == 8:
                    for _ in range(length):
                        string(False)
                else:
                    raise ValueError('Unsupported GGUF array type')
            else:
                raise ValueError('Unsupported GGUF metadata type')
        parameters = 0
        for _ in range(tensors):
            string(False)
            dimensions = number('<I')
            if not 1 <= dimensions <= 4:
                raise ValueError('Invalid GGUF tensor dimensions')
            elements = 1
            for _ in range(dimensions):
                extent = number('<Q')
                if not 1 <= extent <= 2**32:
                    raise ValueError('Invalid GGUF tensor shape')
                elements *= extent
            parameters += elements
            number('<I')  # quantization type
            offset = number('<Q')
            if offset >= size:
                raise ValueError('Invalid GGUF tensor offset')
        result['parameter_count_estimate'] = parameters
        result['tensor_count'] = tensors
        return result

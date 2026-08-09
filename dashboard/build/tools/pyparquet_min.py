"""
pyparquet_min.py — a minimal, dependency-free Parquet reader.

Built because this sandbox has no network access and neither pyarrow nor
fastparquet is installed, but we need to read the REAL pipeline output
files (not regenerate synthetic data) to build the dashboard's data JSON.

Supports exactly what pandas.to_parquet(engine='pyarrow') typically
produces for flat (non-nested) DataFrames: PLAIN and RLE_DICTIONARY value
encodings, RLE/bit-packing-hybrid definition levels for nullable columns,
and SNAPPY / GZIP / UNCOMPRESSED page compression. Not a general-purpose
reader — no nested lists/maps/structs in the data itself (only in the
Thrift metadata, which this does handle generically).
"""

from __future__ import annotations

import ctypes
import gzip
import struct
from dataclasses import dataclass, field

_libzstd = None
def _get_libzstd():
    global _libzstd
    if _libzstd is None:
        _libzstd = ctypes.CDLL("libzstd.so.1")
        _libzstd.ZSTD_getFrameContentSize.restype = ctypes.c_ulonglong
        _libzstd.ZSTD_getFrameContentSize.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        _libzstd.ZSTD_decompress.restype = ctypes.c_size_t
        _libzstd.ZSTD_decompress.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
        ]
        _libzstd.ZSTD_isError.restype = ctypes.c_uint
        _libzstd.ZSTD_isError.argtypes = [ctypes.c_size_t]
    return _libzstd


def _zstd_decompress(data: bytes) -> bytes:
    lib = _get_libzstd()
    content_size = lib.ZSTD_getFrameContentSize(data, len(data))
    if content_size in (0, 2**64 - 1, 2**64 - 2):  # unknown/error sentinels
        content_size = len(data) * 20 + 1024  # generous fallback buffer
    out_buf = ctypes.create_string_buffer(content_size)
    written = lib.ZSTD_decompress(out_buf, content_size, data, len(data))
    if lib.ZSTD_isError(written):
        raise ValueError("ZSTD decompression failed")
    return out_buf.raw[:written]


# --------------------------------------------------------------------------
# Thrift compact-protocol generic decoder
# --------------------------------------------------------------------------

class _Cursor:
    __slots__ = ("buf", "pos")

    def __init__(self, buf: bytes, pos: int = 0):
        self.buf = buf
        self.pos = pos

    def byte(self) -> int:
        b = self.buf[self.pos]
        self.pos += 1
        return b

    def bytes(self, n: int) -> bytes:
        b = self.buf[self.pos:self.pos + n]
        self.pos += n
        return b

    def uvarint(self) -> int:
        result = 0
        shift = 0
        while True:
            b = self.byte()
            result |= (b & 0x7F) << shift
            if not (b & 0x80):
                return result
            shift += 7

    def zigzag(self) -> int:
        n = self.uvarint()
        return (n >> 1) ^ -(n & 1)


def _read_thrift_value(c: _Cursor, ctype: int, in_collection: bool):
    if ctype in (1, 2):  # BOOLEAN_TRUE / BOOLEAN_FALSE
        if in_collection:
            return c.byte() != 0
        return ctype == 1
    if ctype == 3:  # BYTE
        v = c.byte()
        return v - 256 if v >= 128 else v
    if ctype in (4, 5, 6):  # I16 / I32 / I64
        return c.zigzag()
    if ctype == 7:  # DOUBLE
        return struct.unpack("<d", c.bytes(8))[0]
    if ctype == 8:  # BINARY / STRING
        n = c.uvarint()
        return c.bytes(n)
    if ctype in (9, 10):  # LIST / SET
        header = c.byte()
        size = (header >> 4) & 0x0F
        elem_type = header & 0x0F
        if size == 15:
            size = c.uvarint()
        return [_read_thrift_value(c, elem_type, True) for _ in range(size)]
    if ctype == 11:  # MAP
        size = c.uvarint()
        if size == 0:
            return {}
        types_byte = c.byte()
        key_type = (types_byte >> 4) & 0x0F
        val_type = types_byte & 0x0F
        return {
            _read_thrift_value(c, key_type, True): _read_thrift_value(c, val_type, True)
            for _ in range(size)
        }
    if ctype == 12:  # STRUCT
        return _read_thrift_struct(c)
    raise ValueError(f"Unsupported thrift compact type: {ctype}")


def _read_thrift_struct(c: _Cursor) -> dict:
    result = {}
    last_field_id = 0
    while True:
        header = c.byte()
        if header == 0:
            return result
        delta = (header >> 4) & 0x0F
        ctype = header & 0x0F
        if delta == 0:
            field_id = c.zigzag()
        else:
            field_id = last_field_id + delta
        last_field_id = field_id
        result[field_id] = _read_thrift_value(c, ctype, False)


# --------------------------------------------------------------------------
# Snappy raw-block decompression (pure Python)
# --------------------------------------------------------------------------

def _snappy_decompress(data: bytes) -> bytes:
    c = _Cursor(data)
    length = c.uvarint()
    out = bytearray()
    n = len(data)
    while c.pos < n:
        tag = c.byte()
        kind = tag & 0x03
        if kind == 0:  # literal
            top = tag >> 2
            if top < 60:
                lit_len = top + 1
            else:
                extra = top - 59
                lit_len = int.from_bytes(c.bytes(extra), "little") + 1
            out += c.bytes(lit_len)
        elif kind == 1:  # copy, 1-byte offset
            length4 = (tag >> 2) & 0x07
            copy_len = length4 + 4
            offset = ((tag >> 5) << 8) | c.byte()
            start = len(out) - offset
            for i in range(copy_len):
                out.append(out[start + i])
        elif kind == 2:  # copy, 2-byte offset
            copy_len = (tag >> 2) + 1
            offset = int.from_bytes(c.bytes(2), "little")
            start = len(out) - offset
            for i in range(copy_len):
                out.append(out[start + i])
        else:  # kind == 3, copy, 4-byte offset
            copy_len = (tag >> 2) + 1
            offset = int.from_bytes(c.bytes(4), "little")
            start = len(out) - offset
            for i in range(copy_len):
                out.append(out[start + i])
    assert len(out) == length, f"snappy length mismatch: {len(out)} != {length}"
    return bytes(out)


def _decompress(codec: int, data: bytes) -> bytes:
    if codec == 0:
        return data
    if codec == 1:
        return _snappy_decompress(data)
    if codec == 2:
        return gzip.decompress(data)
    if codec == 6:
        return _zstd_decompress(data)
    raise ValueError(f"Unsupported compression codec id: {codec}")


# --------------------------------------------------------------------------
# RLE / bit-packing hybrid decoder (used for def levels and dict indices)
# --------------------------------------------------------------------------

def _decode_rle_bitpacked_hybrid(buf: bytes, bit_width: int, num_values: int, has_length_prefix: bool):
    c = _Cursor(buf)
    if has_length_prefix:
        c.uvarint()  # length prefix (unused; we stop by num_values)
    out = []
    while len(out) < num_values:
        header = c.uvarint()
        if header & 1:  # bit-packed run
            num_groups = header >> 1
            count = num_groups * 8
            byte_count = (count * bit_width + 7) // 8
            packed = c.bytes(byte_count)
            bit_pos = 0
            for _ in range(count):
                val = 0
                for b in range(bit_width):
                    byte_idx = bit_pos // 8
                    bit_idx = bit_pos % 8
                    if byte_idx < len(packed) and (packed[byte_idx] >> bit_idx) & 1:
                        val |= (1 << b)
                    bit_pos += 1
                out.append(val)
        else:  # RLE run
            run_len = header >> 1
            width_bytes = (bit_width + 7) // 8
            raw = c.bytes(width_bytes) + b"\x00" * (4 - width_bytes)
            val = struct.unpack("<i", raw)[0]
            out.extend([val] * run_len)
    return out[:num_values]


def _bit_width_for(max_val: int) -> int:
    if max_val <= 0:
        return 0
    return max_val.bit_length()


# --------------------------------------------------------------------------
# PLAIN encoding decode per physical type
# --------------------------------------------------------------------------

PHYS_BOOLEAN, PHYS_INT32, PHYS_INT64, PHYS_INT96 = 0, 1, 2, 3
PHYS_FLOAT, PHYS_DOUBLE, PHYS_BYTE_ARRAY, PHYS_FIXED_LEN_BYTE_ARRAY = 4, 5, 6, 7


def _decode_plain(buf: bytes, phys_type: int, count: int):
    c = _Cursor(buf)
    out = []
    if phys_type == PHYS_BOOLEAN:
        bit_pos = 0
        for _ in range(count):
            byte_idx = bit_pos // 8
            bit_idx = bit_pos % 8
            out.append((buf[byte_idx] >> bit_idx) & 1 == 1)
            bit_pos += 1
    elif phys_type == PHYS_INT32:
        for _ in range(count):
            out.append(struct.unpack("<i", c.bytes(4))[0])
    elif phys_type == PHYS_INT64:
        for _ in range(count):
            out.append(struct.unpack("<q", c.bytes(8))[0])
    elif phys_type == PHYS_FLOAT:
        for _ in range(count):
            out.append(struct.unpack("<f", c.bytes(4))[0])
    elif phys_type == PHYS_DOUBLE:
        for _ in range(count):
            out.append(struct.unpack("<d", c.bytes(8))[0])
    elif phys_type == PHYS_BYTE_ARRAY:
        for _ in range(count):
            n = struct.unpack("<I", c.bytes(4))[0]
            out.append(c.bytes(n))
    elif phys_type == PHYS_INT96:
        for _ in range(count):
            out.append(c.bytes(12))
    else:
        raise ValueError(f"Unsupported physical type for PLAIN decode: {phys_type}")
    return out


# --------------------------------------------------------------------------
# Page header parsing (thrift field ids per parquet.thrift)
# --------------------------------------------------------------------------

def _parse_page_header(c: _Cursor) -> dict:
    raw = _read_thrift_struct(c)
    return {
        "type": raw.get(1),
        "uncompressed_page_size": raw.get(2),
        "compressed_page_size": raw.get(3),
        "data_page_header": raw.get(5),
        "dictionary_page_header": raw.get(7),
    }


# --------------------------------------------------------------------------
# Column chunk reader
# --------------------------------------------------------------------------

@dataclass
class ColumnInfo:
    name: str
    phys_type: int
    codec: int
    file_offset: int
    total_compressed_size: int
    num_values: int


def _read_column_values(fbuf: bytes, col: ColumnInfo):
    """Returns a list of python values (None for nulls) of length col.num_values."""
    c = _Cursor(fbuf, col.file_offset)
    end = col.file_offset + col.total_compressed_size

    dictionary = None
    values_out = []

    while c.pos < end and len(values_out) < col.num_values:
        page_start = c.pos
        header = _parse_page_header(c)
        page_data_start = c.pos
        raw_page = fbuf[page_data_start:page_data_start + header["compressed_page_size"]]
        page_bytes = _decompress(col.codec, raw_page)
        c.pos = page_data_start + header["compressed_page_size"]

        if header["type"] == 2:  # DICTIONARY_PAGE
            dph = header["dictionary_page_header"] or {}
            n_dict = dph.get(1, 0)
            dictionary = _decode_plain(page_bytes, col.phys_type, n_dict)
            continue

        if header["type"] == 0:  # DATA_PAGE (v1)
            dh = header["data_page_header"] or {}
            n_values = dh.get(1, 0)
            encoding = dh.get(2)
            def_level_enc = dh.get(3)
            rep_level_enc = dh.get(4)

            pc = _Cursor(page_bytes)

            # repetition levels: absent for non-repeated (flat) columns
            # definition levels: 0/1 hybrid RLE, length-prefixed (v1 data page)
            def_levels = [1] * n_values
            # Heuristic: try to detect presence of a def-level section by
            # checking whether num_values written equals col.num_values for
            # this page; if the column is required (non-nullable) parquet
            # typically stores no def-level section at all. We detect this
            # from whether decoding as int32-length-prefixed RLE consumes a
            # sane number of bytes; if col is required we skip straight to
            # values. This is determined by the caller via col metadata in
            # practice, but pandas/pyarrow marks nearly all columns OPTIONAL,
            # so we default to expecting a def-level section.
            if _COLUMN_REQUIRED.get(col.name, False):
                pass  # no def-level section
            else:
                length_prefix = struct.unpack("<I", pc.bytes(4))[0]
                def_buf = pc.bytes(length_prefix)
                bw = 1  # max def level is 1 for flat optional columns
                def_levels = _decode_rle_bitpacked_hybrid(def_buf, bw, n_values, has_length_prefix=False)

            n_non_null = sum(1 for d in def_levels if d == 1)
            remaining = page_bytes[pc.pos:]

            if encoding in (2, 8):  # PLAIN_DICTIONARY / RLE_DICTIONARY
                bit_width = remaining[0]
                idx_buf = remaining[1:]
                indices = _decode_rle_bitpacked_hybrid(idx_buf, bit_width, n_non_null, has_length_prefix=False)
                vals_iter = iter(dictionary[i] for i in indices)
            elif encoding == 0:  # PLAIN
                decoded = _decode_plain(remaining, col.phys_type, n_non_null)
                vals_iter = iter(decoded)
            else:
                raise ValueError(f"Unsupported value encoding: {encoding}")

            for d in def_levels:
                values_out.append(next(vals_iter) if d == 1 else None)
        else:
            # Unsupported page type (e.g. data page v2) — stop gracefully
            break

        if c.pos <= page_start:
            break  # safety against infinite loop

    return values_out[:col.num_values]


# Populated by read_parquet() before decoding, based on schema repetition_type
_COLUMN_REQUIRED: dict = {}


def _decode_bytes_value(v, converted_type):
    if isinstance(v, (bytes, bytearray)):
        try:
            return v.decode("utf-8")
        except UnicodeDecodeError:
            return v
    return v


def read_parquet(path: str, columns: list | None = None, limit_rows: int | None = None):
    """Reads a flat (non-nested) parquet file written by pandas/pyarrow.

    Returns (column_names: list[str], data: dict[str, list], num_rows: int).
    `columns`, if given, restricts which columns are decoded (much faster).
    `limit_rows`, if given, stops after that many rows across row groups.
    """
    with open(path, "rb") as f:
        fbuf = f.read()

    footer_len = struct.unpack("<I", fbuf[-8:-4])[0]
    footer_buf = fbuf[-(8 + footer_len):-8]
    meta = _read_thrift_struct(_Cursor(footer_buf))

    schema = meta.get(2, [])
    total_num_rows = meta.get(3, 0)
    row_groups = meta.get(4, [])

    # schema[0] is the root; children follow flattened (flat columns only)
    col_schema = {}
    global _COLUMN_REQUIRED
    _COLUMN_REQUIRED = {}
    for elem in schema[1:]:
        name = elem.get(4)
        name = name.decode("utf-8") if isinstance(name, bytes) else name
        rep_type = elem.get(3)  # 0=REQUIRED, 1=OPTIONAL, 2=REPEATED
        phys_type = elem.get(1)
        converted = elem.get(6)
        col_schema[name] = {"phys_type": phys_type, "converted_type": converted}
        _COLUMN_REQUIRED[name] = (rep_type == 0)

    want = set(columns) if columns else set(col_schema.keys())
    data = {name: [] for name in col_schema if name in want}

    rows_so_far = 0
    for rg in row_groups:
        rg_num_rows = rg.get(3, 0)
        col_chunks = rg.get(1, [])
        for cc in col_chunks:
            cmeta = cc.get(3, {})
            path_in_schema = cmeta.get(3, [])
            col_name = path_in_schema[-1] if path_in_schema else None
            col_name = col_name.decode("utf-8") if isinstance(col_name, bytes) else col_name
            if col_name not in want:
                continue
            phys_type = cmeta.get(1)
            codec = cmeta.get(4, 0)
            num_values = cmeta.get(5, 0)
            file_offset = cc.get(2)
            # dictionary_page_offset (11) if present is the true start; else data_page_offset (9)
            start_offset = cmeta.get(11) or cmeta.get(9) or file_offset
            total_compressed_size = cmeta.get(7, 0)
            col = ColumnInfo(
                name=col_name, phys_type=phys_type, codec=codec,
                file_offset=start_offset, total_compressed_size=total_compressed_size,
                num_values=num_values,
            )
            vals = _read_column_values(fbuf, col)
            converted = col_schema[col_name]["converted_type"]
            if phys_type == PHYS_BYTE_ARRAY:
                vals = [_decode_bytes_value(v, converted) for v in vals]
            data[col_name].extend(vals)
        rows_so_far += rg_num_rows
        if limit_rows and rows_so_far >= limit_rows:
            break

    n = len(next(iter(data.values()))) if data else 0
    if limit_rows:
        data = {k: v[:limit_rows] for k, v in data.items()}
        n = min(n, limit_rows)
    return list(data.keys()), data, n

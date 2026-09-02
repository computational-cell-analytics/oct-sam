"""Reader for Heidelberg Heyex OCT volumes in VOL format.

The file layout is a fixed 2048 byte file header, the SLO fundus image, and then one record per
B-scan. Each record is a B-scan header followed by ``size_x * size_z`` little-endian float32
intensities. Files can contain padding after the last B-scan, so the layout is derived from the
header instead of the file size.

``read_heyex_vol`` replaces ``eyepy.import_heyex_vol(path).data``. It applies the same intensity
transform as ``eyepy.core.utils.from_vol_intensity``, so the result is identical.
"""
import struct

import numpy as np
from skimage.util import img_as_ubyte

# Size of the file header in bytes. The B-scan records start after the SLO image.
_FILE_HEADER_SIZE = 2048

# Byte offsets of the header fields, taken from the Heyex VOL format description.
_OFFSET_VERSION = 0
_OFFSET_SIZES = 12          # size_x, num_bscans, size_z as int32
_OFFSET_SCALES = 24         # scale_x, distance, scale_z as float64, in millimeter
_OFFSET_SLO_SIZES = 48      # size_x_slo, size_y_slo as int32
_OFFSET_SLO_SCALES = 56     # scale_x_slo, scale_y_slo as float64
_OFFSET_SCAN_POSITION = 84  # "OD" or "OS"
_OFFSET_PATTERN = 96        # scan_pattern, bscan_hdr_size as int32


def read_vol_header(path: str) -> dict:
    """Read the file header of a Heyex VOL volume.

    Args:
        path: File path to the VOL volume.

    Returns:
        Header fields. The distances scale_x, scale_z and distance are in millimeter.
    """
    with open(path, "rb") as f:
        header = f.read(_FILE_HEADER_SIZE)
    if len(header) < _FILE_HEADER_SIZE:
        raise ValueError(f"{path} is too small to be a VOL volume.")

    size_x, num_bscans, size_z = struct.unpack_from("<iii", header, _OFFSET_SIZES)
    scale_x, distance, scale_z = struct.unpack_from("<ddd", header, _OFFSET_SCALES)
    size_x_slo, size_y_slo = struct.unpack_from("<ii", header, _OFFSET_SLO_SIZES)
    scale_x_slo, scale_y_slo = struct.unpack_from("<dd", header, _OFFSET_SLO_SCALES)
    scan_pattern, bscan_hdr_size = struct.unpack_from("<ii", header, _OFFSET_PATTERN)

    return {
        "version": header[_OFFSET_VERSION:_OFFSET_VERSION + 12].split(b"\x00")[0].decode(),
        "size_x": size_x,
        "num_bscans": num_bscans,
        "size_z": size_z,
        "scale_x": scale_x,
        "distance": distance,
        "scale_z": scale_z,
        "size_x_slo": size_x_slo,
        "size_y_slo": size_y_slo,
        "scale_x_slo": scale_x_slo,
        "scale_y_slo": scale_y_slo,
        "scan_position": header[_OFFSET_SCAN_POSITION:_OFFSET_SCAN_POSITION + 4].split(b"\x00")[0].decode(),
        "scan_pattern": scan_pattern,
        "bscan_hdr_size": bscan_hdr_size,
    }


def vol_intensity_transform(data: np.ndarray) -> np.ndarray:
    """Map raw VOL intensities to the contrast used by the Heyex software.

    Args:
        data: Raw float32 intensities. Values are modified in place.

    Returns:
        Transformed intensities as uint8.
    """
    # The export marks absent data with the largest representable float32.
    missing = data == np.finfo(np.float32).max
    valid = data <= 1

    data[valid] = (np.log(data[valid] + 2.44e-04) + 8.3) / 8.285
    data[missing] = 0
    return img_as_ubyte(np.clip(data, 0, 1)).astype(np.ubyte)


def read_heyex_vol(path: str) -> np.ndarray:
    """Read the B-scans of a Heyex VOL volume.

    Args:
        path: File path to the VOL volume.

    Returns:
        The B-scans as uint8 array of shape (num_bscans, size_z, size_x).
    """
    header = read_vol_header(path)
    size_x, size_z = header["size_x"], header["size_z"]
    num_bscans, bscan_hdr_size = header["num_bscans"], header["bscan_hdr_size"]

    raw = np.fromfile(path, dtype=np.uint8)
    bscan_bytes = size_x * size_z * 4
    data_offset = _FILE_HEADER_SIZE + header["size_x_slo"] * header["size_y_slo"]
    record_size = bscan_hdr_size + bscan_bytes

    required = data_offset + num_bscans * record_size
    if raw.size < required:
        raise ValueError(f"{path} is truncated: expected at least {required} bytes, got {raw.size}.")

    bscans = []
    for i in range(num_bscans):
        start = data_offset + i * record_size + bscan_hdr_size
        bscan = raw[start:start + bscan_bytes].view(np.float32).reshape(size_z, size_x)
        bscans.append(bscan)

    return vol_intensity_transform(np.stack(bscans, axis=0).astype(np.float32))

import unittest

import numpy as np


HEIGHT, WIDTH = 60, 120
BAND_HEIGHT, START_ROW = 6, 5


def _band_rows(index):
    row = START_ROW + index * BAND_HEIGHT
    return slice(row, row + BAND_HEIGHT)


def _stacked_bands(n_bands):
    """Build a segmentation of n_bands full-width bands with instance IDs 1..n_bands."""
    seg = np.zeros((HEIGHT, WIDTH), dtype=np.uint32)
    for i in range(n_bands):
        seg[_band_rows(i)] = i + 1
    return seg


def _single_id(seg, mask):
    ids = np.unique(seg[mask])
    assert len(ids) == 1, f"expected one ID, got {ids}"
    return int(ids[0])


class TestAssignLayerIds(unittest.TestCase):
    def test_full_stack_keeps_order(self):
        from oct_tools.postprocessing import assign_layer_ids
        seg = assign_layer_ids(_stacked_bands(7))
        self.assertEqual([int(i) for i in np.unique(seg)[1:]], list(range(1, 8)))
        for i in range(7):
            self.assertEqual(_single_id(seg, np.s_[_band_rows(i), :]), i + 1)

    def test_bottom_band_is_rpe_for_six_layers(self):
        from oct_tools.postprocessing import assign_layer_ids
        seg = _stacked_bands(6)
        bottom = seg == 6
        seg = assign_layer_ids(seg)
        self.assertEqual(_single_id(seg, bottom), 7)

    def test_bottom_band_is_rpe_without_a_complete_column(self):
        # No column contains every ID, so find_layer_order falls back to the median row.
        seg = _stacked_bands(6)
        seg[_band_rows(2), WIDTH // 2:] = 0  # band 3 covers the left half only
        seg[_band_rows(4), :WIDTH // 2] = 0  # band 5 covers the right half only

        from oct_tools.layer_information import find_layer_order
        from oct_tools.postprocessing import assign_layer_ids
        self.assertEqual(find_layer_order(seg), [1, 2, 3, 4, 5, 6])

        bottom = seg == 6
        seg = assign_layer_ids(seg)
        self.assertEqual(_single_id(seg, bottom), 7)

    def test_empty_segmentation_is_left_alone(self):
        # find_layer_order used to return [] here, and assign_layer_id then hit max([]).
        from oct_tools.layer_information import find_layer_order
        from oct_tools.postprocessing import assign_layer_ids
        seg = np.zeros((HEIGHT, WIDTH), dtype=np.uint32)
        self.assertIsNone(find_layer_order(seg))
        np.testing.assert_array_equal(assign_layer_ids(seg), seg)

    def test_unordered_instance_ids(self):
        from oct_tools.postprocessing import assign_layer_ids
        seg = _stacked_bands(3)
        seg[seg == 1], seg[seg == 3] = 30, 10  # top band 30, bottom band 10
        seg = assign_layer_ids(seg)
        self.assertEqual(_single_id(seg, np.s_[_band_rows(0), :]), 1)
        self.assertEqual(_single_id(seg, np.s_[_band_rows(2), :]), 7)


if __name__ == "__main__":
    unittest.main()

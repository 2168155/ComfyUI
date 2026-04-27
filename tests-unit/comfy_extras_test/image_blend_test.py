import sys
from unittest.mock import patch, MagicMock

# `comfy.model_management` initializes the GPU at module import time, which
# fails in CPU-only environments. Stub it out before any `comfy.*` imports
# load it transitively. We don't use it in these tests.
sys.modules.setdefault("comfy.model_management", MagicMock())

import torch  # noqa: E402

# Mock nodes module to prevent CUDA initialization during import
mock_nodes = MagicMock()
mock_nodes.MAX_RESOLUTION = 16384

# Mock server module for PromptServer
mock_server = MagicMock()

with patch.dict("sys.modules", {"nodes": mock_nodes, "server": mock_server}):
    from comfy_extras.nodes_post_processing import Blend  # noqa: E402


class TestImageBlend:
    """Regression tests for the ImageBlend node, especially channel-count handling."""

    def create_test_image(self, batch_size=1, height=64, width=64, channels=3):
        return torch.rand(batch_size, height, width, channels)

    def test_same_shape_rgb(self):
        """Baseline: identical RGB inputs produce an RGB output."""
        image1 = self.create_test_image(channels=3)
        image2 = self.create_test_image(channels=3)
        result = Blend.execute(image1, image2, 0.5, "normal")
        assert result[0].shape == (1, 64, 64, 3)

    def test_rgb_plus_rgba(self):
        """RGB image1 + RGBA image2 should pad image1 to 4 channels."""
        image1 = self.create_test_image(channels=3)
        image2 = self.create_test_image(channels=4)
        result = Blend.execute(image1, image2, 0.5, "normal")
        assert result[0].shape == (1, 64, 64, 4)

    def test_rgba_plus_rgb(self):
        """RGBA image1 + RGB image2 should pad image2 to 4 channels."""
        image1 = self.create_test_image(channels=4)
        image2 = self.create_test_image(channels=3)
        result = Blend.execute(image1, image2, 0.5, "normal")
        assert result[0].shape == (1, 64, 64, 4)

    def test_channel_gap_larger_than_one(self):
        """Channel-count gap > 1 (e.g. 3 vs 5) should not raise.

        This is the exact runtime error reported in CORE-103:
        'The size of tensor a (5) must match the size of tensor b (3) at
        non-singleton dimension 3'.
        """
        image1 = self.create_test_image(channels=3)
        image2 = self.create_test_image(channels=5)
        result = Blend.execute(image1, image2, 0.5, "multiply")
        assert result[0].shape == (1, 64, 64, 5)

    def test_different_size_and_channels(self):
        """Different spatial size AND different channel counts should both be reconciled."""
        image1 = self.create_test_image(height=64, width=64, channels=3)
        image2 = self.create_test_image(height=32, width=32, channels=4)
        result = Blend.execute(image1, image2, 0.5, "screen")
        assert result[0].shape == (1, 64, 64, 4)

    def test_all_blend_modes_with_channel_mismatch(self):
        """Every blend mode should work with mismatched channel counts."""
        image1 = self.create_test_image(channels=3)
        image2 = self.create_test_image(channels=4)
        for mode in [
            "normal",
            "multiply",
            "screen",
            "overlay",
            "soft_light",
            "difference",
        ]:
            result = Blend.execute(image1, image2, 0.5, mode)
            assert result[0].shape == (1, 64, 64, 4), (
                f"blend mode {mode} produced wrong shape"
            )

    def test_output_clamped(self):
        """Output values should always be clamped to [0, 1]."""
        image1 = self.create_test_image(channels=3)
        image2 = self.create_test_image(channels=4)
        result = Blend.execute(image1, image2, 0.5, "normal")
        assert result[0].min() >= 0.0
        assert result[0].max() <= 1.0

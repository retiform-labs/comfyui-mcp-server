"""
Unit tests for randomized batch generation parameters.

Covers:
- _resolve_param: scalar, range, choice pool, "random", "random_unsafe"
- _resolve_lora_config: fixed mode, auto mode (with mocked available LoRAs)
"""
import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Install the shared pass-through MCP double BEFORE importing the server.
sys.path.insert(0, os.path.dirname(__file__))
from _mcpdouble import install as _install_mcp
_install_mcp()

# Ensure the src/ layout package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from comfyui_mcp_server.server import _resolve_param, _resolve_lora_config


class TestResolveParam(unittest.TestCase):
    """Test parameter resolution: scalar, range, choice, random"""

    def test_scalar_int_unchanged(self):
        """Scalar int: passed through"""
        self.assertEqual(_resolve_param(20, "steps"), 20)
        self.assertEqual(_resolve_param(1024, "width"), 1024)

    def test_scalar_float_unchanged(self):
        """Scalar float: passed through"""
        self.assertEqual(_resolve_param(7.0, "cfg_scale"), 7.0)

    def test_scalar_string_unchanged(self):
        """Scalar string: passed through (except 'random')"""
        self.assertEqual(_resolve_param("dpmpp_2m", "sampler"), "dpmpp_2m")

    def test_scalar_bool_unchanged(self):
        """Scalar bool: passed through"""
        self.assertEqual(_resolve_param(True, "foo"), True)
        self.assertEqual(_resolve_param(False, "foo"), False)

    def test_none_returns_none(self):
        """None input: returns None"""
        self.assertIsNone(_resolve_param(None, "steps"))

    def test_list_two_numeric_steps_int_range(self):
        """Range [min, max] for steps/width/height: returns int"""
        result = _resolve_param([20, 40], "steps")
        self.assertIsInstance(result, int)
        self.assertGreaterEqual(result, 20)
        self.assertLessEqual(result, 40)

    def test_list_two_numeric_cfg_float_range(self):
        """Range [min, max] for cfg_scale: returns float"""
        result = _resolve_param([0.5, 1.5], "cfg_scale")
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.5)
        self.assertLessEqual(result, 1.5)

    def test_list_choice_pool(self):
        """Choice pool: random selection from elements"""
        pool = ["euler", "dpmpp_2m", "heun"]
        result = _resolve_param(pool, "sampler")
        self.assertIn(result, pool)

    def test_random_sampler_safe(self):
        """"random" for sampler: from safe pool"""
        result = _resolve_param("random", "sampler")
        # Safe pool samplers (22 total)
        safe_pool = [
            "euler", "euler_ancestral",
            "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu",
            "dpmpp_3m_sde", "dpmpp_3m_sde_gpu",
            "dpmpp_2m_cfg_pp", "dpmpp_2m_sde_heun", "dpmpp_2m_sde_heun_gpu",
            "heun", "heunpp2",
            "dpmpp_2s_ancestral", "dpmpp_2s_ancestral_cfg_pp",
            "dpmpp_sde", "dpmpp_sde_gpu",
            "dpm_2", "dpm_2_ancestral", "lms", "dpm_fast", "dpm_adaptive",
            "uni_pc", "uni_pc_bh2", "ddim"
        ]
        self.assertIn(result, safe_pool)

    def test_random_unsafe_sampler_all_pool(self):
        """"random_unsafe" for sampler: includes unsafe samplers"""
        result = _resolve_param("random_unsafe", "sampler")
        all_pool = [
            "euler", "euler_ancestral",
            "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu",
            "dpmpp_3m_sde", "dpmpp_3m_sde_gpu",
            "dpmpp_2m_cfg_pp", "dpmpp_2m_sde_heun", "dpmpp_2m_sde_heun_gpu",
            "heun", "heunpp2",
            "dpmpp_2s_ancestral", "dpmpp_2s_ancestral_cfg_pp",
            "dpmpp_sde", "dpmpp_sde_gpu",
            "dpm_2", "dpm_2_ancestral", "lms", "dpm_fast", "dpm_adaptive",
            "uni_pc", "uni_pc_bh2", "ddim",
            "lcm", "ddpm",
            "gradient_estimation", "gradient_estimation_cfg_pp",
            "seeds_2", "seeds_3",
            "sa_solver", "sa_solver_pece",
            "exp_heun_2_x0", "exp_heun_2_x0_sde", "er_sde",
            "ipndm", "ipndm_v", "deis",
            "res_multistep", "res_multistep_cfg_pp",
            "res_multistep_ancestral", "res_multistep_ancestral_cfg_pp"
        ]
        self.assertIn(result, all_pool)

    def test_random_scheduler(self):
        """"random" for scheduler: from all schedulers"""
        result = _resolve_param("random", "scheduler")
        schedulers = [
            "simple", "sgm_uniform", "karras", "exponential",
            "ddim_uniform", "beta", "normal", "linear_quadratic", "kl_optimal"
        ]
        self.assertIn(result, schedulers)

    def test_random_seed_returns_int(self):
        """"random" for seed: returns 32-bit int"""
        result = _resolve_param("random", "seed")
        self.assertIsInstance(result, int)
        self.assertGreaterEqual(result, 0)
        self.assertLess(result, 2**32)

    def test_empty_list_returns_none(self):
        """Empty list: returns None"""
        self.assertIsNone(_resolve_param([], "sampler"))

    def test_two_element_non_numeric_choice(self):
        """Two-element string list: choice, not range"""
        pool = ["euler", "dpmpp_2m"]
        result = _resolve_param(pool, "sampler")
        self.assertIn(result, pool)

    def test_dict_returns_as_is(self):
        """Dict: returned as-is (handled by caller)"""
        spec = {"mode": "auto", "count": [0, 2]}
        self.assertEqual(_resolve_param(spec, "loras"), spec)

    def test_random_unsafe_for_other_params(self):
        """"random_unsafe" for unsupported params: returned as-is"""
        self.assertEqual(_resolve_param("random_unsafe", "steps"), "random_unsafe")


class TestResolveLoraConfig(unittest.TestCase):
    """Test LoRA randomization resolution"""

    # ==========================================================================
    # Fixed mode
    # ==========================================================================

    def test_fixed_none_returns_none(self):
        """None input: returns None"""
        self.assertIsNone(_resolve_lora_config(None))

    def test_fixed_list_pass_through(self):
        """List: pass through unchanged"""
        lora_list = [{"name": "v1", "weight": 0.8}]
        self.assertEqual(_resolve_lora_config(lora_list), lora_list)

    def test_fixed_single_scalar_weight(self):
        """Fixed mode with scalar weight"""
        spec = {
            "mode": "fixed",
            "loras": [{"name": "v1", "weight": 0.85}]
        }
        result = _resolve_lora_config(spec)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "v1")
        self.assertAlmostEqual(result[0]["weight"], 0.85)

    def test_fixed_single_range_weight(self):
        """Fixed mode with [min, max] weight: random uniform"""
        spec = {
            "mode": "fixed",
            "loras": [{"name": "v1", "weight": [0.3, 1.2]}]
        }
        result = _resolve_lora_config(spec)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "v1")
        self.assertIsInstance(result[0]["weight"], float)
        self.assertGreaterEqual(result[0]["weight"], 0.3)
        self.assertLessEqual(result[0]["weight"], 1.2)

    def test_fixed_multiple_loras(self):
        """Fixed mode with multiple LoRAs, mixed weight specs"""
        spec = {
            "mode": "fixed",
            "loras": [
                {"name": "v1", "weight": 0.8},
                {"name": "v2", "weight": [0.2, 0.9]},
            ]
        }
        result = _resolve_lora_config(spec)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["name"], "v1")
        self.assertAlmostEqual(result[0]["weight"], 0.8)
        self.assertEqual(result[1]["name"], "v2")
        self.assertIsInstance(result[1]["weight"], float)

    def test_fixed_empty_loras_returns_none(self):
        """Fixed mode with empty loras list: returns None"""
        spec = {"mode": "fixed", "loras": []}
        self.assertIsNone(_resolve_lora_config(spec))

    def test_fixed_missing_name_skipped(self):
        """Fixed mode: entries without name are skipped"""
        spec = {
            "mode": "fixed",
            "loras": [
                {"name": "v1", "weight": 0.8},
                {"weight": 0.5},  # no name
            ]
        }
        result = _resolve_lora_config(spec)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "v1")

    def test_fixed_default_weight(self):
        """Fixed mode: default weight 0.8 when not specified"""
        spec = {
            "mode": "fixed",
            "loras": [{"name": "v1"}]
        }
        result = _resolve_lora_config(spec)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result[0]["weight"], 0.8)

    # ==========================================================================
    # Auto mode (with mocked _get_available_loras)
    # ==========================================================================

    @patch("comfyui_mcp_server.server._get_available_loras")
    def test_auto_zero_count_returns_none(self, mock_get_loras):
        """Auto mode with count=0: returns None"""
        mock_get_loras.return_value = ["lora1", "lora2"]
        spec = {"mode": "auto", "count": 0}
        self.assertIsNone(_resolve_lora_config(spec))

    @patch("comfyui_mcp_server.server._get_available_loras")
    def test_auto_count_one(self, mock_get_loras):
        """Auto mode with count=1: returns single LoRA"""
        mock_get_loras.return_value = ["lora1", "lora2", "lora3"]
        spec = {"mode": "auto", "count": 1, "weight": [0.3, 1.2]}
        result = _resolve_lora_config(spec)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertIn(result[0]["name"], ["lora1", "lora2", "lora3"])
        self.assertIsInstance(result[0]["weight"], float)

    @patch("comfyui_mcp_server.server._get_available_loras")
    def test_auto_count_range(self, mock_get_loras):
        """Auto mode with count [min,max]: returns correct range"""
        mock_get_loras.return_value = ["a", "b", "c", "d"]
        spec = {"mode": "auto", "count": [1, 3], "weight": 0.8}
        result = _resolve_lora_config(spec)
        self.assertIsNotNone(result)
        self.assertGreaterEqual(len(result), 1)
        self.assertLessEqual(len(result), 3)
        for item in result:
            self.assertAlmostEqual(item["weight"], 0.8)

    @patch("comfyui_mcp_server.server._get_available_loras")
    def test_auto_pooled_selection(self, mock_get_loras):
        """Auto mode with explicit pool: selects only from pool"""
        mock_get_loras.return_value = ["lora1", "lora2", "lora3", "lora4"]
        spec = {"mode": "auto", "count": 2, "pool": ["lora1", "lora2"], "weight": 0.7}
        result = _resolve_lora_config(spec)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)
        for item in result:
            self.assertIn(item["name"], ["lora1", "lora2"])
            self.assertAlmostEqual(item["weight"], 0.7)

    @patch("comfyui_mcp_server.server._get_available_loras")
    def test_auto_count_exceeds_pool(self, mock_get_loras):
        """Auto mode: count capped by pool size"""
        mock_get_loras.return_value = ["only"]
        spec = {"mode": "auto", "count": 5, "weight": 0.9}
        result = _resolve_lora_config(spec)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "only")

    @patch("comfyui_mcp_server.server._get_available_loras")
    def test_auto_empty_available(self, mock_get_loras):
        """Auto mode: no LoRAs available returns None"""
        mock_get_loras.return_value = []
        spec = {"mode": "auto", "count": 1, "weight": 0.8}
        self.assertIsNone(_resolve_lora_config(spec))

    @patch("comfyui_mcp_server.server._get_available_loras")
    def test_auto_no_overlap_with_pool(self, mock_get_loras):
        """Auto mode: pool with no overlap returns None"""
        mock_get_loras.return_value = ["lora1", "lora2"]
        spec = {"mode": "auto", "count": 1, "pool": ["lora3"], "weight": 0.8}
        self.assertIsNone(_resolve_lora_config(spec))


class TestIntegrationRoundTrip(unittest.TestCase):
    """Integration: verify randomization produces valid params"""

    def test_sampler_random_bounds(self):
        """100 random samplers stay within declared pools"""
        safe_pool = [
            "euler", "euler_ancestral",
            "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu",
            "dpmpp_3m_sde", "dpmpp_3m_sde_gpu",
            "dpmpp_2m_cfg_pp", "dpmpp_2m_sde_heun", "dpmpp_2m_sde_heun_gpu",
            "heun", "heunpp2",
            "dpmpp_2s_ancestral", "dpmpp_2s_ancestral_cfg_pp",
            "dpmpp_sde", "dpmpp_sde_gpu",
            "dpm_2", "dpm_2_ancestral", "lms", "dpm_fast", "dpm_adaptive",
            "uni_pc", "uni_pc_bh2", "ddim"
        ]
        for _ in range(100):
            result = _resolve_param("random", "sampler")
            self.assertIn(result, safe_pool)

    def test_steps_range_bounds(self):
        """100 random steps stay within range"""
        for _ in range(100):
            result = _resolve_param([10, 50], "steps")
            self.assertIsInstance(result, int)
            self.assertGreaterEqual(result, 10)
            self.assertLessEqual(result, 50)

    def test_cfg_range_bounds(self):
        """100 random CFG values stay within range"""
        for _ in range(100):
            result = _resolve_param([3.0, 12.0], "cfg_scale")
            self.assertIsInstance(result, float)
            self.assertGreaterEqual(result, 3.0)
            self.assertLessEqual(result, 12.0)


if __name__ == "__main__":
    unittest.main()

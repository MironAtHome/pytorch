# Owner(s): ["module: inductor"]

import unittest
from unittest.mock import call, MagicMock, patch

import torch
from torch._inductor.analysis.device_info import (
    _get_amd_smi,
    _get_pynvml,
    datasheet_tops,
    DeviceInfo,
    DeviceSpec,
    lookup_device_info,
)
from torch.testing._internal.common_utils import run_tests, TestCase


class TestDeviceInfo(TestCase):
    def _reset_cache(self):
        import torch._inductor.analysis.device_info as device_info_module

        device_info_module._pynvml_cache = None
        device_info_module._pynvml_initialized = False
        device_info_module._amd_smi_cache = None
        device_info_module._amd_smi_name = None

    def setUp(self):
        self._reset_cache()

    def tearDown(self):
        self._reset_cache()

    def test_lookup_device_info(self):
        h100_info = lookup_device_info("NVIDIA H100")
        self.assertIsNotNone(h100_info)
        if h100_info is not None:
            self.assertEqual(h100_info.dram_gb, 80)
            self.assertIn(torch.float32, h100_info.tops)

        unknown_info = lookup_device_info("Unknown Device")
        self.assertIsNone(unknown_info)

    def test_datasheet_tops_function(self):
        with patch("torch.cuda.get_device_name") as mock_get_device_name:
            mock_get_device_name.return_value = "NVIDIA H100"
            tops = datasheet_tops(torch.float32)
            self.assertIsNotNone(tops)
            self.assertEqual(tops, 67.0)

            tops_tf32 = datasheet_tops(torch.float32, is_tf32=True)
            self.assertEqual(tops_tf32, 989.0)

            mock_get_device_name.return_value = "Unknown Device"
            tops_unknown = datasheet_tops(torch.float32)
            self.assertIsNone(tops_unknown)

            mock_get_device_name.return_value = None
            tops_no_device = datasheet_tops(torch.float32)
            self.assertIsNone(tops_no_device)

    def test_lazy_pynvml_import(self):
        import importlib

        import torch._inductor.analysis.device_info as device_info_module

        original_cache = device_info_module._pynvml_cache
        original_initialized = device_info_module._pynvml_initialized

        try:
            device_info_module._pynvml_cache = None
            device_info_module._pynvml_initialized = False

            importlib.reload(device_info_module)

            with patch("builtins.__import__") as mock_import:
                mock_pynvml_module = MagicMock()
                mock_import.return_value = mock_pynvml_module

                pynvml = device_info_module._get_pynvml()
                self.assertEqual(pynvml, mock_pynvml_module)
                self.assertTrue(mock_import.called)

            device_info_module._pynvml_cache = None
            device_info_module._pynvml_initialized = False

            with patch(
                "builtins.__import__", side_effect=ImportError("pynvml not found")
            ):
                pynvml = device_info_module._get_pynvml()
                self.assertIsNone(pynvml)

        finally:
            device_info_module._pynvml_cache = original_cache
            device_info_module._pynvml_initialized = original_initialized

    @patch("torch._inductor.analysis.device_info._get_pynvml")
    def test_hardware_lookup_clock_hz_success(self, mock_get_pynvml):
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlInit = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = "mock_handle"
        mock_pynvml.nvmlDeviceGetClockInfo.return_value = 1500
        mock_pynvml.NVML_CLOCK_SM = "clock_key"
        mock_pynvml.nvmlShutdown = MagicMock()
        mock_get_pynvml.return_value = mock_pynvml

        result = DeviceInfo._hardware_lookup_clock_hz()
        self.assertEqual(result, 1500 * 1e6)

    def test_lazy_pynvml_import_caching(self):
        with patch("builtins.__import__") as mock_import:
            mock_pynvml_module = MagicMock()
            mock_import.return_value = mock_pynvml_module

            pynvml1 = _get_pynvml()
            self.assertEqual(pynvml1, mock_pynvml_module)
            self.assertEqual(mock_import.call_count, 1)

            pynvml2 = _get_pynvml()
            self.assertEqual(pynvml2, mock_pynvml_module)
            self.assertEqual(mock_import.call_count, 1)

            self.assertEqual(pynvml1, pynvml2)

    def test_hardware_lookup_exception_handling(self):
        with (
            patch("torch.version.hip", None),
            patch(
                "torch.cuda.get_device_properties", side_effect=Exception("CUDA Error")
            ),
            patch(
                "torch._inductor.analysis.device_info._get_pynvml"
            ) as mock_get_pynvml,
        ):
            mock_pynvml = MagicMock()
            mock_pynvml.nvmlInit.side_effect = Exception("NVML Error")
            mock_get_pynvml.return_value = mock_pynvml

            # Test direct hardware lookup methods, not the generic lookup methods
            result = DeviceInfo._hardware_lookup_sm_count()
            self.assertIsNone(result)

            result = DeviceInfo._hardware_lookup_clock_hz()
            self.assertIsNone(result)

    def test_device_mapping_aliases(self):
        mi300x_direct = lookup_device_info("AMD MI300X")
        mi300x_alias = lookup_device_info("AMD INSTINCT MI300X")
        self.assertEqual(mi300x_direct, mi300x_alias)

        mi210x_direct = lookup_device_info("AMD MI210X")
        mi210x_alias = lookup_device_info("AMD INSTINCT MI210X")
        self.assertEqual(mi210x_direct, mi210x_alias)

    def setUp_amd(self):
        import torch._inductor.analysis.device_info as device_info_module

        device_info_module._amd_smi_cache = None
        device_info_module._amd_smi_name = None

    def test_lazy_amd_smi_import_success(self):
        self.setUp_amd()

        with patch("builtins.__import__") as mock_import:
            mock_amd_smi_module = MagicMock()

            def mock_import_func(module_name):
                if module_name == "amdsmi":
                    return mock_amd_smi_module
                raise ImportError(f"No module named '{module_name}'")

            mock_import.side_effect = mock_import_func

            amd_smi = _get_amd_smi()
            self.assertEqual(amd_smi, mock_amd_smi_module)

    def test_lazy_amd_smi_import_failure(self):
        """Test AMD SMI library import failure for all libraries."""
        self.setUp_amd()

        with patch(
            "builtins.__import__", side_effect=ImportError("No AMD library found")
        ):
            amd_smi = _get_amd_smi()
            self.assertIsNone(amd_smi)

    def test_lazy_amd_smi_import_caching(self):
        """Test that AMD SMI import is cached and not repeated."""
        self.setUp_amd()

        with patch("builtins.__import__") as mock_import:
            mock_amd_smi_module = MagicMock()

            def mock_import_func(module_name):
                if module_name == "rocm_smi":
                    return mock_amd_smi_module
                raise ImportError(f"No module named '{module_name}'")

            mock_import.side_effect = mock_import_func

            amd_smi1 = _get_amd_smi()
            self.assertEqual(amd_smi1, mock_amd_smi_module)

            amd_smi2 = _get_amd_smi()
            self.assertEqual(amd_smi2, mock_amd_smi_module)

            self.assertEqual(amd_smi1, amd_smi2)

            expected_calls = [
                call("amdsmi"),
                call("rocm_smi"),
            ]
            mock_import.assert_has_calls(expected_calls, any_order=False)

    @patch("torch.version.hip", "some_hip_version")
    @patch("torch._inductor.analysis.device_info._get_amd_smi")
    def test_amd_hardware_lookup_clock_hz_success(self, mock_get_amd_smi):
        """Test successful AMD clock frequency lookup."""
        mock_amd_smi = MagicMock()
        mock_amd_smi.rsmi_init = MagicMock()
        mock_amd_smi.rsmi_dev_gpu_clk_freq_get.return_value = 2100
        mock_amd_smi.RSMI_CLK_TYPE_SYS = "system_clock"
        mock_amd_smi.rsmi_shut_down = MagicMock()
        mock_get_amd_smi.return_value = mock_amd_smi

        result = DeviceInfo._amd_hardware_lookup_clock_hz()
        self.assertEqual(result, 2100 * 1e6)
        mock_amd_smi.rsmi_dev_gpu_clk_freq_get.assert_called_once_with(
            0, "system_clock"
        )

    @patch("torch.version.hip", "some_hip_version")
    @patch("torch._inductor.analysis.device_info._get_amd_smi")
    def test_amd_hardware_lookup_dram_bw_gbs_not_implemented(self, mock_get_amd_smi):
        """Test AMD memory bandwidth lookup (not implemented)."""
        mock_amd_smi = MagicMock()
        mock_amd_smi.rsmi_init = MagicMock()
        mock_amd_smi.rsmi_shut_down = MagicMock()
        mock_get_amd_smi.return_value = mock_amd_smi

        result = DeviceInfo._amd_hardware_dram_bw_gbs()
        self.assertIsNone(result)

    def test_amd_device_mapping_entries(self):
        """Test that AMD devices are properly represented in device mapping."""
        mi300x = lookup_device_info("AMD MI300X")
        self.assertIsNotNone(mi300x)
        if mi300x is not None:
            self.assertEqual(mi300x.dram_gb, 192.0)
            self.assertEqual(mi300x.dram_bw_gbs, 5300.0)
            self.assertIn(torch.float32, mi300x.tops)

        mi300x_instinct = lookup_device_info("AMD INSTINCT MI300X")
        self.assertEqual(mi300x, mi300x_instinct)

        mi300a = lookup_device_info("AMD MI300A")
        self.assertIsNotNone(mi300a)
        if mi300a is not None:
            self.assertEqual(mi300a.dram_gb, 128.0)
            self.assertEqual(mi300a.dram_bw_gbs, 5300.0)

        mi210x = lookup_device_info("AMD MI210X")
        self.assertIsNotNone(mi210x)
        if mi210x is not None:
            self.assertEqual(mi210x.dram_gb, 64.0)
            self.assertEqual(mi210x.dram_bw_gbs, 1600.0)

        mi210x_instinct = lookup_device_info("AMD INSTINCT MI210X")
        self.assertEqual(mi210x, mi210x_instinct)

    def test_amd_integration_with_datasheet_tops(self):
        """Test datasheet_tops function with AMD devices."""
        with patch("torch.cuda.get_device_name") as mock_get_device_name:
            mock_get_device_name.return_value = "AMD MI300X"

            tops_fp32 = datasheet_tops(torch.float32)
            self.assertEqual(tops_fp32, 163.4)

            tops_fp16 = datasheet_tops(torch.float16)
            self.assertEqual(tops_fp16, 1307.4)

            tops_bf16 = datasheet_tops(torch.bfloat16)
            self.assertEqual(tops_bf16, 1307.4)

            tops_tf32 = datasheet_tops(torch.float32, is_tf32=True)
            self.assertEqual(tops_tf32, 653.7)

    def test_flops_hardware_calculation(self):
        """Test FLOPS calculation now uses datasheet values with clock adjustment."""
        with (
            patch.object(DeviceInfo, "lookup_clock_hz", return_value=1.5e9),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_name", return_value="AMD MI300X"),
        ):
            flops = DeviceInfo.lookup_tops(
                device_name="AMD MI300X", dtype=torch.float32
            )
            # Now uses datasheet value (163.4 TOPS) with clock adjustment
            # Device mapping has clock_hz=2100*1e6, so ratio = 1.5e9 / (2100*1e6) = ~0.714
            datasheet_flops = 163.4 * 1e12
            device_info = lookup_device_info("AMD MI300X")
            if device_info and device_info.clock_hz:
                clock_ratio = 1.5e9 / device_info.clock_hz
                expected_flops = datasheet_flops * clock_ratio
            else:
                expected_flops = datasheet_flops
            self.assertEqual(flops, expected_flops)

    def test_flops_datasheet_calculation(self):
        """Test FLOPS calculation using datasheet TOPS."""
        with (
            patch("torch.cuda.get_device_name") as mock_get_device_name,
            patch("torch.cuda.is_available", return_value=True),
            patch.object(DeviceInfo, "lookup_clock_hz", return_value=1.98e9 / 2),  # Use datasheet clock
        ):
            mock_get_device_name.return_value = "NVIDIA H100"

            flops = DeviceInfo.lookup_tops(
                device_name="NVIDIA H100", dtype=torch.float32
            )
            expected_flops = 67.0 * 1e12 / 2
            self.assertEqual(flops, expected_flops)

    def test_flops_fallback_to_datasheet(self):
        """Test FLOPS fallback to datasheet when hardware lookup fails."""
        with (
            patch("torch.cuda.get_device_name") as mock_get_device_name,
            patch("torch.cuda.is_available", return_value=True),
            patch.object(DeviceInfo, "lookup_clock_hz", return_value=1.98e9 / 2),  # Use datasheet clock
        ):
            mock_get_device_name.return_value = "NVIDIA H100"

            flops = DeviceInfo.lookup_tops(
                device_name="NVIDIA H100", dtype=torch.float32
            )
            expected_flops = 67.0 * 1e12 / 2
            self.assertEqual(flops, expected_flops)

    def test_flops_clock_adjustment_in_fallback(self):
        """Test clock adjustment when falling back to datasheet."""
        custom_device_info = DeviceSpec(
            memory_clock_hz=100,
            tops={torch.float32: 100.0},
            dram_bw_gbs=1000.0,
            dram_gb=16.0,
            sm_count=None,
            clock_hz=1.5e9,
        )

        with (
            patch("torch.cuda.get_device_name") as mock_get_device_name,
            patch(
                "torch._inductor.analysis.device_info.lookup_device_info"
            ) as mock_lookup,
        ):
            mock_get_device_name.return_value = "Custom Device"
            mock_lookup.return_value = custom_device_info

            with patch.object(
                DeviceInfo, "_hardware_lookup_clock_hz", return_value=3.0e9
            ):
                flops = DeviceInfo.lookup_tops(
                    "Custom Device", dtype=torch.float32
                )

                datasheet_flops = 100.0 * 1e12
                clock_ratio = 3.0e9 / 1.5e9
                expected_flops = datasheet_flops * clock_ratio
                self.assertEqual(flops, expected_flops)

    @patch("torch._inductor.analysis.device_info.lookup_device_info")
    def test_flops_clock_adjustment_no_expected_clock(self, mock_lookup):
        """Test fallback behavior when device mapping has None for clock_hz."""
        device_info = DeviceSpec(
            memory_clock_hz=100,
            tops={torch.float32: 100.0},
            dram_bw_gbs=1000.0,
            dram_gb=16.0,
            sm_count=None,
            clock_hz=None,
        )
        mock_lookup.return_value = device_info

        with patch("torch.cuda.get_device_name") as mock_get_device_name:
            mock_get_device_name.return_value = "NVIDIA H100"

            with patch.object(
                DeviceInfo, "_hardware_lookup_clock_hz", return_value=3.0e9
            ):
                flops = DeviceInfo.lookup_tops(
                    "NVIDIA H100", dtype=torch.float32
                )

                expected_flops = 100.0 * 1e12
                self.assertEqual(flops, expected_flops)

    def test_flops_clock_adjustment_none_clock(self):
        """Test fallback behavior when clock lookup returns None."""
        with patch("torch.cuda.get_device_name") as mock_get_device_name:
            mock_get_device_name.return_value = "NVIDIA H100"

            with patch.object(
                DeviceInfo, "_hardware_lookup_clock_hz", return_value=None
            ):
                flops = DeviceInfo.lookup_tops(
                    "NVIDIA H100", dtype=torch.float32
                )

                expected_flops = 67.0 * 1e12
                self.assertEqual(flops, expected_flops)

    def test_flops_no_device_name(self):
        """Test FLOPS calculation when device name is unavailable."""
        with (
            patch("torch.cuda.get_device_name", return_value=None),
            patch("torch.cuda.is_available", return_value=False),
        ):
            # When there's no device name and we force datasheet, it should return None
            with patch(
                "torch._inductor.analysis.device_info.datasheet_tops", return_value=None
            ):
                flops = DeviceInfo.lookup_tops(
                    "NVIDIA H100", dtype=torch.float32
                )
                self.assertIsNone(flops)

            # When cuda is not available, hardware lookup is skipped and datasheet is used
            flops = DeviceInfo.lookup_tops(
                "NVIDIA H100", dtype=torch.float32
            )
            self.assertIsNone(
                flops
            )  # Should be None since cuda.is_available() is False

    def test_flops_unknown_device(self):
        """Test FLOPS calculation with unknown device."""
        with patch("torch.cuda.get_device_name") as mock_get_device_name:
            mock_get_device_name.return_value = "Unknown Device"

            flops = DeviceInfo.lookup_tops(
                "Unknown Device", dtype=torch.float32
            )
            # Should be None for unknown device
            self.assertIsNone(flops)

    def test_flops_partial_hardware_values(self):
        """Test FLOPS calculation with some hardware values missing."""
        with (
            patch("torch.cuda.get_device_name") as mock_get_device_name,
            patch("torch.cuda.is_available", return_value=True),
            patch.object(DeviceInfo, "lookup_clock_hz", return_value=1.98e9 / 2),  # Use datasheet clock
        ):
            mock_get_device_name.return_value = "NVIDIA H100"

            flops = DeviceInfo.lookup_tops(
                device_name="NVIDIA H100", dtype=torch.float32
            )
            expected_flops = 67.0 * 1e12 / 2
            self.assertEqual(flops, expected_flops)

    def test_flops_exception_handling(self):
        """Test FLOPS calculation handles exceptions gracefully."""
        with (
            patch.object(
                DeviceInfo,
                "_hardware_lookup_sm_count",
                side_effect=Exception("Hardware error"),
            ),
            patch("torch.cuda.get_device_name") as mock_get_device_name,
            patch("torch.cuda.is_available", return_value=True),
            patch.object(DeviceInfo, "lookup_clock_hz", return_value=1.98e9 / 2),  # Use datasheet clock
        ):
            mock_get_device_name.return_value = "NVIDIA H100"

            flops = DeviceInfo.lookup_tops(
                "NVIDIA H100", dtype=torch.float32
            )
            expected_flops = 67.0 * 1e12 / 2
            self.assertEqual(flops, expected_flops)

    def test_flops_integration_with_hardware_lookup(self):
        """Test FLOPS integration with datasheet values and clock adjustment."""
        dn = "NVIDIA H100"

        with (
            patch.object(DeviceInfo, "lookup_clock_hz", return_value=1500 * 1e6),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_name", return_value=dn),
        ):
            flops = DeviceInfo.lookup_tops(
                device_name=dn, dtype=torch.float32
            )
            # Now uses datasheet value (67.0 TOPS) with clock adjustment
            # Device mapping has clock_hz=1.98e9, so ratio = 1500*1e6 / 1.98e9 = ~0.7576
            datasheet_flops = 67.0 * 1e12
            device_info = lookup_device_info(dn)
            if device_info and device_info.clock_hz:
                clock_ratio = (1500 * 1e6) / device_info.clock_hz
                expected_flops = datasheet_flops * clock_ratio
            else:
                expected_flops = datasheet_flops
            self.assertEqual(flops, expected_flops)

    @unittest.skipIf(
        False, "pynvml and amdsmi are not available in CI, run these tests locally"
    )
    def test_pynvml_integration(self):
        """Test direct pynvml library integration."""
        try:
            import pynvml

            # Test basic NVML initialization and device access
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            # Test clock frequency retrieval
            sm_clock_mhz = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
            self.assertIsInstance(sm_clock_mhz, int)
            self.assertGreater(sm_clock_mhz, 0)

            # Test memory clock frequency retrieval
            mem_clock_mhz = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
            self.assertIsInstance(mem_clock_mhz, int)
            self.assertGreater(mem_clock_mhz, 0)

            # Test memory bus width retrieval
            bus_width_bits = pynvml.nvmlDeviceGetMemoryBusWidth(handle)
            self.assertIsInstance(bus_width_bits, int)
            self.assertGreater(bus_width_bits, 0)

            # Test bandwidth calculation (same as device_info.py implementation)
            mem_clock_hz = mem_clock_mhz * 1e6
            effective_rate = mem_clock_hz * 2  # GDDR uses DDR so *2
            peak_bw = (effective_rate * bus_width_bits) / 8
            peak_bw_gbs = peak_bw / (1024**3)

            self.assertIsInstance(peak_bw_gbs, float)
            self.assertGreater(peak_bw_gbs, 0)

            pynvml.nvmlShutdown()

        except ImportError:
            self.fail(
                "pynvml library not available - install with 'pip install nvidia-ml-py'"
            )
        except Exception as e:
            self.fail(f"pynvml integration failed: {e}")

    @unittest.skipIf(
        False, "pynvml and amdsmi are not available in CI, run these tests locally"
    )
    @unittest.skipIf(
        not torch.version.hip, "only amd"
    )
    def test_rocm_smi_integration(self):
        """Test direct rocm_smi library integration."""
        try:
            import rocm_smi

            # Test basic ROCm SMI initialization
            rocm_smi.rsmi_init()

            # Test system clock frequency retrieval
            sys_clock_mhz = rocm_smi.rsmi_dev_gpu_clk_freq_get(
                0, rocm_smi.RSMI_CLK_TYPE_SYS
            )
            self.assertIsInstance(sys_clock_mhz, int)
            self.assertGreater(sys_clock_mhz, 0)

            # Test memory clock frequency retrieval
            mem_clock_mhz = rocm_smi.rsmi_dev_gpu_clk_freq_get(
                0, rocm_smi.RSMI_CLK_TYPE_MEM
            )
            self.assertIsInstance(mem_clock_mhz, int)
            self.assertGreater(mem_clock_mhz, 0)

            rocm_smi.rsmi_shut_down()

        except ImportError:
            self.fail("rocm_smi library not available - install ROCm SMI")
        except Exception as e:
            self.fail(f"rocm_smi integration failed: {e}")

    @unittest.skipIf(
        False, "pynvml and amdsmi are not available in CI, run these tests locally"
    )
    @unittest.skipIf(
        not torch.version.hip, "only amd"
    )
    def test_amdsmi_integration(self):
        """Test direct amdsmi library integration."""
        try:
            import amdsmi

            # Test basic AMD SMI initialization
            amdsmi.amdsmi_init()

            # Test device handle retrieval
            device_handle = amdsmi.amdsmi_get_processor_handle(0)
            self.assertIsNotNone(device_handle)

            # Test GPU clock info retrieval
            clock_info = amdsmi.amdsmi_get_gpu_clock_info(device_handle)
            self.assertTrue(hasattr(clock_info, "current_clk"))
            self.assertIsInstance(clock_info.current_clk, int)
            self.assertGreater(clock_info.current_clk, 0)

            # Test GPU memory clock info retrieval
            mem_clock_info = amdsmi.amdsmi_get_gpu_memory_clock_info(device_handle)
            self.assertTrue(hasattr(mem_clock_info, "current_clk"))
            self.assertIsInstance(mem_clock_info.current_clk, int)
            self.assertGreater(mem_clock_info.current_clk, 0)

            amdsmi.amdsmi_shut_down()

        except ImportError:
            self.fail("amdsmi library not available - install AMD SMI")
        except Exception as e:
            self.fail(f"amdsmi integration failed: {e}")

    @unittest.skipIf(
        False, "pynvml and amdsmi are not available in CI, run these tests locally"
    )
    @unittest.skipIf(
        torch.version.hip, "only nvidia"
    )
    def test_pynvml_error_handling(self):
        """Test pynvml error handling for invalid operations."""
        try:
            import pynvml

            pynvml.nvmlInit()

            # Test invalid device index - should raise exception
            with self.assertRaises(Exception):
                pynvml.nvmlDeviceGetHandleByIndex(999)  # Invalid index

            pynvml.nvmlShutdown()

        except ImportError:
            self.skipTest("pynvml library not available")

    @unittest.skipIf(
        False, "pynvml and amdsmi are not available in CI, run these tests locally"
    )
    @unittest.skipIf(
        not torch.version.hip, "only amd"
    )
    def test_amd_smi_error_handling(self):
        """Test AMD SMI error handling for invalid operations."""
        # Try rocm_smi first
        try:
            import rocm_smi

            rocm_smi.rsmi_init()

            # Test invalid device index - should raise exception or return error
            try:
                rocm_smi.rsmi_dev_gpu_clk_freq_get(999, rocm_smi.RSMI_CLK_TYPE_SYS)
            except Exception:
                pass  # Expected for invalid device

            rocm_smi.rsmi_shut_down()
            return

        except ImportError:
            pass

        # Try amdsmi if rocm_smi not available
        try:
            import amdsmi

            amdsmi.amdsmi_init()

            # Test invalid device index - should raise exception
            with self.assertRaises(Exception):
                amdsmi.amdsmi_get_processor_handle(999)  # Invalid index

            amdsmi.amdsmi_shut_down()

        except ImportError:
            self.skipTest("Neither rocm_smi nor amdsmi libraries available")

    @unittest.skipIf(
        False, "pynvml and amdsmi are not available in CI, run these tests locally"
    )
    @unittest.skipIf(
        not torch.version.hip, "only amd"
    )
    def test_library_versions_and_constants_amd(self):
        """Test that expected constants and versions are available."""
        # Test rocm_smi constants
        import rocm_smi

        # Check that required constants exist
        self.assertTrue(hasattr(rocm_smi, "RSMI_CLK_TYPE_SYS"))
        self.assertTrue(hasattr(rocm_smi, "RSMI_CLK_TYPE_MEM"))

        # Check that required functions exist
        self.assertTrue(hasattr(rocm_smi, "rsmi_init"))
        self.assertTrue(hasattr(rocm_smi, "rsmi_dev_gpu_clk_freq_get"))
        self.assertTrue(hasattr(rocm_smi, "rsmi_shut_down"))

        import amdsmi

        # Check that required functions exist
        self.assertTrue(hasattr(amdsmi, "amdsmi_init"))
        self.assertTrue(hasattr(amdsmi, "amdsmi_get_processor_handle"))
        self.assertTrue(hasattr(amdsmi, "amdsmi_get_gpu_clock_info"))
        self.assertTrue(hasattr(amdsmi, "amdsmi_get_gpu_memory_clock_info"))
        self.assertTrue(hasattr(amdsmi, "amdsmi_shut_down"))

    @unittest.skipIf(
        False, "pynvml and amdsmi are not available in CI, run these tests locally"
    )
    @unittest.skipIf(
        torch.version.hip, "only nvidia"
    )
    def test_library_versions_and_constants_nvidia(self):
        """Test that expected constants and versions are available."""
        # Test pynvml constants
        import pynvml

        # Check that required constants exist
        self.assertTrue(hasattr(pynvml, "NVML_CLOCK_SM"))
        self.assertTrue(hasattr(pynvml, "NVML_CLOCK_MEM"))

        # Check that required functions exist
        self.assertTrue(hasattr(pynvml, "nvmlInit"))
        self.assertTrue(hasattr(pynvml, "nvmlDeviceGetHandleByIndex"))
        self.assertTrue(hasattr(pynvml, "nvmlDeviceGetClockInfo"))
        self.assertTrue(hasattr(pynvml, "nvmlDeviceGetMemoryBusWidth"))
        self.assertTrue(hasattr(pynvml, "nvmlShutdown"))

    def test_dram_bw_hardware_calculation(self):
        """Test DRAM bandwidth calculation with memory clock adjustment."""
        with (
            patch.object(DeviceInfo, "lookup_memory_clock_hz", return_value=7e9),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_name", return_value="AMD MI300X"),
        ):
            dram_bw = DeviceInfo.lookup_dram_bw_gbs(device_name="AMD MI300X")
            # Uses datasheet value (5300.0 GB/s) with memory clock adjustment
            # Device mapping has memory_clock_hz=5200*1e6, so ratio = 7e9 / (5200*1e6) = ~1.346
            datasheet_bw = 5300.0
            device_info = lookup_device_info("AMD MI300X")
            if device_info and device_info.memory_clock_hz:
                memory_clock_ratio = 7e9 / device_info.memory_clock_hz
                expected_bw = datasheet_bw * memory_clock_ratio
            else:
                expected_bw = datasheet_bw
            self.assertEqual(dram_bw, expected_bw)

    def test_dram_bw_datasheet_calculation(self):
        """Test DRAM bandwidth calculation using datasheet values."""
        with (
            patch("torch.cuda.get_device_name") as mock_get_device_name,
            patch("torch.cuda.is_available", return_value=True),
            patch.object(DeviceInfo, "lookup_memory_clock_hz", return_value=1.4e10 / 2),  # Use half datasheet memory clock
        ):
            mock_get_device_name.return_value = "NVIDIA H100"

            dram_bw = DeviceInfo.lookup_dram_bw_gbs(device_name="NVIDIA H100")
            expected_bw = 3350 / 2  # Datasheet bandwidth scaled by memory clock ratio
            self.assertEqual(dram_bw, expected_bw)

    def test_dram_bw_fallback_to_datasheet(self):
        """Test DRAM bandwidth fallback to datasheet when hardware lookup fails."""
        with (
            patch("torch.cuda.get_device_name") as mock_get_device_name,
            patch("torch.cuda.is_available", return_value=True),
            patch.object(DeviceInfo, "lookup_memory_clock_hz", return_value=1.4e10 / 2),  # Use half datasheet memory clock
        ):
            mock_get_device_name.return_value = "NVIDIA H100"

            dram_bw = DeviceInfo.lookup_dram_bw_gbs(device_name="NVIDIA H100")
            expected_bw = 3350 / 2  # Datasheet bandwidth scaled by memory clock ratio
            self.assertEqual(dram_bw, expected_bw)

    def test_dram_bw_memory_clock_adjustment_in_fallback(self):
        """Test memory clock adjustment when falling back to datasheet."""
        custom_device_info = DeviceSpec(
            memory_clock_hz=2e9,
            tops={torch.float32: 100.0},
            dram_bw_gbs=1000.0,
            dram_gb=16.0,
            sm_count=None,
            clock_hz=1.5e9,
        )

        with (
            patch("torch.cuda.get_device_name") as mock_get_device_name,
            patch(
                "torch._inductor.analysis.device_info.lookup_device_info"
            ) as mock_lookup,
        ):
            mock_get_device_name.return_value = "Custom Device"
            mock_lookup.return_value = custom_device_info

            with patch.object(
                DeviceInfo, "lookup_memory_clock_hz", return_value=4e9
            ):
                dram_bw = DeviceInfo.lookup_dram_bw_gbs("Custom Device")

                datasheet_bw = 1000.0
                memory_clock_ratio = 4e9 / 2e9
                expected_bw = datasheet_bw * memory_clock_ratio
                self.assertEqual(dram_bw, expected_bw)

    @patch("torch._inductor.analysis.device_info.lookup_device_info")
    def test_dram_bw_memory_clock_adjustment_no_expected_clock(self, mock_lookup):
        """Test fallback behavior when device mapping has None for memory_clock_hz."""
        device_info = DeviceSpec(
            memory_clock_hz=None,
            tops={torch.float32: 100.0},
            dram_bw_gbs=1000.0,
            dram_gb=16.0,
            sm_count=None,
            clock_hz=1.5e9,
        )
        mock_lookup.return_value = device_info

        with patch("torch.cuda.get_device_name") as mock_get_device_name:
            mock_get_device_name.return_value = "NVIDIA H100"

            with patch.object(
                DeviceInfo, "lookup_memory_clock_hz", return_value=4e9
            ):
                dram_bw = DeviceInfo.lookup_dram_bw_gbs("NVIDIA H100")

                expected_bw = 1000.0  # No memory clock adjustment
                self.assertEqual(dram_bw, expected_bw)

    def test_dram_bw_memory_clock_adjustment_none_clock(self):
        """Test fallback behavior when memory clock lookup returns None."""
        with patch("torch.cuda.get_device_name") as mock_get_device_name:
            mock_get_device_name.return_value = "NVIDIA H100"

            with patch.object(
                DeviceInfo, "lookup_memory_clock_hz", return_value=None
            ):
                dram_bw = DeviceInfo.lookup_dram_bw_gbs("NVIDIA H100")

                expected_bw = 3350  # Datasheet value without adjustment
                self.assertEqual(dram_bw, expected_bw)


if __name__ == "__main__":
    run_tests()

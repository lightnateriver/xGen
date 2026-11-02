"""
Unit tests for sequence_parallelism/errors.py.

Covers all 6 exception classes:
- SequenceParallelError (base)
- PadNotSetError
- ProcessGroupError
- ParallelConfigError
- IncompatibleDimensionError
- BackendNotAvailableError
- AllToAllDimensionError
"""
import pytest

from x_base.sequence_parallelism.errors import (
    SequenceParallelError,
    PadNotSetError,
    ProcessGroupError,
    ParallelConfigError,
    IncompatibleDimensionError,
    BackendNotAvailableError,
    AllToAllDimensionError,
)


class TestSequenceParallelError:
    def test_is_exception(self):
        assert issubclass(SequenceParallelError, Exception)

    def test_all_errors_inherit(self):
        for cls in [PadNotSetError, ProcessGroupError, ParallelConfigError,
                    IncompatibleDimensionError, BackendNotAvailableError,
                    AllToAllDimensionError]:
            assert issubclass(cls, SequenceParallelError)


class TestPadNotSetError:
    def test_message_contains_name(self):
        err = PadNotSetError("my_pad")
        assert "my_pad" in str(err)
        assert err.name == "my_pad"

    def test_caught_by_base_class(self):
        with pytest.raises(SequenceParallelError):
            raise PadNotSetError("x")


class TestProcessGroupError:
    def test_without_group(self):
        err = ProcessGroupError("something failed")
        assert "something failed" in str(err)
        assert "[Group:" not in str(err)

    def test_with_group(self):
        err = ProcessGroupError("something failed", group="sp_group")
        assert "[Group: sp_group]" in str(err)

    def test_caught_by_base_class(self):
        with pytest.raises(SequenceParallelError):
            raise ProcessGroupError("x")


class TestParallelConfigError:
    def test_message(self):
        err = ParallelConfigError("invalid config")
        assert "invalid config" in str(err)

    def test_caught_by_base_class(self):
        with pytest.raises(SequenceParallelError):
            raise ParallelConfigError("x")


class TestIncompatibleDimensionError:
    def test_not_divisible_suggests_next_multiple(self):
        err = IncompatibleDimensionError(dim=1, size=130, divisor=4)
        msg = str(err)
        assert "130" in msg
        assert "4" in msg
        assert "132" in msg
        assert err.dim == 1
        assert err.size == 130
        assert err.divisor == 4

    def test_divisible_size_no_pad_needed(self):
        err = IncompatibleDimensionError(dim=1, size=128, divisor=4)
        msg = str(err)
        assert "128" in msg
        assert "128" in msg

    def test_custom_operation(self):
        err = IncompatibleDimensionError(dim=2, size=10, divisor=3, operation="gather")
        assert "gather" in str(err)

    def test_caught_by_base_class(self):
        with pytest.raises(SequenceParallelError):
            raise IncompatibleDimensionError(dim=0, size=1, divisor=2)


class TestBackendNotAvailableError:
    def test_without_hint(self):
        err = BackendNotAvailableError("nccl")
        assert "nccl" in str(err)
        assert "Install" not in str(err)
        assert err.backend_name == "nccl"

    def test_with_hint(self):
        err = BackendNotAvailableError("hccl", install_hint="pip install hccl")
        assert "pip install hccl" in str(err)

    def test_caught_by_base_class(self):
        with pytest.raises(SequenceParallelError):
            raise BackendNotAvailableError("x")


class TestAllToAllDimensionError:
    def test_message_contains_indices(self):
        err = AllToAllDimensionError(scatter_idx=0, gather_idx=3)
        msg = str(err)
        assert "scatter_idx=0" in msg
        assert "gather_idx=3" in msg
        assert err.scatter_idx == 0
        assert err.gather_idx == 3

    def test_caught_by_base_class(self):
        with pytest.raises(SequenceParallelError):
            raise AllToAllDimensionError(scatter_idx=0, gather_idx=0)

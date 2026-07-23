"""Tests for the calculator tool in tools.py."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from tools import create_calculator_tool, ToolLogger


@pytest.fixture
def tool_logger():
    """Create a ToolLogger instance for testing."""
    return ToolLogger(logs_dir="./test_logs")


@pytest.fixture
def calculator(tool_logger):
    """Create a calculator tool instance for testing."""
    return create_calculator_tool(tool_logger)


class TestCalculatorTool:
    """Tests for the calculator tool."""

    def test_basic_addition(self, calculator):
        """Test basic addition operation."""
        result = calculator.invoke("2 + 3")
        assert result == "5"

    def test_basic_subtraction(self, calculator):
        """Test basic subtraction operation."""
        result = calculator.invoke("10 - 4")
        assert result == "6"

    def test_basic_multiplication(self, calculator):
        """Test basic multiplication operation."""
        result = calculator.invoke("3 * 7")
        assert result == "21"

    def test_basic_division(self, calculator):
        """Test basic division operation."""
        result = calculator.invoke("10 / 2")
        assert result == "5.0"

    def test_complex_expression(self, calculator):
        """Test a more complex mathematical expression."""
        result = calculator.invoke("(2 + 3) * 4")
        assert result == "20"

    def test_floating_point_numbers(self, calculator):
        """Test expressions with floating point numbers."""
        result = calculator.invoke("1.5 + 2.5")
        assert result == "4.0"

    def test_division_by_zero(self, calculator):
        """Test that division by zero returns an error message."""
        result = calculator.invoke("10 / 0")
        assert "Error" in result or "division" in result.lower() or "zero" in result.lower()

    def test_invalid_expression_returns_error(self, calculator):
        """Test that invalid expressions return error messages."""
        result = calculator.invoke("import os")
        assert "Error" in result or "Invalid" in result or "error" in result.lower()

    def test_expression_with_letters_rejected(self, calculator):
        """Test that expressions containing letters are rejected."""
        result = calculator.invoke("abc + 1")
        assert "Error" in result or "Invalid" in result

    def test_result_is_string(self, calculator):
        """Test that the result is always a string."""
        result = calculator.invoke("2 + 3")
        assert isinstance(result, str)

    def test_large_numbers(self, calculator):
        """Test with large numbers."""
        result = calculator.invoke("1000000 + 2000000")
        assert result == "3000000"

    def test_negative_numbers(self, calculator):
        """Test expressions with negative numbers."""
        result = calculator.invoke("-5 + 3")
        assert result == "-2"

    def test_nested_parentheses(self, calculator):
        """Test nested parentheses in expressions."""
        result = calculator.invoke("((2 + 3) * (4 - 1))")
        assert result == "15"

    def test_whitespace_handling(self, calculator):
        """Test that extra whitespace is handled correctly."""
        result = calculator.invoke("  2   +   3  ")
        assert result == "5"


class TestToolLogger:
    """Tests for the ToolLogger class."""

    def test_log_tool_use(self, tool_logger):
        """Test that tool usage is logged correctly."""
        entry = tool_logger.log_tool_use(
            "calculator",
            {"expression": "2 + 3"},
            {"result": "5"}
        )
        assert entry["tool_name"] == "calculator"
        assert entry["input"]["expression"] == "2 + 3"
        assert "timestamp" in entry

    def test_get_logs(self, tool_logger):
        """Test retrieving logged entries."""
        tool_logger.log_tool_use("test_tool", {"a": 1}, {"b": 2})
        logs = tool_logger.get_logs()
        assert len(logs) == 1
        assert logs[0]["tool_name"] == "test_tool"

    def test_multiple_logs(self, tool_logger):
        """Test multiple log entries are stored."""
        tool_logger.log_tool_use("tool1", {}, {})
        tool_logger.log_tool_use("tool2", {}, {})
        tool_logger.log_tool_use("tool3", {}, {})
        logs = tool_logger.get_logs()
        assert len(logs) == 3

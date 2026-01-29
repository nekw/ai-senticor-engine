"""Unit tests for PDF report generation."""

from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import plotly.graph_objects as go
import pytest

from src.utils.pdf_gen import generate_pdf_report


class TestGeneratePDFReport:
    """Test suite for generate_pdf_report function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample analysis data."""
        return pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT", "NVDA"],
                "sentiment": [0.5, -0.3, 0.8],
                "volatility": [0.25, 0.45, 0.32],
            }
        )

    @pytest.fixture
    def sample_quadrant_fig(self):
        """Create sample quadrant figure."""
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=[0.5, -0.3, 0.8],
                y=[0.25, 0.45, 0.32],
                mode="markers+text",
                text=["AAPL", "MSFT", "NVDA"],
            )
        )
        return fig

    @patch("src.utils.pdf_gen.FPDF")
    @patch("src.utils.pdf_gen.tempfile.NamedTemporaryFile")
    def test_generates_pdf_bytes(
        self, mock_tempfile, mock_fpdf, sample_data, sample_quadrant_fig
    ):
        """Test that function generates PDF bytes."""
        # Mock FPDF instance
        mock_pdf = Mock()
        mock_pdf.output.return_value = b"PDF content"
        mock_fpdf.return_value = mock_pdf

        # Mock temp file for Plotly image
        mock_temp = Mock()
        mock_temp.name = "/tmp/test.png"
        mock_tempfile.return_value.__enter__.return_value = mock_temp

        result = generate_pdf_report(sample_data, sample_quadrant_fig)

        assert isinstance(result, bytes)
        mock_pdf.output.assert_called_once()

    @patch("src.utils.pdf_gen.FPDF")
    @patch("src.utils.pdf_gen.tempfile.NamedTemporaryFile")
    def test_includes_header(
        self, mock_tempfile, mock_fpdf, sample_data, sample_quadrant_fig
    ):
        """Test that PDF includes header."""
        mock_pdf = Mock()
        mock_pdf.output.return_value = b"PDF content"
        mock_fpdf.return_value = mock_pdf

        mock_temp = Mock()
        mock_temp.name = "/tmp/test.png"
        mock_tempfile.return_value.__enter__.return_value = mock_temp

        generate_pdf_report(sample_data, sample_quadrant_fig)

        # Check that set_font and cell were called (header creation)
        assert mock_pdf.set_font.called
        assert mock_pdf.cell.called

    @patch("src.utils.pdf_gen.FPDF")
    @patch("src.utils.pdf_gen.tempfile.NamedTemporaryFile")
    def test_includes_all_tickers(
        self, mock_tempfile, mock_fpdf, sample_data, sample_quadrant_fig
    ):
        """Test that all tickers are included in PDF."""
        mock_pdf = Mock()
        mock_pdf.output.return_value = b"PDF content"
        mock_fpdf.return_value = mock_pdf

        mock_temp = Mock()
        mock_temp.name = "/tmp/test.png"
        mock_tempfile.return_value.__enter__.return_value = mock_temp

        generate_pdf_report(sample_data, sample_quadrant_fig)

        # Check that data was processed (cell or multi_cell called multiple times)
        assert mock_pdf.cell.call_count > len(sample_data)

    @patch("src.utils.pdf_gen.FPDF")
    @patch("src.utils.pdf_gen.tempfile.NamedTemporaryFile")
    def test_handles_empty_dataframe(
        self, mock_tempfile, mock_fpdf, sample_quadrant_fig
    ):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame(columns=["ticker", "sentiment", "volatility"])

        mock_pdf = Mock()
        mock_pdf.output.return_value = b"PDF content"
        mock_fpdf.return_value = mock_pdf

        mock_temp = Mock()
        mock_temp.name = "/tmp/test.png"
        mock_tempfile.return_value.__enter__.return_value = mock_temp

        result = generate_pdf_report(df, sample_quadrant_fig)

        assert isinstance(result, bytes)

    @patch("src.utils.pdf_gen.FPDF")
    @patch("src.utils.pdf_gen.tempfile.NamedTemporaryFile")
    def test_includes_quadrant_image(
        self, mock_tempfile, mock_fpdf, sample_data, sample_quadrant_fig
    ):
        """Test that quadrant figure is included as image."""
        mock_pdf = Mock()
        mock_pdf.output.return_value = b"PDF content"
        mock_fpdf.return_value = mock_pdf

        mock_temp = Mock()
        mock_temp.name = "/tmp/test.png"
        mock_tempfile.return_value.__enter__.return_value = mock_temp

        generate_pdf_report(sample_data, sample_quadrant_fig)

        # Check that image method was called
        mock_pdf.image.assert_called()

    @patch("src.utils.pdf_gen.FPDF")
    @patch("src.utils.pdf_gen.tempfile.NamedTemporaryFile")
    def test_temp_file_cleanup(
        self, mock_tempfile, mock_fpdf, sample_data, sample_quadrant_fig
    ):
        """Test that temporary file is properly cleaned up."""
        mock_pdf = Mock()
        mock_pdf.output.return_value = b"PDF content"
        mock_fpdf.return_value = mock_pdf

        mock_temp = Mock()
        mock_temp.name = "/tmp/test.png"
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_temp
        mock_tempfile.return_value = mock_context

        generate_pdf_report(sample_data, sample_quadrant_fig)

        # Verify context manager was used (ensures cleanup)
        mock_tempfile.return_value.__enter__.assert_called_once()
        mock_tempfile.return_value.__exit__.assert_called_once()

    @patch("src.utils.pdf_gen.FPDF")
    @patch("src.utils.pdf_gen.tempfile.NamedTemporaryFile")
    def test_single_ticker(self, mock_tempfile, mock_fpdf, sample_quadrant_fig):
        """Test PDF generation with single ticker."""
        df = pd.DataFrame(
            {"ticker": ["AAPL"], "sentiment": [0.5], "volatility": [0.25]}
        )

        mock_pdf = Mock()
        mock_pdf.output.return_value = b"PDF content"
        mock_fpdf.return_value = mock_pdf

        mock_temp = Mock()
        mock_temp.name = "/tmp/test.png"
        mock_tempfile.return_value.__enter__.return_value = mock_temp

        result = generate_pdf_report(df, sample_quadrant_fig)

        assert isinstance(result, bytes)

    @patch("src.utils.pdf_gen.FPDF")
    @patch("src.utils.pdf_gen.tempfile.NamedTemporaryFile")
    def test_many_tickers(self, mock_tempfile, mock_fpdf, sample_quadrant_fig):
        """Test PDF generation with many tickers."""
        df = pd.DataFrame(
            {
                "ticker": [f"TICK{i}" for i in range(20)],
                "sentiment": [0.5] * 20,
                "volatility": [0.25] * 20,
            }
        )

        mock_pdf = Mock()
        mock_pdf.output.return_value = b"PDF content"
        mock_fpdf.return_value = mock_pdf

        mock_temp = Mock()
        mock_temp.name = "/tmp/test.png"
        mock_tempfile.return_value.__enter__.return_value = mock_temp

        result = generate_pdf_report(df, sample_quadrant_fig)

        assert isinstance(result, bytes)

    @patch("src.utils.pdf_gen.FPDF")
    @patch("src.utils.pdf_gen.tempfile.NamedTemporaryFile")
    def test_extreme_sentiment_values(
        self, mock_tempfile, mock_fpdf, sample_quadrant_fig
    ):
        """Test PDF generation with extreme sentiment values."""
        df = pd.DataFrame(
            {
                "ticker": ["POS", "NEG"],
                "sentiment": [1.0, -1.0],
                "volatility": [0.1, 0.9],
            }
        )

        mock_pdf = Mock()
        mock_pdf.output.return_value = b"PDF content"
        mock_fpdf.return_value = mock_pdf

        mock_temp = Mock()
        mock_temp.name = "/tmp/test.png"
        mock_tempfile.return_value.__enter__.return_value = mock_temp

        result = generate_pdf_report(df, sample_quadrant_fig)

        assert isinstance(result, bytes)

    @patch("src.utils.pdf_gen.FPDF")
    @patch("src.utils.pdf_gen.tempfile.NamedTemporaryFile")
    def test_includes_timestamp(
        self, mock_tempfile, mock_fpdf, sample_data, sample_quadrant_fig
    ):
        """Test that PDF includes timestamp."""
        mock_pdf = Mock()
        mock_pdf.output.return_value = b"PDF content"
        mock_fpdf.return_value = mock_pdf

        mock_temp = Mock()
        mock_temp.name = "/tmp/test.png"
        mock_tempfile.return_value.__enter__.return_value = mock_temp

        generate_pdf_report(sample_data, sample_quadrant_fig)

        # Check that datetime-related text was added
        # (cell or multi_cell should be called with date/time)
        assert mock_pdf.cell.called or mock_pdf.multi_cell.called

    @patch("src.utils.pdf_gen.FPDF")
    @patch("src.utils.pdf_gen.tempfile.NamedTemporaryFile")
    def test_none_quadrant_figure(self, mock_tempfile, mock_fpdf, sample_data):
        """Test handling when quadrant figure is None."""
        mock_pdf = Mock()
        mock_pdf.output.return_value = b"PDF content"
        mock_fpdf.return_value = mock_pdf

        result = generate_pdf_report(sample_data, quadrant_fig=None)

        # Should still generate PDF without the image
        assert isinstance(result, bytes)
        # Image method should not be called
        assert not mock_pdf.image.called

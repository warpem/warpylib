"""Warp Matrix2 uses column-major XML, including nonsymmetric matrices."""
import torch
from lxml import etree
from warpylib import TiltSeries
from warpylib.movie import Movie


def test_tilt_series_reads_and_writes_warp_matrix(tmp_path):
    path = tmp_path / "tilt.xml"
    path.write_text('<TiltSeries MagnificationCorrection="1.01, -0.003, 0.007, 0.99"/>')
    ts = TiltSeries()
    ts.load_meta(str(path))
    torch.testing.assert_close(ts.magnification_correction,
                               torch.tensor([[1.01, 0.007], [-0.003, 0.99]]))
    ts.save_meta(str(path))
    values = [float(v) for v in etree.parse(str(path)).getroot().get("MagnificationCorrection").split(",")]
    torch.testing.assert_close(torch.tensor(values), torch.tensor([1.01, -0.003, 0.007, 0.99]))


def test_movie_reads_and_writes_warp_matrix(tmp_path):
    path = tmp_path / "movie.xml"
    path.write_text('<Movie MagnificationCorrection="1.01, -0.003, 0.007, 0.99"/>')
    movie = Movie()
    movie.load_meta(str(path))
    torch.testing.assert_close(movie.magnification_correction,
                               torch.tensor([[1.01, 0.007], [-0.003, 0.99]]))
    movie.save_meta(str(path))
    values = [float(v) for v in etree.parse(str(path)).getroot().get("MagnificationCorrection").split(",")]
    torch.testing.assert_close(torch.tensor(values), torch.tensor([1.01, -0.003, 0.007, 0.99]))


def test_tilt_metadata_strict_load_rejects_partial_defaults(tmp_path):
    import pytest
    from warpylib.tilt_series.io import load_meta
    path = tmp_path / "bad.xml"
    path.write_text('<TiltSeries><Angles>0\n30</Angles><CTF><Param Name="Voltage" Value="bad"/></CTF></TiltSeries>')
    with pytest.raises(ValueError, match="Failed to load tilt metadata"):
        load_meta(TiltSeries(), str(path), strict=True)
    with pytest.raises(FileNotFoundError):
        load_meta(TiltSeries(), str(tmp_path / "missing.xml"), strict=True)

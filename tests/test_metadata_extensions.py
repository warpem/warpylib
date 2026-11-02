"""Beam calibration and unknown namespaced metadata survive Warp geometry edits."""
import json

import pytest
import torch
from lxml import etree
from warpylib import TiltSeries

NS = "urn:beam:refinement:1"


def test_beam_calibration_roundtrip_and_updates_preserve_unknown_nodes(tmp_path):
    path = tmp_path / "tilt.xml"
    ts = TiltSeries(n_tilts=2)
    ts.angles = torch.tensor([-20., 20.])
    ts.save_meta(str(path))
    tree = etree.parse(str(path))
    root = tree.getroot()
    beam = etree.SubElement(root, f"{{{NS}}}Refinement", nsmap={"beam": NS},
        schemaVersion="1", pixelSizeAngstrom="2.0", fieldOfViewAngstrom="256.0",
        normalization="none")
    calibration = {"half_gains": [[1.0, 0.7], [1.0, 0.8]], "tilt_indices": [0, 1]}
    etree.SubElement(beam, f"{{{NS}}}Calibration").text = json.dumps(calibration)
    etree.SubElement(beam, f"{{{NS}}}Future", mode="keep").text = "opaque content"
    etree.SubElement(root, "{urn:other:metadata}Record", keep="yes").text = "other extension"
    tree.write(str(path))

    loaded = TiltSeries()
    loaded.load_meta(str(path), strict=True)
    assert loaded.beam_refinement["attributes"]["pixelSizeAngstrom"] == "2.0"
    assert loaded.beam_refinement["calibration"] == calibration
    assert len(loaded.metadata_extensions) == 2
    dose_weights = loaded.grid_dose_weights.values.clone()
    loaded.angles += 1
    loaded.save_meta(str(path))
    again = TiltSeries()
    again.load_meta(str(path), strict=True)
    assert again.beam_refinement == loaded.beam_refinement
    torch.testing.assert_close(again.grid_dose_weights.values, dose_weights)
    root = etree.parse(str(path)).getroot()
    assert root.find(f"{{{NS}}}Refinement/{{{NS}}}Future").text == "opaque content"
    assert root.find("{urn:other:metadata}Record").get("keep") == "yes"

    again.beam_refinement["attributes"]["iteration"] = "2"
    again.beam_refinement["calibration"]["half_gains"][1][1] = 0.9
    again.save_meta(str(path))
    final = TiltSeries()
    final.load_meta(str(path), strict=True)
    assert final.beam_refinement["attributes"]["iteration"] == "2"
    assert final.beam_refinement["calibration"]["half_gains"][1][1] == 0.9
    assert len(etree.parse(str(path)).getroot().findall(f"{{{NS}}}Refinement")) == 1


def test_beam_extension_can_be_created_and_resets_on_unrelated_load(tmp_path):
    path = tmp_path / "tilt.xml"
    ts = TiltSeries()
    ts.beam_refinement = {"attributes": {"schemaVersion": "1"}, "calibration": {"tilt_indices": [0]}}
    ts.save_meta(str(path))
    ts.load_meta(str(path), strict=True)
    assert ts.beam_refinement["calibration"]["tilt_indices"] == [0]
    empty = tmp_path / "empty.xml"
    TiltSeries().save_meta(str(empty))
    ts.load_meta(str(empty), strict=True)
    assert ts.beam_refinement is None
    assert ts.metadata_extensions == []


def test_invalid_beam_calibration_fails_strict_metadata_loading(tmp_path):
    path = tmp_path / "bad.xml"
    path.write_text(f'<TiltSeries xmlns:beam="{NS}"><beam:Refinement>'
                    '<beam:Calibration>[]</beam:Calibration></beam:Refinement></TiltSeries>')
    with pytest.raises(ValueError, match="Calibration JSON must be an object"):
        TiltSeries().load_meta(str(path), strict=True)

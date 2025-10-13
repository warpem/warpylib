"""Tests for TiltSeries XML serialization."""

import pytest
from pathlib import Path
import xml.etree.ElementTree as ET
from warpylib import TiltSeries


def normalize_float(value_str: str) -> float | None:
    """Try to convert a string to float, return None if not possible."""
    try:
        return float(value_str)
    except (ValueError, TypeError):
        return None


def compare_xml_nodes(node1: ET.Element, node2: ET.Element, path: str = "") -> list[str]:
    """
    Recursively compare two XML nodes, ignoring floating-point precision differences.
    Returns a list of differences found.
    """
    differences = []
    current_path = f"{path}/{node1.tag}" if path else node1.tag

    # Compare tag names
    if node1.tag != node2.tag:
        differences.append(f"{current_path}: Tag mismatch: {node1.tag} != {node2.tag}")
        return differences

    # Compare attributes (keys)
    attrs1 = set(node1.attrib.keys())
    attrs2 = set(node2.attrib.keys())

    missing_in_node2 = attrs1 - attrs2
    missing_in_node1 = attrs2 - attrs1

    if missing_in_node2:
        differences.append(f"{current_path}: Attributes in original but not in copy: {missing_in_node2}")
    if missing_in_node1:
        differences.append(f"{current_path}: Attributes in copy but not in original: {missing_in_node1}")

    # Compare attribute values (ignoring float precision)
    for attr in attrs1 & attrs2:
        val1 = node1.attrib[attr]
        val2 = node2.attrib[attr]

        # Try to parse as floats
        float1 = normalize_float(val1)
        float2 = normalize_float(val2)

        if float1 is not None and float2 is not None:
            # Both are floats - we don't check exact equality
            continue

        # Try to parse as comma-separated floats (like PlaneNormal or MagnificationCorrection)
        # Strip parentheses first
        val1_clean = val1.strip("()").replace(" ", "")
        val2_clean = val2.strip("()").replace(" ", "")

        parts1 = val1_clean.split(",")
        parts2 = val2_clean.split(",")

        if len(parts1) == len(parts2) and len(parts1) > 1:
            # Try to parse all parts as floats
            floats1 = [normalize_float(p) for p in parts1]
            floats2 = [normalize_float(p) for p in parts2]

            if all(f is not None for f in floats1) and all(f is not None for f in floats2):
                # All are floats - we don't check exact equality
                continue

        # Try to parse as semicolon-separated floats (like Zernike coefficients)
        parts1 = val1.split(";")
        parts2 = val2.split(";")

        if len(parts1) == len(parts2) and len(parts1) > 1:
            # Try to parse all parts as floats
            floats1 = [normalize_float(p.strip()) for p in parts1]
            floats2 = [normalize_float(p.strip()) for p in parts2]

            if all(f is not None for f in floats1) and all(f is not None for f in floats2):
                # All are floats - we don't check exact equality
                continue

        if val1 != val2:
            # Not floats or different strings
            differences.append(f"{current_path}@{attr}: Value mismatch: '{val1}' != '{val2}'")

    # Compare text content (ignoring float precision)
    text1 = (node1.text or "").strip()
    text2 = (node2.text or "").strip()

    if text1 or text2:
        # Try to parse as floats (possibly multi-line)
        lines1 = text1.split('\n')
        lines2 = text2.split('\n')

        if len(lines1) != len(lines2):
            differences.append(f"{current_path}: Text line count mismatch: {len(lines1)} != {len(lines2)}")
        else:
            # Compare line by line
            for i, (line1, line2) in enumerate(zip(lines1, lines2)):
                line1 = line1.strip()
                line2 = line2.strip()

                float1 = normalize_float(line1)
                float2 = normalize_float(line2)

                if float1 is not None and float2 is not None:
                    # Both are floats - we don't check exact equality
                    continue
                elif line1 != line2:
                    differences.append(f"{current_path}[text line {i}]: '{line1}' != '{line2}'")

    # Compare tail (text after the element)
    tail1 = (node1.tail or "").strip()
    tail2 = (node2.tail or "").strip()

    if tail1 != tail2:
        # Only report if non-empty
        if tail1 or tail2:
            differences.append(f"{current_path}[tail]: '{tail1}' != '{tail2}'")

    # Compare children (match by tag name, not order)
    children1 = list(node1)
    children2 = list(node2)

    # Group children by tag name
    from collections import defaultdict
    children1_by_tag = defaultdict(list)
    children2_by_tag = defaultdict(list)

    for child in children1:
        children1_by_tag[child.tag].append(child)

    for child in children2:
        children2_by_tag[child.tag].append(child)

    # Find tags only in one file or the other
    tags1 = set(children1_by_tag.keys())
    tags2 = set(children2_by_tag.keys())

    only_in_original = tags1 - tags2
    only_in_copy = tags2 - tags1

    if only_in_original:
        tags_list = [f"{tag} ({len(children1_by_tag[tag])}x)" for tag in sorted(only_in_original)]
        differences.append(f"{current_path}: Child tags only in original: {', '.join(tags_list)}")

    if only_in_copy:
        tags_list = [f"{tag} ({len(children2_by_tag[tag])}x)" for tag in sorted(only_in_copy)]
        differences.append(f"{current_path}: Child tags only in copy: {', '.join(tags_list)}")

    # Compare matching tags
    for tag in tags1 & tags2:
        list1 = children1_by_tag[tag]
        list2 = children2_by_tag[tag]

        # Special handling for Param elements - match by Name attribute
        if tag == "Param":
            # Create dictionaries mapping Name -> element
            params1_by_name = {elem.get("Name"): elem for elem in list1}
            params2_by_name = {elem.get("Name"): elem for elem in list2}

            names1 = set(params1_by_name.keys())
            names2 = set(params2_by_name.keys())

            missing_in_copy = names1 - names2
            missing_in_original = names2 - names1

            if missing_in_copy:
                differences.append(f"{current_path}: Param Names only in original: {', '.join(sorted(missing_in_copy))}")
            if missing_in_original:
                differences.append(f"{current_path}: Param Names only in copy: {', '.join(sorted(missing_in_original))}")

            # Compare matching params
            for name in names1 & names2:
                differences.extend(compare_xml_nodes(params1_by_name[name], params2_by_name[name], current_path))

        elif len(list1) != len(list2):
            differences.append(f"{current_path}: Tag '{tag}' count mismatch: {len(list1)} != {len(list2)}")
            # Compare what we can
            min_len = min(len(list1), len(list2))
            for child1, child2 in zip(list1[:min_len], list2[:min_len]):
                differences.extend(compare_xml_nodes(child1, child2, current_path))
        else:
            # Same count - compare them
            for child1, child2 in zip(list1, list2):
                differences.extend(compare_xml_nodes(child1, child2, current_path))

    return differences


def test_tiltseries_xml_roundtrip():
    """Test that TiltSeries can read and write XML without losing structure."""
    # Paths
    original_path = Path(__file__).parent.parent / 'testdata' / 'TS_1.xml'
    output_dir = Path(__file__).parent.parent / 'testoutputs'
    output_dir.mkdir(exist_ok=True)
    copy_path = output_dir / 'TS_1_copy.xml'

    # Read the original XML using TiltSeries
    ts = TiltSeries()
    ts.load_meta(str(original_path))

    # Write the copy
    ts.save_meta(str(copy_path))

    # Parse both XML files
    tree_original = ET.parse(original_path)
    tree_copy = ET.parse(copy_path)

    root_original = tree_original.getroot()
    root_copy = tree_copy.getroot()

    # Compare the XML structures
    differences = compare_xml_nodes(root_original, root_copy)

    # Filter out expected differences (nodes we're not implementing)
    ignored_tags = {'OptionsCTF', 'PS1D', 'SimulatedScale', 'TiltPS1D', 'TiltSimulatedScale'}
    filtered_differences = []
    for diff in differences:
        # Check if this difference is about an ignored tag
        is_ignored = False
        for tag in ignored_tags:
            if f"Child tags only in original: {tag}" in diff or \
               f"{tag} (" in diff and "only in original" in diff:
                is_ignored = True
                break

        if not is_ignored:
            filtered_differences.append(diff)

    # Report differences (excluding ignored ones)
    if filtered_differences:
        error_msg = f"XML structures differ:\n" + "\n".join(filtered_differences)
        pytest.fail(error_msg)

    print(f"\n✓ XML roundtrip successful: {original_path} -> {copy_path}")
    print(f"  Root tag: {root_original.tag}")
    print(f"  Root attributes: {len(root_original.attrib)}")
    print(f"  Children count: {len(list(root_original))}")

    # Report ignored differences
    ignored_count = len(differences) - len(filtered_differences)
    if ignored_count > 0:
        print(f"  Note: {ignored_count} expected differences ignored (CTF fitting nodes not implemented)")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

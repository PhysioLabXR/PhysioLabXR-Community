from physiolab.contracts import Pipeline

EXPECTED = [
    ("device", "simulated-emg"),
    ("processing", "bandpass"),
    ("processing", "sliding-window"),
    ("model", "demo-emg-rms"),
    ("application", "web-preview"),
]


def validate_pipeline(pipeline: Pipeline):
    """Reject unsupported graphs before executing anything. V1 runtime is one typed chain."""
    nodes = pipeline.nodes
    if len({n.id for n in nodes}) != len(nodes):
        raise ValueError("Node IDs must be unique")
    if [(n.kind, n.implementation) for n in nodes] != EXPECTED:
        raise ValueError("This runtime supports the five-node EMG demo chain only")
    expected_edges = {(a.id, b.id) for a, b in zip(nodes, nodes[1:])}
    actual_edges = [(e.source, e.target) for e in pipeline.edges]
    if len(actual_edges) != len(expected_edges) or set(actual_edges) != expected_edges:
        raise ValueError("Edges must connect the demo chain in order without cycles or duplicates")
    expected_keys = [set(), {"low_hz", "high_hz"}, {"size", "stride"}, {"threshold"}, set()]
    for node, keys in zip(nodes, expected_keys):
        if set(node.parameters) != keys:
            raise ValueError(f"Unexpected parameters for {node.implementation}")
        if any(
            isinstance(v, bool) or not isinstance(v, (float, int)) for v in node.parameters.values()
        ):
            raise ValueError("Runtime parameters must be numeric")
    low, high = nodes[1].parameters["low_hz"], nodes[1].parameters["high_hz"]
    if not 0 < low < high < 50:
        raise ValueError("Bandpass requires 0 < low < high < 50 Hz")
    size, stride = nodes[2].parameters["size"], nodes[2].parameters["stride"]
    if size != int(size) or stride != int(stride) or not 1 <= stride <= size <= 1000:
        raise ValueError("Window size/stride must be integers with 1 <= stride <= size <= 1000")
    if not 0 < nodes[3].parameters["threshold"] <= 2:
        raise ValueError("RMS threshold must be between 0 and 2")

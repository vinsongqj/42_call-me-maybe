from typing import Any, Dict, List, Tuple


def parse_function_schemas(
    functions_data: List[Dict[str, Any]]
) -> Dict[str, List[Tuple[str, str]]]:
    """Convert a list of raw function definitions into a compact schema lookup.

    Transforms the verbose JSON function definition format into a lightweight
    mapping used by ``TrieJSONRulebook`` to enforce parameter ordering and
    typing during constrained decoding.

    Args:
        functions_data: A list of function definition dicts, each containing
            at minimum a ``"name"`` key and a ``"parameters"`` dict whose
            values have a ``"type"`` field (e.g. ``"string"``, ``"number"``,
            ``"boolean"``).

    Returns:
        A dict mapping each function name to an ordered list of
        ``(parameter_name, parameter_type)`` tuples reflecting the order in
        which parameters appear in the definition.

    Example::

        >>> data = [{"name": "fn_add", "parameters": {"a": {"type": "number"},
        ...
          "b": {"type": "number"}}}]
        >>> parse_function_schemas(data)
        {'fn_add': [('a', 'number'), ('b', 'number')]}
    """
    metadata: Dict[str, List[Tuple[str, str]]] = {}
    for fn in functions_data:
        name: str = fn.get("name", "")
        params: Dict[str, Any] = fn.get("parameters", {})

        param_list: List[Tuple[str, str]] = []
        for p_name, p_info in params.items():
            p_type: str = p_info.get("type", "string")
            param_list.append((p_name, p_type))

        metadata[name] = param_list
    return metadata

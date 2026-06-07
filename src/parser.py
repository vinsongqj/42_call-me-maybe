from typing import List, Dict, Any, Tuple


def parse_function_schemas(
    functions_data: List[Dict[str, Any]]
) -> Dict[str, List[Tuple[str, str]]]:
    metadata = {}
    for fn in functions_data:
        name = fn.get("name", "")
        params = fn.get("parameters", {})

        param_list = []
        for p_name, p_info in params.items():
            p_type = p_info.get("type", "string")
            param_list.append((p_name, p_type))

        metadata[name] = param_list
    return metadata

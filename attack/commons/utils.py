import json
import logging


PORT = None

# Relative to the directory running the code (should be attack/)
logger = logging.getLogger("CustomLogger")


def parse_json_str(json_str, prefix="", is_notify=True):
    try:
        if prefix:
            str = prefix + json_str
        else:
            str = json_str

        if "{" in str and "}" in str:
            start = str.index("{")
            end = str.rindex("}")
            str = str[start : end + 1]

        return json.loads(str)

    except json.JSONDecodeError as e:
        if is_notify:
            logger.error(f"Invalid JSON format - {e}: {str}")
        return None

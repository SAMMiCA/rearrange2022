from typing import Optional
import sys


def get_raw_args():
    raw_args = sys.argv[1:]
    filtered_args = []
    remove: Optional[str] = None
    enclose_in_quotes: Optional[str] = None
    for arg in raw_args:
        if remove is not None:
            remove = None
        elif enclose_in_quotes is not None:
            # Within backslash expansion: close former single, open double, create single, close double, reopen single
            inner_quote = r"\'\"\'\"\'"
            # Convert double quotes into backslash double for later expansion
            filtered_args.append(
                inner_quote + arg.replace('"', r"\"").replace("'", r"\"") + inner_quote
            )
            enclose_in_quotes = None
        elif arg in [
            "--runs_on",
            "--ssh_cmd",
            "--conda_env",
            "--rearrange2022_path",
            "--extra_tag",
            "--machine_id",
        ]:
            remove = arg
        elif arg == "--config_kwargs":
            enclose_in_quotes = arg
            filtered_args.append(arg)
        else:
            filtered_args.append(arg)
    return filtered_args


def wrap_single(text):
    return f"'{text}'"


def wrap_single_nested(text):
    # Close former single, start backslash expansion (via $), create new single quote for expansion:
    quote_enter = r"'$'\'"
    # New closing single quote for expansion, close backslash expansion, reopen former single:
    quote_leave = r"\'''"
    return f"{quote_enter}{text}{quote_leave}"


def wrap_double(text):
    return f'"{text}"'


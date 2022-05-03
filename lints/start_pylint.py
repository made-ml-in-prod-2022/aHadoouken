import os
from pylint.lint import Run

print(os.getcwd())

disable_messages = [
    "missing-module-docstring",
    "missing-class-docstring",
    "missing-function-docstring",
    "no-value-for-parameter",
    "invalid-name",
    "logging-fstring-interpolation",
    "unspecified-encoding",
    "unused-argument"
]
disable_flag = ",".join(disable_messages)
pylint_opts = ["ml_app", f"--disable={disable_flag}"]
Run(pylint_opts)


# Stubs for pandas.io.date_converters (Python 3.6)
#
# NOTE: This dynamically typed stub was automatically generated by stubgen.

from typing import Any

def parse_date_time(date_col: Any, time_col: Any): ...
def parse_date_fields(year_col: Any, month_col: Any, day_col: Any): ...
def parse_all_fields(year_col: Any, month_col: Any, day_col: Any, hour_col: Any, minute_col: Any, second_col: Any): ...
def generic_parser(parse_func: Any, *cols: Any): ...

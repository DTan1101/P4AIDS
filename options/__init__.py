"""Tham số dòng lệnh cho train và eval."""

from options.eval_option import EvalOptions, parse_eval_options
from options.train_options import TrainOptions, parse_train_options

__all__ = ["EvalOptions", "TrainOptions", "parse_eval_options", "parse_train_options"]

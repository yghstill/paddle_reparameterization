import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from ppcls.utils import config
from ppcls.engine.engine import Engine

if __name__ == "__main__":
    args = config.parse_args()
    config = config.get_config(
        args.config, overrides=args.override, show=False)
    config.profiler_options = args.profiler_options
    engine = Engine(config, mode="train")
    engine.train()

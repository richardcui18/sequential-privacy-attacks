import argparse
import logging

logger = logging.getLogger(__name__)

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize_parser()

    def initialize_parser(self):
        self.parser.add_argument('--total_time', type=int, default=10, help="Total time length considered in simulation when generating true intent.")
        self.parser.add_argument('--lambda_value', type=float, default=0.5, help="Maximum confidence threshold determined by buyer.")
        self.parser.add_argument("--verbose", type=str, default="False")
        self.parser.add_argument('--num_iter', type=int, default=10, help="Number of iterations in Bi-Directional HMM-RL algorithm.")
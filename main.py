import argparse
import datetime
import json
import os
import shutil
import multiprocessing

import problem as pro

# GLOBAL VARIABLES
TIME = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
PATH = os.path.dirname(os.path.realpath(__file__))
CNFG = os.path.join(os.path.join(PATH, "data"), "config.json")


def get_args():
    """ Process args from command line """
    parser = argparse.ArgumentParser()

    # Set type of run
    parser.add_argument(
        "-t", "--type",
    )

    # Set problem to run
    parser.add_argument(
        "-e", "--environment",
    )

    # Set if problem size tuning
    parser.add_argument(
        "-s", "--size", default=None,
    )

    # Set param to tune
    parser.add_argument(
        "-p", "--param",
    )

    # Set algorithm to run
    parser.add_argument(
        "-a", "--algorithm",
    )

    # Set name of the experiment
    parser.add_argument(
        "-n", "--name", default=f'RUN_DATA_{TIME}',
    )
    return parser.parse_args()


def main():
    """ Main """
    # Process command line
    args = get_args()

    # Process directories and config
    dir_name = os.path.join(os.path.join(PATH, "out"), args.name)
    os.makedirs(dir_name)
    shutil.copy(CNFG, dir_name)

    # Open config
    with open(CNFG, "r") as open_file:
        jsonconfig = json.load(open_file)
    config = jsonconfig[args.environment]

    # Create output directory
    out_dir = os.path.join(dir_name, args.name)
    os.makedirs(out_dir)

    # Core count for problem tuning
    core_count = round(multiprocessing.cpu_count() * .75)

    # Initialize the experiment
    exp = pro.ExperimentClass(args.name, args.type, args.param,
                              args.environment, args.algorithm,
                              args.size, config, core_count, out_dir)

    # Run the experiment
    exp.run()


if __name__ == "__main__":
    main()

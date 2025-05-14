import argparse

def args_parser():
    parser = argparse.ArgumentParser(description='SegHDC Framework')
    parser.add_argument('--dim', default=10000, type=int)
    parser.add_argument('--numClass', default=2, type=int)
    parser.add_argument('--iterations', default=10, type=int)
    parser.add_argument('--path', default="./test/", type=str)
    parser.add_argument('--changed_ratio', default=0.2, type=float)
    parser.add_argument('--skip', default=26, type=int)

    args, unknown = parser.parse_known_args()
    return args




"""
Reference: https://github.com/raulmur/evaluate_ate_scale/blob/master/associate.py

Description:
The Kinect provides the color and depth images in an un-synchronized way.
This means that the set of time stamps from the color images do not intersect with those of the depth images.
Therefore, we need some way of associating color images to depth images.

It reads the time stamps from the rgb.txt file and the depth.txt file, and joins them by finding the best matches.
"""

import argparse

def read_file_list(filename):
    """
    Reads a trajectory from a text file.

    File format:
    The file format is "stamp d1 d2 d3 ...", where stamp denotes the time stamp (to be matched)
    and "d1 d2 d3.." is arbitary data (e.g., a 3D position and 3D orientation) associated to this timestamp.

    Input:
    filename -- File name

    Output:
    dict -- dictionary of (stamp, data) tuples
    """
    file = open(filename)
    data = file.read()
    lines = data.replace(",", " ").replace("\t", " ").split("\n")
    list = [
        [v.strip() for v in line.split(" ") if v.strip() != ""]
        for line in lines
        if len(line) > 0 and line[0] != "#"
    ]
    list = [(float(l[0]), l[1:]) for l in list if len(l) > 1]
    return dict(list)


def associate(first_list, second_list, offset, max_difference) -> list:
    """
    Associate two dictionaries of (stamp,data). As the time stamps never match exactly, we aim
    to find the closest match for every input tuple.

    Input:
    first_list -- first dictionary of (stamp,data) tuples
    second_list -- second dictionary of (stamp,data) tuples
    offset -- time offset between both dictionaries (e.g., to model the delay between the sensors)
    max_difference -- search radius for candidate generation

    Output:
    matches -- list of matched tuples ((stamp1,data1),(stamp2,data2))

    """
    # 1. Get timestamps
    first_keys = list(first_list.keys())  # Convert to list
    second_keys = list(second_list.keys())  # Convert to list

    # 2. Generate potential matches
    potential_matches = [
        (abs(a - (b + offset)), a, b)  # (difference, a, b)
        for a in first_keys
        for b in second_keys
        if abs(a - (b + offset)) < max_difference
    ]

    # 3. Sort by time difference with closest matches first
    potential_matches.sort()

    # 4. Select best unique matches
    matches = []
    for _, a, b in potential_matches:
        if (a in first_keys and b in second_keys):  # Check if timestamps are still in the list
            first_keys.remove(a)  # Removed used timestamps
            second_keys.remove(b)
            matches.append((a, b))

    # 5. Return sorted matches
    matches.sort()

    return matches


if __name__ == "__main__":
    # parse command line
    parser = argparse.ArgumentParser(
        description="""
    This script takes two data files with timestamps and associates them
    """
    )
    parser.add_argument("first_file", help="first text file (format: timestamp data)")
    parser.add_argument("second_file", help="second text file (format: timestamp data)")
    parser.add_argument(
        "--first_only",
        help="only output associated lines from first file",
        action="store_true",
    )
    parser.add_argument(
        "--offset",
        help="time offset added to the timestamps of the second file (default: 0.0)",
        default=0.0,
    )
    parser.add_argument(
        "--max_difference",
        help="maximally allowed time difference for matching entries (default: 0.02)",
        default=0.02,
    )
    args = parser.parse_args()

    first_list = read_file_list(args.first_file)
    second_list = read_file_list(args.second_file)

    matches = associate(
        first_list, second_list, float(args.offset), float(args.max_difference)
    )

    if args.first_only:
        for a, b in matches:
            print("%f %s" % (a, " ".join(first_list[a])))
    else:
        for a, b in matches:
            print(
                "%f %s %f %s"
                % (
                    a,
                    " ".join(first_list[a]),
                    b - float(args.offset),
                    " ".join(second_list[b]),
                )
            )

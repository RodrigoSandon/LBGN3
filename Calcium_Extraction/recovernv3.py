"""
This module contains an example that demonstrates how to recover nvista 3
and nvoke 2 movies that cannot be read because they are missing their JSON
footer of metadata.
"""

import os
import argparse
import datetime
import textwrap
import re

import tqdm
import numpy as np
import isx


def get_timing_and_spacing(input_file, period, start_timestamp):
    # 20yy-mm-dd-hh-mm-ss
    date_pattern = re.compile("20\d{2}(-\d{2}){5}")

    num_pixels = (800, 1280)
    spacing = isx.core.Spacing(num_pixels=num_pixels)
    pixel_size = isx._internal.IsxRatio(3, 1)
    spacing._impl.pixel_width = pixel_size
    spacing._impl.pixel_height = pixel_size

    total_num_pixels = np.prod(num_pixels)
    frame_size_bytes = 2 * total_num_pixels
    header_size_bytes = 2 * 2 * num_pixels[1]
    footer_size_bytes = header_size_bytes
    frame_size_bytes_with_hf = frame_size_bytes + header_size_bytes + footer_size_bytes
    num_frames = os.stat(input_file).st_size // frame_size_bytes_with_hf
    input_file_base = os.path.basename(input_file)

    if not start_timestamp:
        parsed_date = date_pattern.search(input_file_base)[0]
        start_dt = datetime.datetime.strptime(
            parsed_date + "-+0000", "%Y-%m-%d-%H-%M-%S-%z"
        )
        start_timestamp = start_dt.timestamp()

    start_time = isx.core.Time._from_secs_since_epoch(
        isx.core.Duration.from_secs(start_timestamp)
    )
    print("Start time is : ", start_time)
    timing = isx.core.Timing(num_samples=num_frames, period=period, start=start_time)

    return timing, spacing, frame_size_bytes, header_size_bytes, footer_size_bytes


def recover_nv3_movie(input_file, output_file, period, force, start_timestamp):
    """Recovers frames from a corrupt nVista 3 movie file."""

    read_failed = False
    try:
        isx.Movie.read(input_file)
    except Exception as e:
        read_failed = True

    if not read_failed and not force:
        raise Exception(
            "Movie file has valid footer. Use --force to run recovery script anyway."
        )

    (
        timing,
        spacing,
        frame_size_bytes,
        header_size_bytes,
        footer_size_bytes,
    ) = get_timing_and_spacing(input_file, period, start_timestamp)
    output_movie = isx.io.Movie.write(output_file, timing, spacing, np.uint16)
    with tqdm.tqdm(total=timing.num_samples) as pbar:
        with open(input_file, "rb") as f:
            for i in range(timing.num_samples):
                f.seek(header_size_bytes, 1)
                frame_bytes = f.read(frame_size_bytes)
                frame = np.frombuffer(frame_bytes, dtype=np.uint16).reshape(
                    spacing.num_pixels
                )
                output_movie.set_frame_data(i, frame)
                f.seek(footer_size_bytes, 1)
                pbar.update(1)
    output_movie.flush()


def main():
    """Reads a corrupt nVista 3 movie file and writes a new recovered file."""
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(
            """
            Recovers an nVista 3 or nVoke 2 movie without metadata by writing a new file
            with the same frame data but new metadata.

            Description
            -----------

            The start date and time is inferred from the file name to the nearest second,
            or can be passed as an argument.

            The number of frames in the movie is inferred from the file size assuming that
            each frame has 1280x800 pixels of type uint16.

            The frame period can be provided in using the --period_msecs argument and is
            50 milliseconds (corresponding to 20 Hz) by default.abs

            All other metadata will take on default values, which in many cases, will be empty.

            Example
            -------

            For example, to recover a file with a desired frame period of 100 milliseconds,
            run the following from the command line.

            python recovernv3.py --period_msecs 100 2018-07-19-13-36-04_video.isxd 2018-07-19-13-36-04_video-recovered.isxd

            Arguments
            ---------
            """
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="The path of the input .isxd file. E.g. data_dir/2018-07-19-13-36-04_video.isxd",
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="The path of the output .isxd file. E.g. data_dir/2018-07-19-13-36-04_video-recovered.isxd",
    )
    parser.add_argument(
        "-p",
        "--period_msecs",
        required=False,
        default=50,
        type=int,
        help="The frame period in milliseconds.",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Set flag to force recovery script even if file has valid footer.",
    )
    parser.add_argument(
        "-t",
        "--timestamp_secs",
        required=False,
        type=int,
        help="The timestamp of the start of the movie in seconds. E.g. 1580120183 - If argument not passed, start time is inferred from filename. ",
    )

    args = parser.parse_args()

    recover_nv3_movie(
        args.input_file,
        args.output_file,
        period=isx.core.Duration.from_msecs(args.period_msecs),
        force=args.force,
        start_timestamp=args.timestamp_secs,
    )


if __name__ == "__main__":
    main()

import os
import isx
from pathlib import Path  # to work with dir
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import label, regionprops

def show_img(img_matrix, footprints=None, title=None,
             percentile=99, plot_individual=False, color='red'):
    """
    function shows an image and (optionally) overlays contour/footprint map

    Parameters:
        img_matrix: a background image
        footprints: a 2D contour map or 3D cell footprints
        title: (str) image title
        percentile: used to set colormap vmax value to control the image brightness
        plot_individual: (boolean)
            plot one footprint at a time, slower compared to contour plot

    Returns:
        None

    """

    plt.figure(figsize=(8, 5))
    plt.imshow(img_matrix,
               interpolation='nearest',
               cmap="gray",
               vmax=np.percentile(img_matrix, percentile))

    if footprints is not None and footprints.size != 0:
        if footprints.ndim >2 and plot_individual is True:
            for cell in range(footprints.shape[2]):
                plt.contour(footprints[:, :, cell],
                            colors=color,
                            linewidths=0.2,
                            alpha=0.4)
        elif plot_individual is False:
            if footprints.ndim >2:
                footprints = np.ndarray.max(footprints, axis=2)
            plt.contour(footprints,
                colors=color,
                linewidths=0.2,
                alpha=0.4)

    if title is not None:
        if footprints is not None and footprints.size == 0:
            plt.title("no footprints detected")
        else:
            plt.title(title)

    plt.xticks([])
    plt.yticks([])
    plt.show()


def project_isxd(input_isxd, input_low_cutoff=0.02, input_high_cutoff=0.3):
    """
    function applies spatial filtering and mean projection to input isxd recording
    Parameters:
        input_isxd (isxd): short red channel recording
        input_low_cutoff (float, optional): spatial filtering setting. Defaults to 0.02.
        input_high_cutoff (float, optional):spatial filtering setting. Defaults to 0.3.

    Returns:
        None
    """

    # make new output file name
    bp_file = str(Path(os.getcwd(), input_isxd.stem + '-bp.isxd'))
    load_movie = isx.Movie.read(str(input_isxd))
    total_frames = load_movie.timing.num_samples
    trim_isxd = str(Path(os.getcwd(), input_isxd.stem + '-trim.isxd'))

    if os.path.exists(trim_isxd):
        os.remove(trim_isxd)
    if total_frames>100:
        isx.trim_movie(str(input_isxd), trim_isxd, crop_segments=[100,total_frames])
    else:
        trim_isxd = input_isxd

    if not os.path.exists(bp_file):
        isx.spatial_filter(
            input_movie_files=str(trim_isxd),
            output_movie_files=bp_file,
            low_cutoff=input_low_cutoff,
            high_cutoff=input_high_cutoff,
            retain_mean=False,
            # leave subtract_global_minimum setting as true
            # for correct dff display
            subtract_global_minimum=True)
        print('Bandpass filtering completed. A new file has been generated.')
    else:
        print(str(bp_file) + " already exists, process skipped\n")


    projected_red = str(Path(os.getcwd(), input_isxd.stem + '-bp-mean_proj.isxd'))
    if not os.path.exists(projected_red):
        isx.project_movie(
            input_movie_files=bp_file,
            output_image_file=projected_red,
            stat_type='mean')  # types: 'mean', 'min', or 'max'
        print('mean projection completed.\n')
    else:
        print(projected_red + " already exists, process skipped\n")

    return projected_red


def LoG(n):
    """
    function generates a LoG filter
    The filter generation and convolution procedure were implemented
    based on functions described at
    https://projectsflix.com/opencv/laplacian-blob-detector-using-python/

    Parameters:
        n : int
            kernel size

    Returns: 2D matrix
        LoG filter with a given kernel size

    """

    n = np.ceil(n)
    sigma = n/6
    y, x = np.ogrid[-n//2:n//2+1, -n//2:n//2+1]
    y_filter = np.exp(-(y*y/(2.*sigma*sigma)))
    x_filter = np.exp(-(x*x/(2.*sigma*sigma)))
    final_filter = (-(2*sigma**2) + (x*x + y*y)) * \
        (x_filter*y_filter) * (1/(2*np.pi*sigma**4))

    return final_filter


def LoG_convolve(img_matrix, kernel_size):
    """
    function convolves image with a LoG filter with given kernel size
    and normalizes the output image to 16 bit scale

    Parameters:
        img_matrix: input image
        kernel_size: the kernel size for the LoG filter

    Returns: 2D matrix
        convolved and normalized image

    """

    filter_LoG = LoG(kernel_size)  # filter generation
    image = cv2.filter2D(img_matrix, -1, filter_LoG)  # convolving image
    image = np.pad(image, ((0, 0), (0, 0)), 'constant')  # padding
    image = np.square(image)  # squaring the response
    img_LoG_norm = np.zeros(image.shape)
    img_LoG_norm = cv2.normalize(image, img_LoG_norm,
                                 0, 65535, cv2.NORM_MINMAX)
    return img_LoG_norm


def mask_around_center(img_matrix, center, max_radius, min_radius=0):
    """
    function draws a circular mask confined by the two distance boundaries from a given center

    Parameters:
        img_matrix: input image matching the footprint image size
        center: int tuple
            cell centroid x and y coordinates
        max_radius: outer boundary
        min_radius: inner boundary

    Returns: True mask
        circular or donut shaped ROIs confined by the two distance boundaries from the center

    """

    h, w = img_matrix.shape[0], img_matrix.shape[1]
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = ((dist_from_center <= max_radius) &
            (dist_from_center >= min_radius))

    return mask


def ring_contrast(img_matrix, footprint, cell_size, n=2):
    """
    function calculates the luminance contrast
    by measuring the average cell body luminance and
    dividing it by the average ring background luminance.

    Parameters:
        img_matrix: input image matching the footprint image size
        footprint: 2D boolean matrix
            a single cell footprint
        cell_size: int
            cell size defining the cell body
        n: int
            background confined by cell_size and cell_size+n

    Returns:
        the average luminance ratio between the cell body and the background

    """

    cell_size_extra = cell_size + n
    peri = center = img_matrix.copy()

    centroids = ndimage.measurements.center_of_mass(footprint)
    y0, x0 = int(centroids[0]), int(centroids[1])

    center_mask = mask_around_center(img_matrix,
                                     center=(x0, y0),
                                     max_radius=cell_size)
    center[~center_mask] = 0

    peri_mask = mask_around_center(img_matrix,
                                   center=(x0, y0),
                                   max_radius=cell_size_extra,
                                   min_radius=cell_size)

    peri[~peri_mask] = 0

    return (np.mean(center)/np.mean(peri)).astype(float)


def footprint_filter(img_matrix, footprints, max_size, min_size, average_size=None, contrast=0.5):
    """
    function applies cell size and contrast (optional) constraints to filter footprints

    Parameters:
        img_matrix: input image matching the footprint image size
        footprints: 3D footprints matrix
            h x w x cell_index
        max_size: maximal size
        min_size: minimal size
            cell size filter limiting cell selection to the between range.
        average_size: optional
            used to calculate body-to-background contrast
        contrast:
            set the minimal body-to-background luminance contrast

    Returns:
        footprint subsets after size and contrast filtering

    """

    cell_number = footprints.shape[2]
    cell_filter = [False for i in range(cell_number)]

    for i in range(cell_number):
        if average_size is not None:
            cell_contrast = ring_contrast(img_matrix,
                                        footprints[:, :, i],
                                        average_size)
        cell_size = np.sum(footprints[:, :, i])

        if ((cell_size < max_size) and
                (cell_size > min_size) and
                ((cell_contrast > contrast) if average_size is not None else True)):

            cell_filter[i] = True

    return footprints[:, :, cell_filter]


def contour_to_footprint(img_matrix, contour):
    """
    function converts contour to footprints

    Parameters:
        img_matrix: input image matching the contour image size
        contour: threshold contour

    Returns:
        segmented footprints

    """

    labels = label(contour)
    props = regionprops(labels)
    h, w = img_matrix.shape[0], img_matrix.shape[1]
    footprints = np.empty(shape=(h, w, len(props)))

    for i, prop in enumerate(props):

        # option1
        xy_coord = prop.coords
        for each in xy_coord:
            footprints[each[0], each[1], i] = True

        # option2
        # if i is not 0:
        #   footprints[labels==i,i-1] = True

    return footprints


def threshold_img(img_matrix, std_factor):
    """
    function applies an intensity threshold to detect suprathreshold footprints

    Parameters:
        img_matrix: image to be thresholded
        std_factor: multiplier for standard deviation based threshold

    Returns:
        suprathreshold footprints

    """

    threshold = np.mean(img_matrix)+np.std(img_matrix)*std_factor
    _, contour = cv2.threshold(img_matrix,
                               threshold,
                               65535,
                               cv2.THRESH_BINARY)

    footprints = contour_to_footprint(img_matrix, contour)

    return footprints


def expand_footprints(img_matrix, footprints, threshold=0.7):
    """
    function adds suprathreshold ROIs from the input image to given footprints

    Parameters:
        img_matrix: raw image before LOG convolution
        footprints: footprints to be expanded
        threshold: raw image threshold

    Returns:
        expanded footprints

    """

    supertheshold_contour = (img_matrix >= np.max(img_matrix)*threshold)
    LoG_contour = np.ndarray.max(footprints, axis=2)
    expanded_contour = ((supertheshold_contour == 1) | (LoG_contour == 1))
    expanded_footprints = contour_to_footprint(img_matrix, expanded_contour)

    return expanded_footprints


def shrink_footprints(img_matrix, footprints, threshold=0.7):
    """
    function intersects suprathreshold RIOs with given footprints

    Parameters:
        img_matrix: raw image before LOG convolution
        footprints: input footprints
        threshold: raw image threshold

    Returns:
        shrinked footprints

    """

    supertheshold_contour = (img_matrix >= np.max(img_matrix)*threshold)
    LoG_contour = np.ndarray.max(footprints, axis=2)
    shrinked_contour = ((supertheshold_contour == 1) & (LoG_contour == 1))
    shrinked_footprints = contour_to_footprint(img_matrix, shrinked_contour)

    return shrinked_footprints


def split_merge_cells(img_matrix, LoG_footprints, medium_footprints, cell_size):
    """
    function splits medium size footprints based on watershed filter using local maximal Euclidean distance
    and merges with differential footprints sets (LoG_footprints-medium_footprints)

    Parameters:
        img_matrix: input image matching the footprint image size
        LoG_footprints: super sets
        medium_footprints:: medium size subsets
        cell_size: threshold to split medium size footprints

    Returns:
        split, and merged footprints

    """

    if medium_footprints.size != 0:

    # compute the exact Euclidean distance from every binary
    # pixel to the nearest zero pixel, then find peaks in this
    # distance map
        mask = np.ndarray.max(medium_footprints, axis=2)
        D = ndimage.distance_transform_edt(mask)
        localMax = peak_local_max(D,
                                indices=False,
                                min_distance=int(cell_size),
                                labels=mask.astype(int))

        # perform a connected component analysis on the local peaks,
        # using 8-connectivity, then apply the Watershed algorithm
        markers = ndimage.label(localMax,
                                structure=np.ones((3, 3)))[0]
        labels = watershed(-D, markers, mask=mask)

        # detect differential footprints that are not medium sized from expanded sets
        medium_size_contour = np.ndarray.max(medium_footprints, axis=2)
        expanded_contour = np.ndarray.max(LoG_footprints, axis=2)
        diff_contour = ((expanded_contour == 1) & (medium_size_contour == 0))
        diff_footprints = contour_to_footprint(img_matrix, diff_contour)

        # merge the footprints from split sets to difference sets
        split_footprints = contour_to_footprint(img_matrix, labels)
        final_footprints = np.concatenate(
            (diff_footprints, split_footprints), axis=2)
    else:
        final_footprints = LoG_footprints
        split_footprints = np.empty(shape=(0,0))
    return split_footprints, final_footprints


def merge_cells(small_footprints, splitted_footprints):
    """
    function merges two sets of footprints
    Parameters:
        splitted_footprints: medium size footprints after splitting
        small_footprints: small size footprints

    Returns:
        merged footprints

    """
    if splitted_footprints.size != 0:
        merged_footprints = np.concatenate(
            (splitted_footprints, small_footprints), axis=2)
        return merged_footprints
    else:
        return small_footprints


def footprints_export_to_isxd(input_isxd, footprints, suffix):
    """
    function exports footprints to isxd file

    Parameters:
        input_isxd: reference image to create cell sets and copy metadata
        footprints: sets to be exported
        suffix: suffix to autoname output isxd file

    Returns:
        footprint cellsets in isxd format

    """

    movie = isx.Movie.read(str(input_isxd))
    output = str(Path(os.getcwd(), input_isxd.stem+suffix+".isxd"))

    if os.path.exists(output):
        os.remove(output)
    cell_set = isx.CellSet.write(output, timing=isx.Timing(num_samples=1),
                                 spacing=movie.spacing)

    for i in range(footprints.shape[2]):
        image = footprints[:, :, i].astype(np.float32)
        trace = np.empty(1).astype(np.float32)
        cell_set.set_cell_data(i, image, trace, 'C{}'.format(i))

    cell_set.flush()
    del cell_set
    copy_metadata(input_isxd,output)
    return output

def read_metadata(filename, sizeof_size_t=8, endianness="little"):
    """
    Reads json metadata stored in an .isxd file

    Arguments
    ---------
    filename : str
        The .isxd filename
    sizeof_size_t : int > 0
        Number of bytes used to represent a size_t type variable in C++
    endianness
        Endianness of your machine

    Returns
    -------
    json_metadata : dict
        Metadata represented as a json dictionary
    """

    with open(filename, 'rb') as infile:
        # Inspired by isxJsonUtils.cpp
        infile.seek(-sizeof_size_t, 2)
        header_size = infile.read(sizeof_size_t)
        header_size = int.from_bytes(header_size, endianness)
        bottom_offset = header_size + 1 + sizeof_size_t
        infile.seek(-bottom_offset, 2)
        string_json = str(infile.read(bottom_offset - sizeof_size_t - 1).decode("utf-8"))
        json_metadata = json.loads(string_json)
    return json_metadata

def write_metadata(filename, json_metadata, sizeof_size_t=8, endianness="little"):
    """
    Writes json metadata to an .isxd file

    Arguments
    ---------
    filename : str
        The .isxd filename
    json_metadata : dict
        Metadata represented as a json dictionary
    sizeof_size_t : int > 0
        Number of bytes used to represent a size_t type variable in C++
    endianness
        Endianness of your machine
    """

    with open(filename, 'rb+') as infile:
        infile.seek(-sizeof_size_t, 2)
        header_size = infile.read(sizeof_size_t)
        header_size = int.from_bytes(header_size, endianness)
        bottom_offset = header_size + 1 + sizeof_size_t
        infile.seek(-bottom_offset, 2)

        infile.truncate()
        # process non-ascii characters correctly
        string_json = json.dumps(json_metadata, indent=4, ensure_ascii=False) + "\0"
        infile.write(bytes(string_json, "utf-8"))

        # calculate number of bytes in string by encoding to utf-8
        string_json = string_json.encode("utf-8")
        json_length = int.to_bytes(len(string_json) - 1, sizeof_size_t, endianness)
        infile.write(json_length)

def copy_metadata(input_isxd, autoseg_isxd):
    """
    function copies metadata from input_isxd to autoseg_isxd file

    Parameters:
        input_isxd: reference file that contains the metadata to be copied
        autoseg_isxd: newly generated isxd cellmap file, the metadata copy target

    Returns:
        None

    """

    # read metadata for autoseg isxd cell set
    autoseg_metadata = read_metadata(autoseg_isxd)
    # read metadata of input isxd movie
    movie_metadata = read_metadata(input_isxd)

    # overwrite extra properties of autoseg isxd cell set
    # the extraProperties key in the metadata is where information about the acquisition settings is stored
    # e.g., miniscope type (dual color, multiplane, etc.)
    autoseg_metadata['extraProperties'] = movie_metadata['extraProperties']
    write_metadata(autoseg_isxd, autoseg_metadata)

    return None


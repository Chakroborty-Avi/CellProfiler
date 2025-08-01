import numpy as np
import scipy.ndimage
import skimage.morphology
from numpy.typing import NDArray
from pydantic import validate_call, ConfigDict
from typing import Annotated, Optional, Dict, Any, Union
from centrosome.cpmorphology import fixup_scipy_ndimage_result as fix 
from cellprofiler_library.opts.measuregranularity import C_GRANULARITY


def rescale_pixel_data_and_mask(new_shape, subsample_size, im_pixel_data, im_mask, dimensions):
    if dimensions == 2:
        i, j = (
            np.mgrid[0 : new_shape[0], 0 : new_shape[1]].astype(float)
            / subsample_size
        )
        pixels = scipy.ndimage.map_coordinates(im_pixel_data, (i, j), order=1)
        mask = (
            scipy.ndimage.map_coordinates(im_mask.astype(float), (i, j)).astype(float) > 0.9
        )
    else:
        k, i, j = (
            np.mgrid[0 : new_shape[0], 0 : new_shape[1], 0 : new_shape[2]].astype(float)
            / subsample_size
        )
        pixels = scipy.ndimage.map_coordinates(im_pixel_data, (k, i, j), order=1)
        mask = (
            scipy.ndimage.map_coordinates(im_mask.astype(float), (k, i, j)).astype(float) > 0.9
            )
    return pixels, mask


def get_mapped(dimensions, new_shape, back_shape, back_pixels):
    if dimensions == 2:
        i, j = np.mgrid[0 : new_shape[0], 0 : new_shape[1]].astype(float)
        #
        # Make sure the mapping only references the index range of
        # back_pixels.
        #
        i *= float(back_shape[0] - 1) / float(new_shape[0] - 1)
        j *= float(back_shape[1] - 1) / float(new_shape[1] - 1)
        back_pixels = scipy.ndimage.map_coordinates(back_pixels, (i, j), order=1).astype(float)
    else:
        k, i, j = np.mgrid[
            0 : new_shape[0], 0 : new_shape[1], 0 : new_shape[2]
        ].astype(float)
        k *= float(back_shape[0] - 1) / float(new_shape[0] - 1)
        i *= float(back_shape[1] - 1) / float(new_shape[1] - 1)
        j *= float(back_shape[2] - 1) / float(new_shape[2] - 1)
        back_pixels = scipy.ndimage.map_coordinates(back_pixels, (k, i, j), order=1).astype(float)
    return back_pixels


def get_footprint(radius, dimensions):
    if dimensions == 2:
        footprint = skimage.morphology.disk(radius, dtype=bool)
    else:
        footprint = skimage.morphology.ball(radius, dtype=bool)
    return footprint

def measure_granularity(
        im_pixel_data: NDArray[np.float32],
        im_mask: NDArray[np.bool_],
        subsample_size: float,
        image_sample_size: float,
        element_size: int,
        object_records,
        granular_spectrum_length,
        image_name: str,
        dimensions: int = 2,
        ):
    #
    # Downsample the image and mask
    #
    new_shape = np.array(im_pixel_data.shape)
    if subsample_size < 1:
        new_shape = new_shape * subsample_size
        pixels, mask = rescale_pixel_data_and_mask(new_shape, subsample_size, im_pixel_data, im_mask, dimensions)
    else:
        pixels = im_pixel_data.copy()
        mask = im_mask.copy()
    #
    # Remove background pixels using a greyscale tophat filter
    #
    if image_sample_size < 1:
        back_shape = new_shape * image_sample_size
        back_pixels, back_mask = rescale_pixel_data_and_mask(back_shape, image_sample_size, im_pixel_data, im_mask, dimensions)
    else:
        back_pixels = pixels
        back_mask = mask
        back_shape = new_shape
    radius = element_size
    if dimensions == 2:
        footprint = skimage.morphology.disk(radius, dtype=bool)
    else:
        footprint = skimage.morphology.ball(radius, dtype=bool)
        
    back_pixels_mask = np.zeros_like(back_pixels)
    back_pixels_mask[back_mask == True] = back_pixels[back_mask == True]
    back_pixels = skimage.morphology.erosion(back_pixels_mask, footprint=footprint)
    back_pixels_mask = np.zeros_like(back_pixels)
    back_pixels_mask[back_mask == True] = back_pixels[back_mask == True]
    back_pixels = skimage.morphology.dilation(back_pixels_mask, footprint=footprint)
    if image_sample_size < 1:
        back_pixels = get_mapped(dimensions, new_shape, back_shape, back_pixels)
    pixels -= back_pixels
    pixels[pixels < 0] = 0

    #
    # Transcribed from the Matlab module: granspectr function
    #
    # CALCULATES GRANULAR SPECTRUM, ALSO KNOWN AS SIZE DISTRIBUTION,
    # GRANULOMETRY, AND PATTERN SPECTRUM, SEE REF.:
    # J.Serra, Image Analysis and Mathematical Morphology, Vol. 1. Academic Press, London, 1989
    # Maragos,P. "Pattern spectrum and multiscale shape representation", IEEE Transactions on Pattern Analysis and Machine Intelligence, 11, N 7, pp. 701-716, 1989
    # L.Vincent "Granulometries and Opening Trees", Fundamenta Informaticae, 41, No. 1-2, pp. 57-90, IOS Press, 2000.
    # L.Vincent "Morphological Area Opening and Closing for Grayscale Images", Proc. NATO Shape in Picture Workshop, Driebergen, The Netherlands, pp. 197-208, 1992.
    # I.Ravkin, V.Temov "Bit representation techniques and image processing", Applied Informatics, v.14, pp. 41-90, Finances and Statistics, Moskow, 1988 (in Russian)
    # THIS IMPLEMENTATION INSTEAD OF OPENING USES EROSION FOLLOWED BY RECONSTRUCTION
    #
    ng = granular_spectrum_length
    startmean = np.mean(pixels[mask])
    ero = pixels.copy()
    # Mask the test image so that masked pixels will have no effect
    # during reconstruction
    #
    ero[~mask] = 0
    currentmean = startmean
    startmean = max(startmean, np.finfo(float).eps)
    measurements_arr = []
    image_measurements_arr = []

    if dimensions == 2:
        footprint = skimage.morphology.disk(1, dtype=bool)
    else:
        footprint = skimage.morphology.ball(1, dtype=bool)
    statistics = [image_name]
    for i in range(1, ng + 1):
        prevmean = currentmean
        ero_mask = np.zeros_like(ero)
        ero_mask[mask == True] = ero[mask == True]
        ero = skimage.morphology.erosion(ero_mask, footprint=footprint)
        rec = skimage.morphology.reconstruction(ero, pixels, footprint=footprint)
        currentmean = np.mean(rec[mask])
        gs = (prevmean - currentmean) * 100 / startmean
        statistics += ["%.2f" % gs]
        feature = C_GRANULARITY % (i, image_name)
        image_measurements_arr += [(feature, gs)]
        # measurements.add_image_measurement(feature, gs)
        #
        # Restore the reconstructed image to the shape of the
        # original image so we can match against object labels
        #
        orig_shape = im_pixel_data.shape
        rec = get_mapped(dimensions, orig_shape, new_shape, rec)

        #
        # Calculate the means for the objects
        #
        for object_record in object_records:
            assert isinstance(object_record, ObjectRecord)
            if object_record.nobjects > 0:
                new_mean = fix(
                    scipy.ndimage.mean(
                        rec, object_record.labels, object_record.range
                    )
                )
                gss = (
                    (object_record.current_mean - new_mean)
                    * 100
                    / object_record.start_mean
                )
                object_record.current_mean = new_mean
            else:
                gss = np.zeros((0,))
            # measurements.add_measurement(object_record.name, feature, gss)
            measurements_arr += [(object_record.name, feature, gss)]
    return measurements_arr, image_measurements_arr, statistics

#
# For each object, build a little record
#

class ObjectRecord(object):
    def __init__(self, name, segmented, im_mask, im_pixel_data):
        self.name = name
        self.labels = segmented
        self.nobjects = np.max(self.labels)
        if self.nobjects != 0:
            self.range = np.arange(1, np.max(self.labels) + 1)
            self.labels = self.labels.copy()
            self.labels[~im_mask] = 0
            self.current_mean = fix(
                scipy.ndimage.mean(im_pixel_data, self.labels, self.range)
            )
            self.start_mean = np.maximum(
                self.current_mean, np.finfo(float).eps
            )
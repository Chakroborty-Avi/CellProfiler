from typing import List, Tuple, Generator
from ..types import Image2DGrayscale, ImageGrayscale, ImageGrayscaleMask
import numpy
from ..functions.image_processing import apply_threshold, get_global_threshold
import cellprofiler_library.opts.threshold as Threshold
from ..functions.measurement import measure_correlation_and_slope, measure_manders_coefficient, measure_rwc_coefficient, measure_overlap_coefficient, measure_costes_coefficient, get_thresholded_images_and_counts
from ..opts.measurecolocalization import MeasurementFormat, MeasurementType
from pydantic import Field
from typing import Annotated, Optional, Dict, Any, Union
from ..opts.measurecolocalization import CostesMethod
def measure_colocalization(args):
    pass

def measure_colocalization_images(args):
    pass

def measure_colocalization_objects(args):
    pass

def measure_colocalization_images_and_objects(args):
    pass

def run_image_pair_images(
    first_pixel_data: Annotated[ImageGrayscale, Field(description="First image pixel data")],
    second_pixel_data: Annotated[ImageGrayscale, Field(description="Second image pixel data")],
    first_image_name: Annotated[str, Field(description="First image name")] = "First image",
    second_image_name: Annotated[str, Field(description="Second image name")] = "Second image",
    mask: Annotated[Optional[ImageGrayscaleMask], Field(description="Mask")] = None, 
    first_threshold_value: Annotated[float, Field(description="First image threshold value")]=100, 
    second_threshold_value: Annotated[float, Field(description="Second image threshold value")]=100, 
    measurement_types: Annotated[List[MeasurementType], Field(description="Measurement types")] = [MeasurementType.CORRELATION, MeasurementType.MANDERS, MeasurementType.RWC, MeasurementType.OVERLAP, MeasurementType.COSTES],
    **kwargs
    ) -> Tuple[List[Tuple[str, float]], Dict[str, float]]:
    """Calculate the correlation between the pixels of two images"""

    result = []
    corr = numpy.NaN
    slope = numpy.NaN
    C1 = numpy.NaN
    C2 = numpy.NaN
    M1 = numpy.NaN
    M2 = numpy.NaN
    RWC1 = numpy.NaN
    RWC2 = numpy.NaN
    overlap = numpy.NaN
    K1 = numpy.NaN
    K2 = numpy.NaN
    if mask is not None and numpy.any(mask):
        fi = first_pixel_data[mask]
        si = second_pixel_data[mask]

        if MeasurementType.CORRELATION in measurement_types:
            res, corr, slope = measure_correlation_and_slope(fi, si, first_image_name, second_image_name)
            result += res

        combined_thresh = None
        fi_thresh = None
        si_thresh = None
        tot_fi_thr = None
        tot_si_thr = None
        if set(measurement_types).intersection({MeasurementType.MANDERS, MeasurementType.RWC, MeasurementType.OVERLAP}) != set():
            # Get channel-specific thresholds from thresholds array
            # Threshold as percentage of maximum intensity in each channel
            _, _, fi_thresh, si_thresh, tot_fi_thr, tot_si_thr, combined_thresh = get_thresholded_images_and_counts(fi, si, first_threshold_value, second_threshold_value)

        if MeasurementType.MANDERS in measurement_types:
            res, M1, M2 = measure_manders_coefficient(fi, si, first_image_name, second_image_name, None, None, fi_thresh, si_thresh, tot_fi_thr, tot_si_thr)
            result += res

        if MeasurementType.RWC in measurement_types:
            res, RWC1, RWC2 = measure_rwc_coefficient(fi, si, first_image_name, second_image_name, None, None, fi_thresh, si_thresh, tot_fi_thr, tot_si_thr, combined_thresh)
            result += res

        if MeasurementType.OVERLAP in measurement_types:
            res, overlap, K1, K2 = measure_overlap_coefficient(fi, si, first_image_name, second_image_name, None, None, fi_thresh, si_thresh)
            result += res

        if MeasurementType.COSTES in measurement_types:
            first_image_scale = kwargs.get("first_image_scale", None)
            second_image_scale = kwargs.get("second_image_scale", None)
            costes_method = kwargs.get("costes_method", CostesMethod.FAST)
            assert costes_method in CostesMethod.__members__.values()
            res, C1, C2 = measure_costes_coefficient(
                fi, 
                si, 
                first_image_name, 
                second_image_name, 
                first_image_scale,
                second_image_scale, 
                None, 
                None,
                fi_thresh, 
                si_thresh,
                costes_method=costes_method
                )
            result += res

    else:
        corr = numpy.NaN
        slope = numpy.NaN
        C1 = numpy.NaN
        C2 = numpy.NaN
        M1 = numpy.NaN
        M2 = numpy.NaN
        RWC1 = numpy.NaN
        RWC2 = numpy.NaN
        overlap = numpy.NaN
        K1 = numpy.NaN
        K2 = numpy.NaN

    #
    # Add the measurements
    #
    corr_measurement = MeasurementFormat.CORRELATION_FORMAT % (first_image_name, second_image_name)
    slope_measurement = MeasurementFormat.SLOPE_FORMAT % (first_image_name, second_image_name)
    overlap_measurement = MeasurementFormat.OVERLAP_FORMAT % (first_image_name, second_image_name)

    k_measurement_1 = MeasurementFormat.K_FORMAT % (first_image_name, second_image_name)
    k_measurement_2 = MeasurementFormat.K_FORMAT % (second_image_name, first_image_name)
    
    manders_measurement_1 = MeasurementFormat.MANDERS_FORMAT % (first_image_name, second_image_name)
    manders_measurement_2 = MeasurementFormat.MANDERS_FORMAT % (second_image_name, first_image_name)

    rwc_measurement_1 = MeasurementFormat.RWC_FORMAT % (first_image_name, second_image_name)
    rwc_measurement_2 = MeasurementFormat.RWC_FORMAT % (second_image_name, first_image_name)

    costes_measurement_1 = MeasurementFormat.COSTES_FORMAT % (first_image_name, second_image_name)
    costes_measurement_2 = MeasurementFormat.COSTES_FORMAT % (second_image_name, first_image_name)
    measurements = {}
    if MeasurementType.CORRELATION in measurement_types:
        measurements[corr_measurement] = corr
        measurements[slope_measurement] = slope
    if MeasurementType.OVERLAP in measurement_types:        
        measurements[overlap_measurement] = overlap
        measurements[k_measurement_1] = K1
        measurements[k_measurement_2] = K2

    if MeasurementType.MANDERS in measurement_types:
        measurements[manders_measurement_1] = M1
        measurements[manders_measurement_2] = M2
    if MeasurementType.RWC in measurement_types:
        measurements[rwc_measurement_1] = RWC1
        measurements[rwc_measurement_2] = RWC2
    if MeasurementType.COSTES in measurement_types:
        measurements[costes_measurement_1] = C1
        measurements[costes_measurement_2] = C2

    return result, measurements
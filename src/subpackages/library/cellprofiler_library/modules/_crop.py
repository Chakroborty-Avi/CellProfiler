from typing import Annotated, Optional, Tuple
from pydantic import Field, validate_call, ConfigDict
from ..types import Image2D, Image2DMask
import numpy
from cellprofiler_library.functions.image_processing import get_cropped_mask, get_cropped_image_mask, get_cropped_image_pixels
from ..opts.crop import RemovalMethod

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def remove_rows_and_columns(
        removal_method:     Annotated[RemovalMethod, Field(description="Removal method")],
        orig_image_pixels:  Annotated[Image2D, Field(description="Pixel values of the original image")],
        cropping:           Annotated[Image2DMask, Field(description="The region of interest to be kept. 1 for pixels to keep, 0 for pixels to remove")],
        mask:               Annotated[Optional[Image2DMask], Field(description="Previous cropping's mask")],
        orig_image_mask:    Annotated[Optional[Image2DMask], Field(description="Mask that may have been set on the original image")],
        ) -> Tuple[Image2D, Image2DMask, Image2DMask]:
    #
    # Crop the mask
    #
    mask = get_cropped_mask(cropping, mask, removal_method)

    #
    # Crop the image_mask 
    image_mask = get_cropped_image_mask(cropping, mask, orig_image_mask, removal_method)

    #
    # Crop the image
    #
    cropped_pixel_data = get_cropped_image_pixels(orig_image_pixels, cropping, mask, removal_method)

    return cropped_pixel_data, mask, image_mask


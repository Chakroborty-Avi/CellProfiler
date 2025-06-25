from typing import Annotated, Optional, Tuple
from pydantic import Field, validate_call, ConfigDict
from ..types import Image2D, Image2DMask
import numpy
from cellprofiler_library.functions.image_processing import get_final_cropping_keep_rows_and_columns, get_final_cropping_remove_rows_and_columns, apply_crop_keep_rows_and_columns, apply_crop_remove_rows_and_columns
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
    # Get the final cropping
    #
    mask, image_mask = get_final_cropping(cropping, mask, orig_image_mask, removal_method)

    #
    # apply final cropping based on the removal method
    #
    if removal_method == RemovalMethod.NO:
        cropped_pixel_data = apply_crop_keep_rows_and_columns(orig_image_pixels, cropping)
    elif removal_method in (RemovalMethod.EDGES, RemovalMethod.ALL):
        cropped_pixel_data = apply_crop_remove_rows_and_columns(orig_image_pixels, cropping, mask, removal_method)
    else:
        raise NotImplementedError(f"Unimplemented removal method: {removal_method}")

    return cropped_pixel_data, mask, image_mask

def get_final_cropping(
        cropping:           Annotated[Image2DMask, Field(description="The region of interest to be kept. 1 for pixels to keep, 0 for pixels to remove")],
        mask:               Annotated[Optional[Image2DMask], Field(description="Previous cropping's mask")],
        orig_image_mask:    Annotated[Optional[Image2DMask], Field(description="Mask that may have been set on the original image")] = None,
        removal_method:     Annotated[RemovalMethod, Field(description="Removal method")] = RemovalMethod.NO,
) -> Tuple[Image2DMask, Image2DMask]:
    image_mask = None
    if removal_method == RemovalMethod.NO:
        mask, image_mask = get_final_cropping_keep_rows_and_columns(
            cropping, mask, orig_image_mask
        )
    elif removal_method in (RemovalMethod.EDGES, RemovalMethod.ALL):
        crop_internal = removal_method == RemovalMethod.ALL
        mask, image_mask = get_final_cropping_remove_rows_and_columns(
            cropping, mask, orig_image_mask, crop_internal
        )
    else:
        raise NotImplementedError(f"Unimplemented removal method: {removal_method}")
    
    return mask, image_mask


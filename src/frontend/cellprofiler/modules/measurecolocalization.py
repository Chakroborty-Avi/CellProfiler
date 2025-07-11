"""
MeasureColocalization
=====================

**MeasureColocalization** measures the colocalization and correlation
between intensities in different images (e.g., different color channels)
on a pixel-by-pixel basis, within identified objects or across an entire
image.

Given two or more images, this module calculates the correlation &
colocalization (Overlap, Manders, Costes’ Automated Threshold & Rank
Weighted Colocalization) between the pixel intensities. The correlation
/ colocalization can be measured for entire images, or a correlation
measurement can be made within each individual object. Correlations /
Colocalizations will be calculated between all pairs of images that are
selected in the module, as well as between selected objects. For
example, if correlations are to be measured for a set of red, green, and
blue images containing identified nuclei, measurements will be made
between the following:

-  The blue and green, red and green, and red and blue images.
-  The nuclei in each of the above image pairs.

A good primer on colocalization theory can be found on the `SVI website`_.

You can find a helpful review on colocalization from Aaron *et al*. `here`_.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          YES
============ ============ ===============

Measurements made by this module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  *Correlation:* The correlation between a pair of images *I* and *J*,
   calculated as Pearson’s correlation coefficient. The formula is
   covariance(\ *I* ,\ *J*)/[std(\ *I* ) × std(\ *J*)].
-  *Slope:* The slope of the least-squares regression between a pair of
   images I and J. Calculated using the model *A* × *I* + *B* = *J*, where *A* is the slope.
-  *Overlap coefficient:* The overlap coefficient is a modification of
   Pearson’s correlation where average intensity values of the pixels are
   not subtracted from the original intensity values. For a pair of
   images R and G, the overlap coefficient is measured as r = sum(Ri \*
   Gi) / sqrt (sum(Ri\*Ri)\*sum(Gi\*Gi)).
-  *Manders coefficient:* The Manders coefficient for a pair of images R
   and G is measured as M1 = sum(Ri_coloc)/sum(Ri) and M2 =
   sum(Gi_coloc)/sum(Gi), where Ri_coloc = Ri when Gi > 0, 0 otherwise
   and Gi_coloc = Gi when Ri >0, 0 otherwise.
-  *Manders coefficient (Costes Automated Threshold):* Costes’ automated
   threshold estimates maximum threshold of intensity for each image
   based on correlation. Manders coefficient is applied on thresholded
   images as Ri_coloc = Ri when Gi > Gthr and Gi_coloc = Gi when Ri >
   Rthr where Gthr and Rthr are thresholds calculated using Costes’
   automated threshold method.
-  *Rank Weighted Colocalization coefficient:* The RWC coefficient for a
   pair of images R and G is measured as RWC1 =
   sum(Ri_coloc\*Wi)/sum(Ri) and RWC2 = sum(Gi_coloc\*Wi)/sum(Gi),
   where Wi is Weight defined as Wi = (Rmax - Di)/Rmax where Rmax is the
   maximum of Ranks among R and G based on the max intensity, and Di =
   abs(Rank(Ri) - Rank(Gi)) (absolute difference in ranks between R and
   G) and Ri_coloc = Ri when Gi > 0, 0 otherwise and Gi_coloc = Gi
   when Ri >0, 0 otherwise. (Singan et al. 2011, BMC Bioinformatics
   12:407).

References
^^^^^^^^^^

-  Aaron JS, Taylor AB, Chew TL. Image co-localization - co-occurrence versus correlation.
   J Cell Sci. 2018;131(3):jcs211847. Published 2018 Feb 8. doi:10.1242/jcs.211847


   
.. _SVI website: http://svi.nl/ColocalizationTheory
.. _here: https://jcs.biologists.org/content/joces/131/3/jcs211847.full.pdf
"""

import numpy
import scipy.ndimage
import scipy.stats
from cellprofiler_core.constants.measurement import COLTYPE_FLOAT
from cellprofiler_core.module import Module
from cellprofiler_core.setting import Divider, Binary, ValidationError
from cellprofiler_core.setting.choice import Choice
from cellprofiler_core.setting.subscriber import (
    LabelListSubscriber,
    ImageListSubscriber,
)
from cellprofiler_core.setting import SettingsGroup, HiddenCount
from cellprofiler_core.setting.text import Float
from cellprofiler_core.setting.subscriber import ImageSubscriber, LabelSubscriber
from cellprofiler_core.setting.do_something import DoSomething, RemoveSettingButton
from cellprofiler_core.utilities.core.object import size_similarly
from centrosome.cpmorphology import fixup_scipy_ndimage_result as fix
from scipy.linalg import lstsq
from cellprofiler_core.setting.text import ImageName
from cellprofiler_core.image import Image
from cellprofiler_library.functions.image_processing import apply_threshold_to_objects
from cellprofiler_library.functions.measurement import measure_correlation_and_slope_from_objects, measure_manders_coefficient_from_objects, measure_rwc_coefficient_from_objects, measure_overlap_coefficient_from_objects, measure_costes_coefficient_from_objects, get_thresholded_images_and_counts
from cellprofiler_library.functions.image_processing import apply_threshold, get_global_threshold
import cellprofiler_library.opts.threshold as Threshold 
from cellprofiler_library.opts.measurecolocalization import MeasurementType
from cellprofiler_library.modules._measurecolocalization import run_image_pair_images, run_image_pair_objects

M_IMAGES = "Across entire image"
M_OBJECTS = "Within objects"
M_IMAGES_AND_OBJECTS = "Both"

# The number of settings per threshold
THRESHOLD_SETTING_COUNT = 2

# The number of settings per save mask
SAVE_MASK_SETTING_COUNT = 3

# The number of settings other than the threshold or save image mask settings 
FIXED_SETTING_COUNT = 17

M_FAST = "Fast"
M_FASTER = "Faster"
M_ACCURATE = "Accurate"

"""Feature name format for the correlation measurement"""
F_CORRELATION_FORMAT = "Correlation_Correlation_%s_%s"

"""Feature name format for the slope measurement"""
F_SLOPE_FORMAT = "Correlation_Slope_%s_%s"

"""Feature name format for the overlap coefficient measurement"""
F_OVERLAP_FORMAT = "Correlation_Overlap_%s_%s"

"""Feature name format for the Manders Coefficient measurement"""
F_K_FORMAT = "Correlation_K_%s_%s"

"""Feature name format for the Manders Coefficient measurement"""
F_KS_FORMAT = "Correlation_KS_%s_%s"

"""Feature name format for the Manders Coefficient measurement"""
F_MANDERS_FORMAT = "Correlation_Manders_%s_%s"

"""Feature name format for the RWC Coefficient measurement"""
F_RWC_FORMAT = "Correlation_RWC_%s_%s"

"""Feature name format for the Costes Coefficient measurement"""
F_COSTES_FORMAT = "Correlation_Costes_%s_%s"

class MeasureColocalization(Module):
    module_name = "MeasureColocalization"
    category = "Measurement"
    variable_revision_number = 6

    def create_settings(self):
        """Create the initial settings for the module"""

        self.images_list = ImageListSubscriber(
            "Select images to measure",
            [],
            doc="""Select images to measure the correlation/colocalization in.""",
        )

        self.objects_list = LabelListSubscriber(
            "Select objects to measure",
            [],
            doc="""\
*(Used only when "Within objects" or "Both" are selected)*

Select the objects to be measured.""",
        )

        self.thresholds_list = []

        self.thr = Float(
            "Set threshold as percentage of maximum intensity for the images",
            15,
            minval=0,
            maxval=99,
            doc="""\
You may choose to measure colocalization metrics only for those pixels above 
a certain threshold. Select the threshold as a percentage of the maximum intensity 
of the above image [0-99].

This value is used by the Overlap, Manders, and Rank Weighted Colocalization 
measurements.
""",
        )

        self.images_or_objects = Choice(
            "Select where to measure correlation",
            [M_IMAGES, M_OBJECTS, M_IMAGES_AND_OBJECTS],
            doc="""\
You can measure the correlation in several ways:

-  *%(M_OBJECTS)s:* Measure correlation only in those pixels previously
   identified as within an object. You will be asked to choose which object
   type to measure within.
-  *%(M_IMAGES)s:* Measure the correlation across all pixels in the
   images.
-  *%(M_IMAGES_AND_OBJECTS)s:* Calculate both measurements above.

All methods measure correlation on a pixel by pixel basis.
"""
            % globals(),
        )

        self.spacer = Divider(line=True)
        self.spacer_2 = Divider(line=True)
        self.thresholds_count = HiddenCount(self.thresholds_list, "Threshold count")
        self.wants_channel_thresholds = Binary(
            "Enable image specific thresholds?",
            False,
            doc="""\
Select *{YES}* to specify a unique threshold for selected images. Default value set above will be used for all selected images without a custom threshold.
        """.format(
                **{"YES": "Yes"}
            ),
            callback=self.__auto_add_threshold_input_box,
        )
        self.wants_threshold_visualization = Binary(
            "Enable threshold visualization?",
            False,
            doc="""
Select *{YES}* to choose images to visualize the thresholding output. This outputs the image mask that is generated after thresholding.
        """.format(
                **{"YES": "Yes"}
            )
        )
        self.threshold_visualization_list = ImageListSubscriber(
            "Select images to visualize thresholds",
            [],
            doc="""
Select images to visualize the thresholding output.
        """.format(
                **{"YES": "Yes"}
            ),
        )

        self.do_all = Binary(
            "Run all metrics?",
            True,
            doc="""\
Select *{YES}* to run all of CellProfiler's correlation 
and colocalization algorithms on your images and/or objects; 
otherwise select *{NO}* to pick which correlation and 
colocalization algorithms to run.
""".format(
                **{"YES": "Yes", "NO": "No"}
            ),
        )

        self.do_corr_and_slope = Binary(
            "Calculate correlation and slope metrics?",
            True,
            doc="""\
Select *{YES}* to run the Pearson correlation and slope metrics.
""".format(
                **{"YES": "Yes"}
            ),
        )

        self.do_manders = Binary(
            "Calculate the Manders coefficients?",
            True,
            doc="""\
Select *{YES}* to run the Manders coefficients.
""".format(
                **{"YES": "Yes"}
            ),
        )

        self.do_rwc = Binary(
            "Calculate the Rank Weighted Colocalization coefficients?",
            True,
            doc="""\
Select *{YES}* to run the Rank Weighted Colocalization coefficients.
""".format(
                **{"YES": "Yes"}
            ),
        )

        self.do_overlap = Binary(
            "Calculate the Overlap coefficients?",
            True,
            doc="""\
Select *{YES}* to run the Overlap coefficients.
""".format(
                **{"YES": "Yes"}
            ),
        )

        self.do_costes = Binary(
            "Calculate the Manders coefficients using Costes auto threshold?",
            True,
            doc="""\
Select *{YES}* to run the Manders coefficients using Costes auto threshold.
""".format(
                **{"YES": "Yes"}
            ),
        )

        self.fast_costes = Choice(
            "Method for Costes thresholding",
            [M_FASTER, M_FAST, M_ACCURATE],
            doc=f"""\
This setting determines the method used to calculate the threshold for use within the
Costes calculations. The *{M_FAST}* and *{M_ACCURATE}* modes will test candidate thresholds
in descending order until the optimal threshold is reached. Selecting *{M_FAST}* will attempt 
to skip candidates when results are far from the optimal value being sought. Selecting *{M_ACCURATE}* 
will test every possible threshold value. When working with 16-bit images these methods can be extremely 
time-consuming. Selecting *{M_FASTER}* will use a modified bisection algorithm to find the threshold 
using a shrinking window of candidates. This is substantially faster but may produce slightly lower 
thresholds in exceptional circumstances.

In the vast majority of instances the results of all strategies should be identical. We recommend using 
*{M_FAST}* mode when working with 8-bit images and *{M_FASTER}* mode when using 16-bit images.

Alternatively, you may want to disable these specific measurements entirely 
(available when "*Run All Metrics?*" is set to "*No*").
"""
        )
        self.add_threshold_button = DoSomething("", "Add another threshold", self.add_threshold)
        self.save_mask_list = []
        self.save_image_mask_count = HiddenCount(self.save_mask_list, "Save mask count")
        self.wants_masks_saved = Binary(
            "Save thresholded mask?",
            False,
            doc="""Select *{YES}* to save the masks obtained after performing the thresholding operation.
            """.format(**{'YES': "Yes"}),
            callback=self.__auto_add_save_mask_input_box,
        )
        self.add_save_mask_button = DoSomething("", "Add another save mask", self.add_save_mask)

    def __auto_add_threshold_input_box(self, _):
        if not self.wants_channel_thresholds.value:
            if self.thresholds_count.value == 0:
                self.add_threshold()

    def __auto_add_save_mask_input_box(self, _):
        if not self.wants_masks_saved.value:
            if self.save_image_mask_count.value == 0:
                self.add_save_mask()
        
    def add_threshold(self, removable=True):
        group = SettingsGroup()
        group.removable = removable
        
        group.append(
            "image_name",
            ImageSubscriber(
                "Select the image",
                "None",
                doc="""\
Select the image that you want to use for this operation.""",
            ),
        )
        group.append(
            "threshold_for_channel",
            Float(
                "Set threshold as percentage of maximum intensity of selected image",
                15.0,
                minval=0.0,
                maxval=99.0,
                doc="""\
Select the threshold as a percentage of the maximum intensity of the above image [0-99].
You can set a different threshold for each image selected in the module.
""",
            ),
        )

        if removable:
            group.append("remover", RemoveSettingButton("", "Remove this image", self.thresholds_list, group))
        group.append("divider", Divider())
        self.thresholds_list.append(group)

    def add_save_mask(self, removable=True):
        """Add a new group for each image to save the mask for"""
        group = SettingsGroup()
        group.removable = removable
        """Save the thresholded mask to the image set"""
        
        # The name of the image from the image set
        group.append(
            "image_name",
            ImageSubscriber(
                "Which image mask would you like to save",
                doc="""Select the image mask that you would like to save. The default thresholding value will be used unless an image specific threshold is specified. The mask will be saved as a new image in the image set.""",
            )
        )

        # ask if the user wants to perform thresholding over the entire image or a specific object
        group.append(
            "save_mask_wants_objects",
            Binary(
                "Use object for thresholding?",
                False,
                doc="""\
    Select *{YES}* to use obejcts when performing the thresholding operation.
            """.format(
                    **{"YES": "Yes"}
                ),
                callback=self.__auto_add_threshold_input_box,
            )
        )

        # The name of the object that the user would like to use for thresholding (this is visible only if save_mask_wants_objects is selected)
        group.append(
            "choose_object",
            LabelSubscriber(
                "Select an Object for threhsolding",
                "Select an Object",
                doc="""Select the name of the object that you would like to use to generate the mask. Custom threshold is applied if previously specified; default value will be used otherwise"""
            )
        )
        
        # This is the name that will be given to the new image (mask) that is created by thresholding
        group.append(
            "save_image_name",
            ImageName(
            "Name the output image",
            "ColocalizationMask",
            doc="""Enter the name you want to call the image mask produced by this module. """,
            )
        )

        if removable:
            group.append("remover", RemoveSettingButton("", "Remove this image", self.save_mask_list, group))
        group.append("divider", Divider())
        self.save_mask_list.append(group)

    def settings(self):
        """Return the settings to be saved in the pipeline"""
        result = [
            self.images_list,
            self.thr
            ]
        result += [self.wants_channel_thresholds, self.thresholds_count]
        for threshold in self.thresholds_list:
            result += [threshold.image_name, threshold.threshold_for_channel]
        result += [
            self.wants_threshold_visualization,
            self.threshold_visualization_list,
            self.images_or_objects,
            self.objects_list,
            self.do_all,
            self.do_corr_and_slope,
            self.do_manders,
            self.do_rwc,
            self.do_overlap,
            self.do_costes,
            self.fast_costes,
            self.wants_masks_saved,
            self.save_image_mask_count,
        ]
        for save_mask in self.save_mask_list:
            # image_name is the name of the image in the image set
            # save_image_name is the name that the user would like to give to the output mask
            result += [save_mask.image_name, save_mask.save_mask_wants_objects] 
            if save_mask.save_mask_wants_objects.value:
                result += [save_mask.choose_object] 
            result += [save_mask.save_image_name]

        return result

    def visible_settings(self):
        result = [
            self.images_list,
            self.spacer,
            self.thr,
            self.wants_channel_thresholds,
        ]
        if self.wants_channel_thresholds.value:
            for threshold in self.thresholds_list:
                result += [threshold.image_name, threshold.threshold_for_channel]
                if threshold.removable:
                    result += [threshold.remover, Divider(line=False)]
            result += [self.add_threshold_button, self.spacer_2]
        result += [self.wants_threshold_visualization]
        if self.wants_threshold_visualization.value == True:
            result += [self.threshold_visualization_list]
        result += [self.images_or_objects,]
        if self.wants_objects():
            result += [self.objects_list]
        result += [self.do_all]
        if not self.do_all:
            result += [
                self.do_corr_and_slope,
                self.do_manders,
                self.do_rwc,
                self.do_overlap,
                self.do_costes,
            ]
        if self.do_all or self.do_costes:
            result += [self.fast_costes]
        result += [Divider(line=True)]
        result += [ self.wants_masks_saved ]
        if self.wants_masks_saved.value:
            for save_mask in self.save_mask_list:
                result += [save_mask.image_name, save_mask.save_mask_wants_objects]
                if save_mask.save_mask_wants_objects.value:
                    # Object selector is shown only if the radio button save_mask_wants_objects is selected
                    result += [save_mask.choose_object]
                result += [save_mask.save_image_name]
                if save_mask.removable:
                    result += [save_mask.remover, Divider(line=False)]
            result += [self.add_save_mask_button]
        return result

    def help_settings(self):
        """Return the settings to be displayed in the help menu"""
        help_settings = [
            self.images_or_objects,
            self.thr,
            self.wants_channel_thresholds,
            self.wants_threshold_visualization,
            self.threshold_visualization_list,

            self.images_list,
            self.objects_list,
            self.do_all,
            self.fast_costes,
            self.wants_masks_saved
        ]
        return help_settings
    
    def prepare_settings(self, setting_values):
        value_count = len(setting_values)
        threshold_count = int(setting_values[3])

        # compute the index at which the save image settings count is stored 
        # 4 fixed settings + <n settings for threshold> + 12 fixed settings
        fixed_settings_set_1 = (
            self.images_list,
            self.thr,
            self.wants_channel_thresholds,
            self.thresholds_count

        )
        fixed_settings_set_2 = (
            self.wants_threshold_visualization,
            self.threshold_visualization_list,
            self.images_or_objects,
            self.objects_list,
            self.do_all,
            self.do_corr_and_slope,
            self.do_manders,
            self.do_rwc,
            self.do_overlap,
            self.do_costes,
            self.fast_costes,
            self.wants_masks_saved,
        )
        save_image_settings_count_idx = len(fixed_settings_set_1) + (threshold_count * THRESHOLD_SETTING_COUNT) + len(fixed_settings_set_2)


        save_image_count = int(setting_values[save_image_settings_count_idx])
        assert (
            (value_count - FIXED_SETTING_COUNT)  
            - (THRESHOLD_SETTING_COUNT * threshold_count) 
            - (SAVE_MASK_SETTING_COUNT * save_image_count) 
            == 0
            )
        del self.thresholds_list[threshold_count:]
        while len(self.thresholds_list) < threshold_count:
            self.add_threshold(removable=True)
        del self.save_mask_list[save_image_count:]
        while len(self.save_mask_list) < save_image_count:
            self.add_save_mask(removable=True)

    def get_image_pairs(self):
        """Yield all permutations of pairs of images to correlate

        Yields the pairs of images in a canonical order.
        """
        for i in range(len(self.images_list.value) - 1):
            for j in range(i + 1, len(self.images_list.value)):
                yield (
                    self.images_list.value[i],
                    self.images_list.value[j],
                )

    def wants_images(self):
        """True if the user wants to measure correlation on whole images"""
        return self.images_or_objects in (M_IMAGES, M_IMAGES_AND_OBJECTS)

    def wants_objects(self):
        """True if the user wants to measure per-object correlations"""
        return self.images_or_objects in (M_OBJECTS, M_IMAGES_AND_OBJECTS)
    
    def verify_image_dims(self, workspace, image_name1, image_name2):
        """Verify that the images have the same dimensions and return the dimensions"""
        image1_dims = workspace.image_set.get_image(image_name1).dimensions
        image2_dims = workspace.image_set.get_image(image_name2).dimensions
        if image1_dims != image2_dims:
            raise ValidationError(
                f"Image dimensions do not match for {image_name1}({image1_dims}) and {image_name2}({image2_dims}). ",
                self.images_list
            )
        return image1_dims
    def prepare_images(self, workspace, first_image_name, second_image_name):
        first_image = workspace.image_set.get_image(first_image_name, must_be_grayscale=True)
        second_image = workspace.image_set.get_image(second_image_name, must_be_grayscale=True)
        
        first_pixel_count = numpy.prod(first_image.pixel_data.shape)
        second_pixel_count = numpy.prod(second_image.pixel_data.shape)
        
        first_mask = first_image.mask
        second_mask = second_image.mask
        
        first_pixel_data = first_image.pixel_data
        second_pixel_data = second_image.pixel_data
        if first_pixel_count < second_pixel_count:
            second_pixel_data = first_image.crop_image_similarly(second_image.pixel_data)
            second_mask = first_image.crop_image_similarly(second_image.mask)
        elif second_pixel_count < first_pixel_count:
            first_pixel_data = second_image.crop_image_similarly(first_image.pixel_data)
            first_mask = second_image.crop_image_similarly(first_image.mask)
        mask = (
            first_mask
            & second_mask
            & (~numpy.isnan(first_pixel_data))
            & (~numpy.isnan(second_pixel_data))
        )
        return first_pixel_data, second_pixel_data, mask

    def prepare_images_objects(self, workspace, first_image_name, second_image_name, object_name):
        first_image = workspace.image_set.get_image(first_image_name, must_be_grayscale=True)
        second_image = workspace.image_set.get_image(second_image_name, must_be_grayscale=True)
        objects = workspace.object_set.get_objects(object_name)
        labels = objects.segmented
        object_count = objects.count
        try:
            first_pixels = objects.crop_image_similarly(first_image.pixel_data)
            first_mask = objects.crop_image_similarly(first_image.mask)
        except ValueError:
            first_pixels, m1 = size_similarly(labels, first_image.pixel_data)
            first_mask, m1 = size_similarly(labels, first_image.mask)
            first_mask[~m1] = False
        try:
            second_pixels = objects.crop_image_similarly(second_image.pixel_data)
            second_mask = objects.crop_image_similarly(second_image.mask)
        except ValueError:
            second_pixels, m1 = size_similarly(labels, second_image.pixel_data)
            second_mask, m1 = size_similarly(labels, second_image.mask)
            second_mask[~m1] = False
        mask = (labels > 0) & first_mask & second_mask
        first_pixels = first_pixels[mask]
        second_pixels = second_pixels[mask]
        labels = labels[mask]
        first_pixel_data = first_image.pixel_data
        first_mask = first_image.mask
        first_pixel_count = numpy.prod(first_image.pixel_data.shape)
        second_pixel_data = second_image.pixel_data
        second_mask = second_image.mask
        second_pixel_count = numpy.prod(second_image.pixel_data.shape)
        #
        # Crop the larger image similarly to the smaller one
        #
        if first_pixel_count < second_pixel_count:
            second_pixel_data = first_image.crop_image_similarly(second_pixel_data)
            second_mask = first_image.crop_image_similarly(second_mask)
        elif second_pixel_count < first_pixel_count:
            first_pixel_data = second_image.crop_image_similarly(first_pixel_data)
            first_mask = second_image.crop_image_similarly(first_mask)
        mask = (
            first_mask
            & second_mask
            & (~numpy.isnan(first_pixel_data))
            & (~numpy.isnan(second_pixel_data))
        )
        return 
    
    def run(self, workspace):
        """Calculate measurements on an image set"""
        col_labels = ["First image", "Second image", "Objects", "Measurement", "Value"]
        statistics = []
        image_dims = None
        if len(self.images_list.value) < 2:
            raise ValueError("At least 2 images must be selected for analysis.")
        for first_image_name, second_image_name in self.get_image_pairs():
            image_dims = self.verify_image_dims(workspace, first_image_name, second_image_name)

            first_pixel_data, second_pixel_data, mask = self.prepare_images(workspace, first_image_name, second_image_name)
            kwargs = {}
            if self.wants_images():
                first_threshold_value = self.get_image_threshold_value(first_image_name)
                second_threshold_value = self.get_image_threshold_value(second_image_name)
                measurement_types = []
                if self.do_corr_and_slope:
                    measurement_types.append(MeasurementType.CORRELATION)
                if self.do_manders:
                    measurement_types.append(MeasurementType.MANDERS)
                if self.do_rwc:
                    measurement_types.append(MeasurementType.RWC)
                if self.do_overlap:
                    measurement_types.append(MeasurementType.OVERLAP)
                if self.do_costes:
                    measurement_types.append(MeasurementType.COSTES)
                    kwargs["costes_method"] = self.fast_costes.value
                    kwargs["first_image_scale"] = workspace.image_set.get_image(first_image_name).scale
                    kwargs["second_image_scale"] = workspace.image_set.get_image(second_image_name).scale
                measurements_result, colocalization_measurements = run_image_pair_images(
                    first_pixel_data, second_pixel_data, first_image_name, second_image_name, mask, first_threshold_value, second_threshold_value, measurement_types, **kwargs
                )
                statistics += measurements_result
                for measurement_name, measurement_value in colocalization_measurements.items():
                    workspace.measurements.add_image_measurement(measurement_name, measurement_value)

            if self.wants_objects():
                
                for object_name in self.objects_list.value:
                    """Calculate per-object correlations between intensities in two images"""
                    first_image = workspace.image_set.get_image(
                        first_image_name, must_be_grayscale=True
                    )
                    second_image = workspace.image_set.get_image(
                        second_image_name, must_be_grayscale=True
                    )
                    objects = workspace.object_set.get_objects(object_name)
                    #
                    # Crop both images to the size of the labels matrix
                    #
                    labels = objects.segmented
                    object_count = objects.count
                    try:
                        first_pixels = objects.crop_image_similarly(first_image.pixel_data)
                        first_mask = objects.crop_image_similarly(first_image.mask)
                    except ValueError:
                        first_pixels, m1 = size_similarly(labels, first_image.pixel_data)
                        first_mask, m1 = size_similarly(labels, first_image.mask)
                        first_mask[~m1] = False
                    try:
                        second_pixels = objects.crop_image_similarly(second_image.pixel_data)
                        second_mask = objects.crop_image_similarly(second_image.mask)
                    except ValueError:
                        second_pixels, m1 = size_similarly(labels, second_image.pixel_data)
                        second_mask, m1 = size_similarly(labels, second_image.mask)
                        second_mask[~m1] = False
                    mask = (labels > 0) & first_mask & second_mask
                    first_pixels = first_pixels[mask]
                    second_pixels = second_pixels[mask]
                    labels = labels[mask]
                    result = []
                    first_pixel_data = first_image.pixel_data
                    first_mask = first_image.mask
                    first_pixel_count = numpy.prod(first_pixel_data.shape)
                    second_pixel_data = second_image.pixel_data
                    second_mask = second_image.mask
                    second_pixel_count = numpy.prod(second_pixel_data.shape)
                    #
                    # Crop the larger image similarly to the smaller one
                    #
                    if first_pixel_count < second_pixel_count:
                        second_pixel_data = first_image.crop_image_similarly(second_pixel_data)
                        second_mask = first_image.crop_image_similarly(second_mask)
                    elif second_pixel_count < first_pixel_count:
                        first_pixel_data = second_image.crop_image_similarly(first_pixel_data)
                        first_mask = second_image.crop_image_similarly(first_mask)
                    mask = (
                        first_mask
                        & second_mask
                        & (~numpy.isnan(first_pixel_data))
                        & (~numpy.isnan(second_pixel_data))
                    )
                    first_threshold_value = self.get_image_threshold_value(first_image_name)
                    second_threshold_value = self.get_image_threshold_value(second_image_name)
                    measurement_types = []
                    if self.do_corr_and_slope:
                        measurement_types.append(MeasurementType.CORRELATION)
                    if self.do_manders:
                        measurement_types.append(MeasurementType.MANDERS)
                    if self.do_rwc:
                        measurement_types.append(MeasurementType.RWC)
                    if self.do_overlap:
                        measurement_types.append(MeasurementType.OVERLAP)
                    if self.do_costes:
                        measurement_types.append(MeasurementType.COSTES)
                        kwargs["costes_method"] = self.fast_costes.value
                        kwargs["first_image_scale"] = workspace.image_set.get_image(first_image_name).scale
                        kwargs["second_image_scale"] = workspace.image_set.get_image(second_image_name).scale
                    measurements_result, colocalization_measurements = run_image_pair_objects(
                        first_pixel_data, second_pixel_data, first_pixels, second_pixels, labels, object_count, first_image_name, second_image_name, object_name, mask, first_threshold_value, second_threshold_value, measurement_types, **kwargs
                    )
                    statistics += measurements_result
                    for measurement_name, measurement_value in colocalization_measurements.items():
                        workspace.measurements.add_measurement(object_name, measurement_name, measurement_value)

        if self.wants_masks_saved.value:
            self.save_requested_masks(workspace)
        if self.show_window:
            workspace.display_data.statistics = statistics
            workspace.display_data.col_labels = col_labels
            workspace.display_data.dimensions = image_dims

    def display(self, workspace, figure):
        statistics = workspace.display_data.statistics
        num_image_rows = 1 # for the original images
        num_image_cols = 2 # for the results table + padding before the results table to prevent overlap
        # For each image, create a new column and for each object, create a new row of subplot
        if self.wants_threshold_visualization.value and self.threshold_visualization_list.value:
            num_image_cols += len(self.threshold_visualization_list.value)
            if self.wants_objects():
                num_image_rows += len(self.objects_list.value)
            if self.wants_images():
                num_image_rows += 1
            figure.set_subplots((num_image_cols, num_image_rows))
            # set subplot dimensions to enable 3d visualization
            figure.set_subplots(
                dimensions=workspace.display_data.dimensions,
                subplots=(num_image_cols, num_image_rows)
            )
            self.show_threshold_visualization(figure, workspace)
        else:
            num_image_cols -= 1
            figure.set_subplots((1, 1))
            
        figure.subplot_table(
            num_image_cols-1, 0, statistics, workspace.display_data.col_labels, title='', n_cols=1, n_rows=num_image_rows
        )

    def show_threshold_visualization(self, figure, workspace):
        """
        Visualize the thresholded images.
        Assumptions:
        - Image mask is used to determine the pixels to be thresholded
        - Mask generated after thresholding is visualized
        - When object correlation is selected, all objects selected are visualized
        - All images are shown on the same subplot
        """
        if not self.wants_threshold_visualization.value:
            return
        for idx, image_name in enumerate(self.threshold_visualization_list.value):
            plotting_row = 0
            image = workspace.image_set.get_image(image_name, must_be_grayscale=True)
            # Plot original
            figure.subplot_imshow_grayscale(
                idx,
                plotting_row,
                image.pixel_data,
                title = image_name + " (Original)",
                sharexy=figure.subplot(0, 0)
            )
            plotting_row += 1

            # Thresholding code used from run_image_pair_images() and run_image_pair_objects()
            threshold_value = self.get_image_threshold_value(image_name)
            if self.wants_images():
                thr_i_out = self.get_thresholded_mask(image_name, workspace)
                figure.subplot_imshow_grayscale(
                    idx,
                    plotting_row, 
                    thr_i_out,
                    title = image_name + f" (Threshold = {threshold_value})",
                    sharexy=figure.subplot(0, 0)
                    )
                
                plotting_row += 1
            if self.wants_objects():
                for object_name in self.objects_list.value:
                    threshold_mask_image = self.get_thresholded_mask(image_name, workspace, object_name=object_name)
                    figure.subplot_imshow_grayscale(
                        idx,
                        plotting_row,
                        threshold_mask_image,
                        title=image_name  + f" ({object_name}), (Threshold: {threshold_value})",
                        sharexy=figure.subplot(0, 0)
                    )
                    plotting_row += 1

    def get_thresholded_mask(self, image_name, workspace, object_name=None):

        image = workspace.image_set.get_image(image_name, must_be_grayscale=True)
        t_val = self.get_image_threshold_value(image_name)
        # Thresholding code used from run_image_pair_images() and run_image_pair_objects()
        image_pixel_data = image.pixel_data
        image_mask = image.mask
        image_mask = image_mask & (~numpy.isnan(image_pixel_data))
        output_image_arr = numpy.zeros_like(image_pixel_data)
        if object_name is None:
            # perform on the entire image
            
            if numpy.any(image_mask):
                thr_i = get_global_threshold(image_pixel_data, None, Threshold.Method.MAX_INTENSITY_PERCENTAGE, max_intensity_percentage=t_val)
                output_image_arr, _ = apply_threshold(image_pixel_data, thr_i)
        else:
            # perform on the object
            objects = workspace.object_set.get_objects(object_name)
            labels = objects.segmented
            try:
                image_pixels = objects.crop_image_similarly(image_pixel_data)
                image_mask = objects.crop_image_similarly(image_mask)
            except ValueError:
                image_pixels, m1 = size_similarly(labels, image_pixel_data)
                image_mask, m1 = size_similarly(labels, image_mask)
                image_mask[~m1] = False
            output_image_arr = apply_threshold_to_objects(image_pixels, labels, t_val, image_mask)  

        return output_image_arr

    def save_requested_masks(self, workspace):
        # Iterate over the list of save masks
        for save_mask in self.save_mask_list:
            image_name = save_mask.image_name.value
            object_name = save_mask.choose_object.value if save_mask.save_mask_wants_objects.value else None
            save_image_name = save_mask.save_image_name.value
            original_image = workspace.image_set.get_image(image_name, must_be_grayscale=True)
            
            # Call the relevant funcitons to get the thresholded masks
            t_val = self.get_image_threshold_value(image_name)
            output_image = Image(self.get_thresholded_mask(image_name, workspace, object_name), parent_image=original_image)

            # Save the mask to the image set
            workspace.image_set.add(save_image_name, output_image)
            

    def get_image_threshold_value(self, image_name):
        if self.wants_channel_thresholds.value:
            for threshold in self.thresholds_list:
                if threshold.image_name == image_name:
                    return threshold.threshold_for_channel.value
        return self.thr.value

    def get_measurement_columns(self, pipeline):
        """Return column definitions for all measurements made by this module"""
        columns = []
        for first_image, second_image in self.get_image_pairs():
            if self.wants_images():
                if self.do_corr_and_slope:
                    columns += [
                        (
                            "Image",
                            F_CORRELATION_FORMAT % (first_image, second_image),
                            COLTYPE_FLOAT,
                        ),
                        (
                            "Image",
                            F_SLOPE_FORMAT % (first_image, second_image),
                            COLTYPE_FLOAT,
                        ),
                    ]
                if self.do_overlap:
                    columns += [
                        (
                            "Image",
                            F_OVERLAP_FORMAT % (first_image, second_image),
                            COLTYPE_FLOAT,
                        ),
                        (
                            "Image",
                            F_K_FORMAT % (first_image, second_image),
                            COLTYPE_FLOAT,
                        ),
                        (
                            "Image",
                            F_K_FORMAT % (second_image, first_image),
                            COLTYPE_FLOAT,
                        ),
                    ]
                if self.do_manders:
                    columns += [
                        (
                            "Image",
                            F_MANDERS_FORMAT % (first_image, second_image),
                            COLTYPE_FLOAT,
                        ),
                        (
                            "Image",
                            F_MANDERS_FORMAT % (second_image, first_image),
                            COLTYPE_FLOAT,
                        ),
                    ]

                if self.do_rwc:
                    columns += [
                        (
                            "Image",
                            F_RWC_FORMAT % (first_image, second_image),
                            COLTYPE_FLOAT,
                        ),
                        (
                            "Image",
                            F_RWC_FORMAT % (second_image, first_image),
                            COLTYPE_FLOAT,
                        ),
                    ]
                if self.do_costes:
                    columns += [
                        (
                            "Image",
                            F_COSTES_FORMAT % (first_image, second_image),
                            COLTYPE_FLOAT,
                        ),
                        (
                            "Image",
                            F_COSTES_FORMAT % (second_image, first_image),
                            COLTYPE_FLOAT,
                        ),
                    ]

            if self.wants_objects():
                for i in range(len(self.objects_list.value)):
                    object_name = self.objects_list.value[i]
                    if self.do_corr_and_slope:
                        columns += [
                            (
                                object_name,
                                F_CORRELATION_FORMAT % (first_image, second_image),
                                COLTYPE_FLOAT,
                            )
                        ]
                    if self.do_overlap:
                        columns += [
                            (
                                object_name,
                                F_OVERLAP_FORMAT % (first_image, second_image),
                                COLTYPE_FLOAT,
                            ),
                            (
                                object_name,
                                F_K_FORMAT % (first_image, second_image),
                                COLTYPE_FLOAT,
                            ),
                            (
                                object_name,
                                F_K_FORMAT % (second_image, first_image),
                                COLTYPE_FLOAT,
                            ),
                        ]
                    if self.do_manders:
                        columns += [
                            (
                                object_name,
                                F_MANDERS_FORMAT % (first_image, second_image),
                                COLTYPE_FLOAT,
                            ),
                            (
                                object_name,
                                F_MANDERS_FORMAT % (second_image, first_image),
                                COLTYPE_FLOAT,
                            ),
                        ]
                    if self.do_rwc:
                        columns += [
                            (
                                object_name,
                                F_RWC_FORMAT % (first_image, second_image),
                                COLTYPE_FLOAT,
                            ),
                            (
                                object_name,
                                F_RWC_FORMAT % (second_image, first_image),
                                COLTYPE_FLOAT,
                            ),
                        ]
                    if self.do_costes:
                        columns += [
                            (
                                object_name,
                                F_COSTES_FORMAT % (first_image, second_image),
                                COLTYPE_FLOAT,
                            ),
                            (
                                object_name,
                                F_COSTES_FORMAT % (second_image, first_image),
                                COLTYPE_FLOAT,
                            ),
                        ]
        return columns

    def get_categories(self, pipeline, object_name):
        """Return the categories supported by this module for the given object

        object_name - name of the measured object or IMAGE
        """
        if (object_name == "Image" and self.wants_images()) or (
            (object_name != "Image")
            and self.wants_objects()
            and (object_name in self.objects_list.value)
        ):
            return ["Correlation"]
        return []

    def get_measurements(self, pipeline, object_name, category):
        if self.get_categories(pipeline, object_name) == [category]:
            results = []
            if self.do_corr_and_slope:
                if object_name == "Image":
                    results += ["Correlation", "Slope"]
                else:
                    results += ["Correlation"]
            if self.do_overlap:
                results += ["Overlap", "K"]
            if self.do_manders:
                results += ["Manders"]
            if self.do_rwc:
                results += ["RWC"]
            if self.do_costes:
                results += ["Costes"]
            return results
        return []

    def get_measurement_images(self, pipeline, object_name, category, measurement):
        """Return the joined pairs of images measured"""
        result = []
        if measurement in self.get_measurements(pipeline, object_name, category):
            for i1, i2 in self.get_image_pairs():
                result.append("%s_%s" % (i1, i2))
                # For asymmetric, return both orderings
                if measurement in ("K", "Manders", "RWC", "Costes"):
                    result.append("%s_%s" % (i2, i1))
        return result

    def validate_module(self, pipeline):
        """Make sure chosen objects are selected only once"""
        if len(self.images_list.value) < 2:
            raise ValidationError("This module needs at least 2 images to be selected", self.images_list)

        if self.wants_objects():
            if len(self.objects_list.value) == 0:
                raise ValidationError("No object sets selected", self.objects_list)
            
        # Raise validation error if threshold is set twice
        thresholds_list_image_names = [i.image_name.value for i in self.thresholds_list]
        if len(thresholds_list_image_names) != len(set(thresholds_list_image_names)):
            raise ValidationError("Thresholds are set for the same image more than once", thresholds_list_image_names)

    def upgrade_settings(self, setting_values, variable_revision_number, module_name):
        """Adjust the setting values for pipelines saved under old revisions"""
        if variable_revision_number < 2:
            raise NotImplementedError(
                "Automatic upgrade for this module is not supported in CellProfiler 3."
            )

        if variable_revision_number == 2:
            image_count = int(setting_values[0])
            idx_thr = image_count + 2
            setting_values = (
                setting_values[:idx_thr] + ["15.0"] + setting_values[idx_thr:]
            )
            variable_revision_number = 3

        if variable_revision_number == 3:
            num_images = int(setting_values[0])
            num_objects = int(setting_values[1])
            div_img = 2 + num_images
            div_obj = div_img + 2 + num_objects
            images_set = set(setting_values[2:div_img])
            thr_mode = setting_values[div_img : div_img + 2]
            objects_set = set(setting_values[div_img + 2 : div_obj])
            other_settings = setting_values[div_obj:]
            if "None" in images_set:
                images_set.remove("None")
            if "None" in objects_set:
                objects_set.remove("None")
            images_string = ", ".join(map(str, images_set))
            objects_string = ", ".join(map(str, objects_set))
            setting_values = (
                [images_string] + thr_mode + [objects_string] + other_settings
            )
            variable_revision_number = 4
        if variable_revision_number == 4:
            # Add costes mode switch
            setting_values += [M_FASTER]
            variable_revision_number = 5

        if variable_revision_number == 5:
            # Settings values returned by upgrade_settings() should match the setting values in settings()
            # Version upgrade from 4 --> 5 does not apply this rule so it is fixed here:
            
            # To determine if the upgrade is needed, check the total number of settings
            if len(setting_values) == 5:
                # Assumption: `run_all` is set to "Yes" by default
                setting_values = setting_values[:-1] + ['Yes']*6 + setting_values[-1:]

            if len(setting_values) != 11:
                raise Warning(f"The Measure Colocalization module contains an invalid number of settings. Please check the module configuration and save a new pipeline. ")
            
            """
            add 'No' for custom thresholds and '0' for custom threshold counts
            """
            setting_values = setting_values[:2] + ['No', '0', 'No', ''] + setting_values[2:] + ['No', '0']
            
            variable_revision_number = 6

        return setting_values, variable_revision_number

    def volumetric(self):
        return True

def get_scale(scale_1, scale_2):
    if scale_1 is not None and scale_2 is not None:
        return max(scale_1, scale_2)
    elif scale_1 is not None:
        return scale_1
    elif scale_2 is not None:
        return scale_2
    else:
        return 255

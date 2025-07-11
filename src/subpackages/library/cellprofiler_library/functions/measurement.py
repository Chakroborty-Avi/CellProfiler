import numpy as np
import scipy
import centrosome
import centrosome.cpmorphology
import centrosome.filter
import centrosome.propagate
import centrosome.fastemd
from sklearn.cluster import KMeans
from typing import Tuple
import numpy
import skimage

from cellprofiler_library.opts import measureimageoverlap as mio
from cellprofiler_library.functions.segmentation import convert_labels_to_ijv
from cellprofiler_library.functions.segmentation import indices_from_ijv
from cellprofiler_library.functions.segmentation import count_from_ijv
from cellprofiler_library.functions.segmentation import areas_from_ijv
from cellprofiler_library.functions.segmentation import cast_labels_to_label_set

from cellprofiler_library.opts.objectsizeshapefeatures import ObjectSizeShapeFeatures
from scipy.linalg import lstsq
from cellprofiler_library.types import Image2DGrayscale, ObjectLabelsDense
from typing import Optional, Tuple, List, Sequence
from cellprofiler_library.opts.measurecolocalization import CostesMethod
from centrosome.cpmorphology import fixup_scipy_ndimage_result as fix
import scipy.ndimage
from numpy.typing import NDArray



def measure_image_overlap_statistics(
    ground_truth_image,
    test_image,
    mask=None,
):
    # Check that the inputs are binary
    if not np.array_equal(ground_truth_image, ground_truth_image.astype(bool)):
        raise ValueError("ground_truth_image is not a binary image")
    
    if not np.array_equal(test_image, test_image.astype(bool)):
        raise ValueError("test_image is not a binary image")

    if mask is None:
        mask = np.ones_like(ground_truth_image, bool)

    orig_shape = ground_truth_image.shape
    
    # Covert 3D image to 2D long
    if ground_truth_image.ndim > 2:
        
        ground_truth_image = ground_truth_image.reshape(
            -1, ground_truth_image.shape[-1]
        )
        test_image = test_image.reshape(-1, test_image.shape[-1])

        mask = mask.reshape(-1, mask.shape[-1])

    false_positives = test_image & ~ground_truth_image

    false_positives[~mask] = False

    false_negatives = (~test_image) & ground_truth_image

    false_negatives[~mask] = False

    true_positives = test_image & ground_truth_image

    true_positives[~mask] = False

    true_negatives = (~test_image) & (~ground_truth_image)

    true_negatives[~mask] = False

    false_positive_count = np.sum(false_positives)

    true_positive_count = np.sum(true_positives)

    false_negative_count = np.sum(false_negatives)

    true_negative_count = np.sum(true_negatives)

    labeled_pixel_count = true_positive_count + false_positive_count

    true_count = true_positive_count + false_negative_count

    if labeled_pixel_count == 0:
        precision = 1.0
    else:
        precision = float(true_positive_count) / float(labeled_pixel_count)

    if true_count == 0:
        recall = 1.0
    else:
        recall = float(true_positive_count) / float(true_count)

    if (precision + recall) == 0:
        f_factor = 0.0  # From http://en.wikipedia.org/wiki/F1_score
    else:
        f_factor = 2.0 * precision * recall / (precision + recall)

    negative_count = false_positive_count + true_negative_count

    if negative_count == 0:
        false_positive_rate = 0.0

        true_negative_rate = 1.0
    else:
        false_positive_rate = float(false_positive_count) / float(negative_count)

        true_negative_rate = float(true_negative_count) / float(negative_count)
    if true_count == 0:
        false_negative_rate = 0.0

        true_positive_rate = 1.0
    else:
        false_negative_rate = float(false_negative_count) / float(true_count)

        true_positive_rate = float(true_positive_count) / float(true_count)

    ground_truth_labels, ground_truth_count = scipy.ndimage.label(
        ground_truth_image & mask, np.ones((3, 3), bool)
    )

    test_labels, test_count = scipy.ndimage.label(
        test_image & mask, np.ones((3, 3), bool)
    )

    rand_index, adjusted_rand_index = compute_rand_index(
        test_labels, ground_truth_labels, mask
    )

    data = {
        "true_positives": true_positives.reshape(orig_shape),
        "true_negatives": true_negatives.reshape(orig_shape),
        "false_positives": false_positives.reshape(orig_shape),
        "false_negatives": false_negatives.reshape(orig_shape),
        "Ffactor": f_factor,
        "Precision": precision,
        "Recall": recall,
        "TruePosRate": true_positive_rate,
        "FalsePosRate": false_positive_rate,
        "FalseNegRate": false_negative_rate,
        "TrueNegRate": true_negative_rate,
        "RandIndex": rand_index,
        "AdjustedRandIndex": adjusted_rand_index,
    }

    return data


def compute_rand_index(test_labels, ground_truth_labels, mask):
    """Calculate the Rand Index

    http://en.wikipedia.org/wiki/Rand_index

    Given a set of N elements and two partitions of that set, X and Y

    A = the number of pairs of elements in S that are in the same set in
        X and in the same set in Y
    B = the number of pairs of elements in S that are in different sets
        in X and different sets in Y
    C = the number of pairs of elements in S that are in the same set in
        X and different sets in Y
    D = the number of pairs of elements in S that are in different sets
        in X and the same set in Y

    The rand index is:   A + B
                            -----
                        A+B+C+D


    The adjusted rand index is the rand index adjusted for chance
    so as not to penalize situations with many segmentations.

    Jorge M. Santos, Mark Embrechts, "On the Use of the Adjusted Rand
    Index as a Metric for Evaluating Supervised Classification",
    Lecture Notes in Computer Science,
    Springer, Vol. 5769, pp. 175-184, 2009. Eqn # 6

    ExpectedIndex = best possible score

    ExpectedIndex = sum(N_i choose 2) * sum(N_j choose 2)

    MaxIndex = worst possible score = 1/2 (sum(N_i choose 2) + sum(N_j choose 2)) * total

    A * total - ExpectedIndex
    -------------------------
    MaxIndex - ExpectedIndex

    returns a tuple of the Rand Index and the adjusted Rand Index
    """
    ground_truth_labels = ground_truth_labels[mask].astype(np.uint32)
    test_labels = test_labels[mask].astype(np.uint32)
    if len(test_labels) > 0:
        #
        # Create a sparse matrix of the pixel labels in each of the sets
        #
        # The matrix, N(i,j) gives the counts of all of the pixels that were
        # labeled with label I in the ground truth and label J in the
        # test set.
        #
        N_ij = scipy.sparse.coo_matrix(
            (np.ones(len(test_labels)), (ground_truth_labels, test_labels))
        ).toarray()

        def choose2(x):
            """Compute # of pairs of x things = x * (x-1) / 2"""
            return x * (x - 1) / 2

        #
        # Each cell in the matrix is a count of a grouping of pixels whose
        # pixel pairs are in the same set in both groups. The number of
        # pixel pairs is n * (n - 1), so A = sum(matrix * (matrix - 1))
        #
        A = np.sum(choose2(N_ij))
        #
        # B is the sum of pixels that were classified differently by both
        # sets. But the easier calculation is to find A, C and D and get
        # B by subtracting A, C and D from the N * (N - 1), the total
        # number of pairs.
        #
        # For C, we take the number of pixels classified as "i" and for each
        # "j", subtract N(i,j) from N(i) to get the number of pixels in
        # N(i,j) that are in some other set = (N(i) - N(i,j)) * N(i,j)
        #
        # We do the similar calculation for D
        #
        N_i = np.sum(N_ij, 1)
        N_j = np.sum(N_ij, 0)
        C = np.sum((N_i[:, np.newaxis] - N_ij) * N_ij) / 2
        D = np.sum((N_j[np.newaxis, :] - N_ij) * N_ij) / 2
        total = choose2(len(test_labels))
        # an astute observer would say, why bother computing A and B
        # when all we need is A+B and C, D and the total can be used to do
        # that. The calculations aren't too expensive, though, so I do them.
        B = total - A - C - D
        rand_index = (A + B) / total
        #
        # Compute adjusted Rand Index
        #
        expected_index = np.sum(choose2(N_i)) * np.sum(choose2(N_j))
        max_index = (np.sum(choose2(N_i)) + np.sum(choose2(N_j))) * total / 2

        adjusted_rand_index = (A * total - expected_index) / (
            max_index - expected_index
        )
    else:
        rand_index = adjusted_rand_index = np.nan
    return rand_index, adjusted_rand_index


def compute_earth_movers_distance(
    ground_truth_image,
    test_image,
    mask=None,
    decimation_method: mio.DM = mio.DM.KMEANS,
    max_distance: int = 250,
    max_points: int = 250,
    penalize_missing: bool = False,
):
    """Compute the earthmovers distance between two sets of objects

    src_objects - move pixels from these objects

    dest_objects - move pixels to these objects

    returns the earth mover's distance
    """

    # Check that the inputs are binary
    if not np.array_equal(ground_truth_image, ground_truth_image.astype(bool)):
        raise ValueError("ground_truth_image is not a binary image")
    
    if not np.array_equal(test_image, test_image.astype(bool)):
        raise ValueError("test_image is not a binary image")

    if mask is None:
        mask = np.ones_like(ground_truth_image, bool)

    # Covert 3D image to 2D long
    if ground_truth_image.ndim > 2:
        ground_truth_image = ground_truth_image.reshape(
            -1, ground_truth_image.shape[-1]
        )

        test_image = test_image.reshape(-1, test_image.shape[-1])

        mask = mask.reshape(-1, mask.shape[-1])

    # ground truth labels
    dest_labels = scipy.ndimage.label(
        ground_truth_image & mask, np.ones((3, 3), bool)
    )[0]
    dest_labelset = cast_labels_to_label_set(dest_labels)
    dest_ijv = convert_labels_to_ijv(dest_labels, validate=False)
    dest_ijv_indices = indices_from_ijv(dest_ijv, validate=False)
    dest_count = count_from_ijv(
        dest_ijv, indices=dest_ijv_indices, validate=False)
    dest_areas = areas_from_ijv(
        dest_ijv, indices=dest_ijv_indices, validate=False)

    # test labels
    src_labels = scipy.ndimage.label(
        test_image & mask, np.ones((3, 3), bool)
    )[0]
    src_labelset = cast_labels_to_label_set(src_labels)
    src_ijv = convert_labels_to_ijv(src_labels, validate=False)
    src_ijv_indices = indices_from_ijv(src_ijv, validate=False)
    src_count = count_from_ijv(
        src_ijv, indices=src_ijv_indices, validate=False)
    src_areas = areas_from_ijv(
        src_ijv, indices=src_ijv_indices, validate=False)

    #
    # if either foreground set is empty, the emd is the penalty.
    #
    for lef_count, right_areas in (
        (src_count, dest_areas),
        (dest_count, src_areas),
    ):
        if lef_count == 0:
            if penalize_missing:
                return np.sum(right_areas) * max_distance
            else:
                return 0
    if decimation_method == mio.DM.KMEANS:
        isrc, jsrc = get_kmeans_points(src_ijv, dest_ijv, max_points)
        idest, jdest = isrc, jsrc
    elif decimation_method == mio.DM.SKELETON:
        isrc, jsrc = get_skeleton_points(src_labelset, src_labels.shape, max_points)
        idest, jdest = get_skeleton_points(dest_labelset, dest_labels.shape, max_points)
    else:
        raise TypeError("Unknown type for decimation method: %s" % decimation_method)
    src_weights, dest_weights = [
        get_weights(i, j, get_labels_mask(labelset, shape))
        for i, j, labelset, shape in (
            (isrc, jsrc, src_labelset, src_labels.shape),
            (idest, jdest, dest_labelset, dest_labels.shape),
        )
    ]
    ioff, joff = [
        src[:, np.newaxis] - dest[np.newaxis, :]
        for src, dest in ((isrc, idest), (jsrc, jdest))
    ]
    c = np.sqrt(ioff * ioff + joff * joff).astype(np.int32)
    c[c > max_distance] = max_distance
    extra_mass_penalty = max_distance if penalize_missing else 0

    emd = centrosome.fastemd.emd_hat_int32(
        src_weights.astype(np.int32),
        dest_weights.astype(np.int32),
        c,
        extra_mass_penalty=extra_mass_penalty,
    )
    return emd


def get_labels_mask(labelset, shape):
    labels_mask = np.zeros(shape, bool)
    for labels, indexes in labelset:
        labels_mask = labels_mask | labels > 0
    return labels_mask


def get_skeleton_points(labelset, shape, max_points):
    """Get points by skeletonizing the objects and decimating"""
    total_skel = np.zeros(shape, bool)

    for labels, indexes in labelset:
        colors = centrosome.cpmorphology.color_labels(labels)
        for color in range(1, np.max(colors) + 1):
            labels_mask = colors == color
            skel = centrosome.cpmorphology.skeletonize(
                labels_mask,
                ordering=scipy.ndimage.distance_transform_edt(labels_mask)
                * centrosome.filter.poisson_equation(labels_mask),
            )
            total_skel = total_skel | skel

    n_pts = np.sum(total_skel)

    if n_pts == 0:
        return np.zeros(0, np.int32), np.zeros(0, np.int32)

    i, j = np.where(total_skel)

    if n_pts > max_points:
        #
        # Decimate the skeleton by finding the branchpoints in the
        # skeleton and propagating from those.
        #
        markers = np.zeros(total_skel.shape, np.int32)
        branchpoints = centrosome.cpmorphology.branchpoints(
            total_skel
        ) | centrosome.cpmorphology.endpoints(total_skel)
        markers[branchpoints] = np.arange(np.sum(branchpoints)) + 1
        #
        # We compute the propagation distance to that point, then impose
        # a slightly arbitrary order to get an unambiguous ordering
        # which should number the pixels in a skeleton branch monotonically
        #
        ts_labels, distances = centrosome.propagate.propagate(
            np.zeros(markers.shape), markers, total_skel, 1
        )
        order = np.lexsort((j, i, distances[i, j], ts_labels[i, j]))
        #
        # Get a linear space of self.max_points elements with bounds at
        # 0 and len(order)-1 and use that to select the points.
        #
        order = order[np.linspace(0, len(order) - 1, max_points).astype(int)]
        return i[order], j[order]

    return i, j


def get_kmeans_points(src_ijv, dest_ijv, max_points):
    """Get representative points in the objects using K means

    src_ijv - get some of the foreground points from the source ijv labeling
    dest_ijv - get the rest of the foreground points from the ijv labeling
                objects

    returns a vector of i coordinates of representatives and a vector
            of j coordinates
    """

    ijv = np.vstack((src_ijv, dest_ijv))
    if len(ijv) <= max_points:
        return ijv[:, 0], ijv[:, 1]
    random_state = np.random.RandomState()
    random_state.seed(ijv.astype(int).flatten())
    kmeans = KMeans(n_clusters=max_points, tol=2, random_state=random_state)
    kmeans.fit(ijv[:, :2])
    return (
        kmeans.cluster_centers_[:, 0].astype(np.uint32),
        kmeans.cluster_centers_[:, 1].astype(np.uint32),
    )


def get_weights(i, j, labels_mask):
    """Return the weights to assign each i,j point

    Assign each pixel in the labels mask to the nearest i,j and return
    the number of pixels assigned to each i,j
    """
    #
    # Create a mapping of chosen points to their index in the i,j array
    #
    total_skel = np.zeros(labels_mask.shape, int)
    total_skel[i, j] = np.arange(1, len(i) + 1)
    #
    # Compute the distance from each chosen point to all others in image,
    # return the nearest point.
    #
    ii, jj = scipy.ndimage.distance_transform_edt(
        total_skel == 0, return_indices=True, return_distances=False
    )
    #
    # Filter out all unmasked points
    #
    ii, jj = [x[labels_mask] for x in (ii, jj)]
    if len(ii) == 0:
        return np.zeros(0, np.int32)
    #
    # Use total_skel to look up the indices of the chosen points and
    # bincount the indices.
    #
    result = np.zeros(len(i), np.int32)
    bc = np.bincount(total_skel[ii, jj])[1:]
    result[: len(bc)] = bc
    return result


def measure_object_size_shape(
    labels,
    desired_properties,
    calculate_zernikes: bool = True,
    calculate_advanced: bool = True,
    spacing: Tuple = None
):
    label_indices = numpy.unique(labels[labels != 0])
    nobjects = len(label_indices)
    
    if spacing is None:
        spacing = (1.0,) * labels.ndim

    if len(labels.shape) == 2:
        # 2D
        props = skimage.measure.regionprops_table(labels, properties=desired_properties)

        formfactor = 4.0 * numpy.pi * props["area"] / props["perimeter"] ** 2
        denom = [max(x, 1) for x in 4.0 * numpy.pi * props["area"]]
        compactness = props["perimeter"] ** 2 / denom

        max_radius = numpy.zeros(nobjects)
        median_radius = numpy.zeros(nobjects)
        mean_radius = numpy.zeros(nobjects)
        min_feret_diameter = numpy.zeros(nobjects)
        max_feret_diameter = numpy.zeros(nobjects)
        zernike_numbers = centrosome.zernike.get_zernike_indexes(ObjectSizeShapeFeatures.ZERNIKE_N.value + 1)

        zf = {}
        for n, m in zernike_numbers:
            zf[(n, m)] = numpy.zeros(nobjects)

        for index, mini_image in enumerate(props["image"]):
            # Pad image to assist distance tranform
            mini_image = numpy.pad(mini_image, 1)
            distances = scipy.ndimage.distance_transform_edt(mini_image)
            max_radius[index] = centrosome.cpmorphology.fixup_scipy_ndimage_result(
                scipy.ndimage.maximum(distances, mini_image)
            )
            mean_radius[index] = centrosome.cpmorphology.fixup_scipy_ndimage_result(
                scipy.ndimage.mean(distances, mini_image)
            )
            median_radius[index] = centrosome.cpmorphology.median_of_labels(
                distances, mini_image.astype("int"), [1]
            )

        #
        # Zernike features
        #
        if calculate_zernikes:
            zf_l = centrosome.zernike.zernike(zernike_numbers, labels, label_indices)
            for (n, m), z in zip(zernike_numbers, zf_l.transpose()):
                zf[(n, m)] = z

        if nobjects > 0:
            chulls, chull_counts = centrosome.cpmorphology.convex_hull(
                labels, label_indices
            )
            #
            # Feret diameter
            #
            (
                min_feret_diameter,
                max_feret_diameter,
            ) = centrosome.cpmorphology.feret_diameter(
                chulls, chull_counts, label_indices
            )

            features_to_record = {
                ObjectSizeShapeFeatures.F_AREA.value: props["area"],
                ObjectSizeShapeFeatures.F_PERIMETER.value: props["perimeter"],
                ObjectSizeShapeFeatures.F_MAJOR_AXIS_LENGTH.value: props["major_axis_length"],
                ObjectSizeShapeFeatures.F_MINOR_AXIS_LENGTH.value: props["minor_axis_length"],
                ObjectSizeShapeFeatures.F_ECCENTRICITY.value: props["eccentricity"],
                ObjectSizeShapeFeatures.F_ORIENTATION.value: props["orientation"] * (180 / numpy.pi),
                ObjectSizeShapeFeatures.F_CENTER_X.value: props["centroid-1"],
                ObjectSizeShapeFeatures.F_CENTER_Y.value: props["centroid-0"],
                ObjectSizeShapeFeatures.F_BBOX_AREA.value: props["bbox_area"],
                ObjectSizeShapeFeatures.F_MIN_X.value: props["bbox-1"],
                ObjectSizeShapeFeatures.F_MAX_X.value: props["bbox-3"],
                ObjectSizeShapeFeatures.F_MIN_Y.value: props["bbox-0"],
                ObjectSizeShapeFeatures.F_MAX_Y.value: props["bbox-2"],
                ObjectSizeShapeFeatures.F_FORM_FACTOR.value: formfactor,
                ObjectSizeShapeFeatures.F_EXTENT.value: props["extent"],
                ObjectSizeShapeFeatures.F_SOLIDITY.value: props["solidity"],
                ObjectSizeShapeFeatures.F_COMPACTNESS.value: compactness,
                ObjectSizeShapeFeatures.F_EULER_NUMBER.value: props["euler_number"],
                ObjectSizeShapeFeatures.F_MAXIMUM_RADIUS.value: max_radius,
                ObjectSizeShapeFeatures.F_MEAN_RADIUS.value: mean_radius,
                ObjectSizeShapeFeatures.F_MEDIAN_RADIUS.value: median_radius,
                ObjectSizeShapeFeatures.F_CONVEX_AREA.value: props["convex_area"],
                ObjectSizeShapeFeatures.F_MIN_FERET_DIAMETER.value: min_feret_diameter,
                ObjectSizeShapeFeatures.F_MAX_FERET_DIAMETER.value: max_feret_diameter,
                ObjectSizeShapeFeatures.F_EQUIVALENT_DIAMETER.value: props["equivalent_diameter"],
            }
            if calculate_advanced:
                features_to_record.update(
                    {
                        ObjectSizeShapeFeatures.F_SPATIAL_MOMENT_0_0.value: props["moments-0-0"],
                        ObjectSizeShapeFeatures.F_SPATIAL_MOMENT_0_1.value: props["moments-0-1"],
                        ObjectSizeShapeFeatures.F_SPATIAL_MOMENT_0_2.value: props["moments-0-2"],
                        ObjectSizeShapeFeatures.F_SPATIAL_MOMENT_0_3.value: props["moments-0-3"],
                        ObjectSizeShapeFeatures.F_SPATIAL_MOMENT_1_0.value: props["moments-1-0"],
                        ObjectSizeShapeFeatures.F_SPATIAL_MOMENT_1_1.value: props["moments-1-1"],
                        ObjectSizeShapeFeatures.F_SPATIAL_MOMENT_1_2.value: props["moments-1-2"],
                        ObjectSizeShapeFeatures.F_SPATIAL_MOMENT_1_3.value: props["moments-1-3"],
                        ObjectSizeShapeFeatures.F_SPATIAL_MOMENT_2_0.value: props["moments-2-0"],
                        ObjectSizeShapeFeatures.F_SPATIAL_MOMENT_2_1.value: props["moments-2-1"],
                        ObjectSizeShapeFeatures.F_SPATIAL_MOMENT_2_2.value: props["moments-2-2"],
                        ObjectSizeShapeFeatures.F_SPATIAL_MOMENT_2_3.value: props["moments-2-3"],
                        ObjectSizeShapeFeatures.F_CENTRAL_MOMENT_0_0.value: props["moments_central-0-0"],
                        ObjectSizeShapeFeatures.F_CENTRAL_MOMENT_0_1.value: props["moments_central-0-1"],
                        ObjectSizeShapeFeatures.F_CENTRAL_MOMENT_0_2.value: props["moments_central-0-2"],
                        ObjectSizeShapeFeatures.F_CENTRAL_MOMENT_0_3.value: props["moments_central-0-3"],
                        ObjectSizeShapeFeatures.F_CENTRAL_MOMENT_1_0.value: props["moments_central-1-0"],
                        ObjectSizeShapeFeatures.F_CENTRAL_MOMENT_1_1.value: props["moments_central-1-1"],
                        ObjectSizeShapeFeatures.F_CENTRAL_MOMENT_1_2.value: props["moments_central-1-2"],
                        ObjectSizeShapeFeatures.F_CENTRAL_MOMENT_1_3.value: props["moments_central-1-3"],
                        ObjectSizeShapeFeatures.F_CENTRAL_MOMENT_2_0.value: props["moments_central-2-0"],
                        ObjectSizeShapeFeatures.F_CENTRAL_MOMENT_2_1.value: props["moments_central-2-1"],
                        ObjectSizeShapeFeatures.F_CENTRAL_MOMENT_2_2.value: props["moments_central-2-2"],
                        ObjectSizeShapeFeatures.F_CENTRAL_MOMENT_2_3.value: props["moments_central-2-3"],
                        ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_0_0.value: props["moments_normalized-0-0"],
                        ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_0_1.value: props["moments_normalized-0-1"],
                        ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_0_2.value: props["moments_normalized-0-2"],
                        ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_0_3.value: props["moments_normalized-0-3"],
                        ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_1_0.value: props["moments_normalized-1-0"],
                        ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_1_1.value: props["moments_normalized-1-1"],
                        ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_1_2.value: props["moments_normalized-1-2"],
                        ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_1_3.value: props["moments_normalized-1-3"],
                        ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_2_0.value: props["moments_normalized-2-0"],
                        ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_2_1.value: props["moments_normalized-2-1"],
                        ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_2_2.value: props["moments_normalized-2-2"],
                        ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_2_3.value: props["moments_normalized-2-3"],
                        ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_3_0.value: props["moments_normalized-3-0"],
                        ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_3_1.value: props["moments_normalized-3-1"],
                        ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_3_2.value: props["moments_normalized-3-2"],
                        ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_3_3.value: props["moments_normalized-3-3"],
                        ObjectSizeShapeFeatures.F_HU_MOMENT_0.value: props["moments_hu-0"],
                        ObjectSizeShapeFeatures.F_HU_MOMENT_1.value: props["moments_hu-1"],
                        ObjectSizeShapeFeatures.F_HU_MOMENT_2.value: props["moments_hu-2"],
                        ObjectSizeShapeFeatures.F_HU_MOMENT_3.value: props["moments_hu-3"],
                        ObjectSizeShapeFeatures.F_HU_MOMENT_4.value: props["moments_hu-4"],
                        ObjectSizeShapeFeatures.F_HU_MOMENT_5.value: props["moments_hu-5"],
                        ObjectSizeShapeFeatures.F_HU_MOMENT_6.value: props["moments_hu-6"],
                        ObjectSizeShapeFeatures.F_INERTIA_TENSOR_0_0.value: props["inertia_tensor-0-0"],
                        ObjectSizeShapeFeatures.F_INERTIA_TENSOR_0_1.value: props["inertia_tensor-0-1"],
                        ObjectSizeShapeFeatures.F_INERTIA_TENSOR_1_0.value: props["inertia_tensor-1-0"],
                        ObjectSizeShapeFeatures.F_INERTIA_TENSOR_1_1.value: props["inertia_tensor-1-1"],
                        ObjectSizeShapeFeatures.F_INERTIA_TENSOR_EIGENVALUES_0.value: props[
                            "inertia_tensor_eigvals-0"
                        ],
                        ObjectSizeShapeFeatures.F_INERTIA_TENSOR_EIGENVALUES_1.value: props[
                            "inertia_tensor_eigvals-1"
                        ],
                    }
                )

            if calculate_zernikes:
                features_to_record.update(
                    {f"Zernike_{n}_{m}": zf[(n, m)] for n, m in zernike_numbers}
                )

    else:
        # 3D
        props = skimage.measure.regionprops_table(labels, properties=desired_properties)
        # SurfaceArea
        surface_areas = numpy.zeros(len(props["label"]))
        for index, label in enumerate(props["label"]):
            # this seems less elegant than you might wish, given that regionprops returns a slice,
            # but we need to expand the slice out by one voxel in each direction, or surface area freaks out
            volume = labels[
                max(props["bbox-0"][index] - 1, 0) : min(
                    props["bbox-3"][index] + 1, labels.shape[0]
                ),
                max(props["bbox-1"][index] - 1, 0) : min(
                    props["bbox-4"][index] + 1, labels.shape[1]
                ),
                max(props["bbox-2"][index] - 1, 0) : min(
                    props["bbox-5"][index] + 1, labels.shape[2]
                ),
            ]
            volume = volume == label
            verts, faces, _normals, _values = skimage.measure.marching_cubes(
                volume,
                method="lewiner",
                spacing=spacing,
                level=0,
            )
            surface_areas[index] = skimage.measure.mesh_surface_area(verts, faces)

        features_to_record = {
            ObjectSizeShapeFeatures.F_VOLUME.value: props["area"],
            ObjectSizeShapeFeatures.F_SURFACE_AREA.value: surface_areas,
            ObjectSizeShapeFeatures.F_MAJOR_AXIS_LENGTH.value: props["major_axis_length"],
            ObjectSizeShapeFeatures.F_MINOR_AXIS_LENGTH.value: props["minor_axis_length"],
            ObjectSizeShapeFeatures.F_CENTER_X.value: props["centroid-2"],
            ObjectSizeShapeFeatures.F_CENTER_Y.value: props["centroid-1"],
            ObjectSizeShapeFeatures.F_CENTER_Z.value: props["centroid-0"],
            ObjectSizeShapeFeatures.F_BBOX_VOLUME.value: props["bbox_area"],
            ObjectSizeShapeFeatures.F_MIN_X.value: props["bbox-2"],
            ObjectSizeShapeFeatures.F_MAX_X.value: props["bbox-5"],
            ObjectSizeShapeFeatures.F_MIN_Y.value: props["bbox-1"],
            ObjectSizeShapeFeatures.F_MAX_Y.value: props["bbox-4"],
            ObjectSizeShapeFeatures.F_MIN_Z.value: props["bbox-0"],
            ObjectSizeShapeFeatures.F_MAX_Z.value: props["bbox-3"],
            ObjectSizeShapeFeatures.F_EXTENT.value: props["extent"],
            ObjectSizeShapeFeatures.F_EULER_NUMBER.value: props["euler_number"],
            ObjectSizeShapeFeatures.F_EQUIVALENT_DIAMETER.value: props["equivalent_diameter"],
        }
        if calculate_advanced:
            features_to_record[ObjectSizeShapeFeatures.F_SOLIDITY.value] = props["solidity"]
    return features_to_record, props["label"], nobjects



def measure_object_size_shape(
    labels,
    desired_properties,
    calculate_zernikes: bool = True,
    calculate_advanced: bool = True,
    spacing: Tuple = None
):
    label_indices = numpy.unique(labels[labels != 0])
    nobjects = len(label_indices)
    
    if spacing is None:
        spacing = (1.0,) * labels.ndim

    if len(labels.shape) == 2:
        # 2D
        props = skimage.measure.regionprops_table(labels, properties=desired_properties)

        formfactor = 4.0 * numpy.pi * props["area"] / props["perimeter"] ** 2
        denom = [max(x, 1) for x in 4.0 * numpy.pi * props["area"]]
        compactness = props["perimeter"] ** 2 / denom

        max_radius = numpy.zeros(nobjects)
        median_radius = numpy.zeros(nobjects)
        mean_radius = numpy.zeros(nobjects)
        min_feret_diameter = numpy.zeros(nobjects)
        max_feret_diameter = numpy.zeros(nobjects)
        zernike_numbers = centrosome.zernike.get_zernike_indexes(ObjectSizeShapeFeatures.ZERNIKE_N.value + 1)

        zf = {}
        for n, m in zernike_numbers:
            zf[(n, m)] = numpy.zeros(nobjects)

        for index, mini_image in enumerate(props["image"]):
            # Pad image to assist distance tranform
            mini_image = numpy.pad(mini_image, 1)
            distances = scipy.ndimage.distance_transform_edt(mini_image)
            max_radius[index] = centrosome.cpmorphology.fixup_scipy_ndimage_result(
                scipy.ndimage.maximum(distances, mini_image)
            )
            mean_radius[index] = centrosome.cpmorphology.fixup_scipy_ndimage_result(
                scipy.ndimage.mean(distances, mini_image)
            )
            median_radius[index] = centrosome.cpmorphology.median_of_labels(
                distances, mini_image.astype("int"), [1]
            )

        #
        # Zernike features
        #
        if calculate_zernikes:
            zf_l = centrosome.zernike.zernike(zernike_numbers, labels, label_indices)
            for (n, m), z in zip(zernike_numbers, zf_l.transpose()):
                zf[(n, m)] = z

        if nobjects > 0:
            chulls, chull_counts = centrosome.cpmorphology.convex_hull(
                labels, label_indices
            )
            #
            # Feret diameter
            #
            (
                min_feret_diameter,
                max_feret_diameter,
            ) = centrosome.cpmorphology.feret_diameter(
                chulls, chull_counts, label_indices
            )

            features_to_record = {
                ObjectSizeShapeFeatures.F_AREA.value: props["area"],
                ObjectSizeShapeFeatures.F_PERIMETER.value: props["perimeter"],
                ObjectSizeShapeFeatures.F_MAJOR_AXIS_LENGTH.value: props["major_axis_length"],
                ObjectSizeShapeFeatures.F_MINOR_AXIS_LENGTH.value: props["minor_axis_length"],
                ObjectSizeShapeFeatures.F_ECCENTRICITY.value: props["eccentricity"],
                ObjectSizeShapeFeatures.F_ORIENTATION.value: props["orientation"] * (180 / numpy.pi),
                ObjectSizeShapeFeatures.F_CENTER_X.value: props["centroid-1"],
                ObjectSizeShapeFeatures.F_CENTER_Y.value: props["centroid-0"],
                ObjectSizeShapeFeatures.F_BBOX_AREA.value: props["bbox_area"],
                ObjectSizeShapeFeatures.F_MIN_X.value: props["bbox-1"],
                ObjectSizeShapeFeatures.F_MAX_X.value: props["bbox-3"],
                ObjectSizeShapeFeatures.F_MIN_Y.value: props["bbox-0"],
                ObjectSizeShapeFeatures.F_MAX_Y.value: props["bbox-2"],
                ObjectSizeShapeFeatures.F_FORM_FACTOR.value: formfactor,
                ObjectSizeShapeFeatures.F_EXTENT.value: props["extent"],
                ObjectSizeShapeFeatures.F_SOLIDITY.value: props["solidity"],
                ObjectSizeShapeFeatures.F_COMPACTNESS.value: compactness,
                ObjectSizeShapeFeatures.F_EULER_NUMBER.value: props["euler_number"],
                ObjectSizeShapeFeatures.F_MAXIMUM_RADIUS.value: max_radius,
                ObjectSizeShapeFeatures.F_MEAN_RADIUS.value: mean_radius,
                ObjectSizeShapeFeatures.F_MEDIAN_RADIUS.value: median_radius,
                ObjectSizeShapeFeatures.F_CONVEX_AREA.value: props["convex_area"],
                ObjectSizeShapeFeatures.F_MIN_FERET_DIAMETER.value: min_feret_diameter,
                ObjectSizeShapeFeatures.F_MAX_FERET_DIAMETER.value: max_feret_diameter,
                ObjectSizeShapeFeatures.F_EQUIVALENT_DIAMETER.value: props["equivalent_diameter"],
            }
            if calculate_advanced:
                features_to_record.update(
                    {
                        ObjectSizeShapeFeatures.F_SPATIAL_MOMENT_0_0.value: props["moments-0-0"],
                        ObjectSizeShapeFeatures.F_SPATIAL_MOMENT_0_1.value: props["moments-0-1"],
                        ObjectSizeShapeFeatures.F_SPATIAL_MOMENT_0_2.value: props["moments-0-2"],
                        ObjectSizeShapeFeatures.F_SPATIAL_MOMENT_0_3.value: props["moments-0-3"],
                        ObjectSizeShapeFeatures.F_SPATIAL_MOMENT_1_0.value: props["moments-1-0"],
                        ObjectSizeShapeFeatures.F_SPATIAL_MOMENT_1_1.value: props["moments-1-1"],
                        ObjectSizeShapeFeatures.F_SPATIAL_MOMENT_1_2.value: props["moments-1-2"],
                        ObjectSizeShapeFeatures.F_SPATIAL_MOMENT_1_3.value: props["moments-1-3"],
                        ObjectSizeShapeFeatures.F_SPATIAL_MOMENT_2_0.value: props["moments-2-0"],
                        ObjectSizeShapeFeatures.F_SPATIAL_MOMENT_2_1.value: props["moments-2-1"],
                        ObjectSizeShapeFeatures.F_SPATIAL_MOMENT_2_2.value: props["moments-2-2"],
                        ObjectSizeShapeFeatures.F_SPATIAL_MOMENT_2_3.value: props["moments-2-3"],
                        ObjectSizeShapeFeatures.F_CENTRAL_MOMENT_0_0.value: props["moments_central-0-0"],
                        ObjectSizeShapeFeatures.F_CENTRAL_MOMENT_0_1.value: props["moments_central-0-1"],
                        ObjectSizeShapeFeatures.F_CENTRAL_MOMENT_0_2.value: props["moments_central-0-2"],
                        ObjectSizeShapeFeatures.F_CENTRAL_MOMENT_0_3.value: props["moments_central-0-3"],
                        ObjectSizeShapeFeatures.F_CENTRAL_MOMENT_1_0.value: props["moments_central-1-0"],
                        ObjectSizeShapeFeatures.F_CENTRAL_MOMENT_1_1.value: props["moments_central-1-1"],
                        ObjectSizeShapeFeatures.F_CENTRAL_MOMENT_1_2.value: props["moments_central-1-2"],
                        ObjectSizeShapeFeatures.F_CENTRAL_MOMENT_1_3.value: props["moments_central-1-3"],
                        ObjectSizeShapeFeatures.F_CENTRAL_MOMENT_2_0.value: props["moments_central-2-0"],
                        ObjectSizeShapeFeatures.F_CENTRAL_MOMENT_2_1.value: props["moments_central-2-1"],
                        ObjectSizeShapeFeatures.F_CENTRAL_MOMENT_2_2.value: props["moments_central-2-2"],
                        ObjectSizeShapeFeatures.F_CENTRAL_MOMENT_2_3.value: props["moments_central-2-3"],
                        ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_0_0.value: props["moments_normalized-0-0"],
                        ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_0_1.value: props["moments_normalized-0-1"],
                        ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_0_2.value: props["moments_normalized-0-2"],
                        ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_0_3.value: props["moments_normalized-0-3"],
                        ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_1_0.value: props["moments_normalized-1-0"],
                        ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_1_1.value: props["moments_normalized-1-1"],
                        ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_1_2.value: props["moments_normalized-1-2"],
                        ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_1_3.value: props["moments_normalized-1-3"],
                        ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_2_0.value: props["moments_normalized-2-0"],
                        ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_2_1.value: props["moments_normalized-2-1"],
                        ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_2_2.value: props["moments_normalized-2-2"],
                        ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_2_3.value: props["moments_normalized-2-3"],
                        ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_3_0.value: props["moments_normalized-3-0"],
                        ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_3_1.value: props["moments_normalized-3-1"],
                        ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_3_2.value: props["moments_normalized-3-2"],
                        ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_3_3.value: props["moments_normalized-3-3"],
                        ObjectSizeShapeFeatures.F_HU_MOMENT_0.value: props["moments_hu-0"],
                        ObjectSizeShapeFeatures.F_HU_MOMENT_1.value: props["moments_hu-1"],
                        ObjectSizeShapeFeatures.F_HU_MOMENT_2.value: props["moments_hu-2"],
                        ObjectSizeShapeFeatures.F_HU_MOMENT_3.value: props["moments_hu-3"],
                        ObjectSizeShapeFeatures.F_HU_MOMENT_4.value: props["moments_hu-4"],
                        ObjectSizeShapeFeatures.F_HU_MOMENT_5.value: props["moments_hu-5"],
                        ObjectSizeShapeFeatures.F_HU_MOMENT_6.value: props["moments_hu-6"],
                        ObjectSizeShapeFeatures.F_INERTIA_TENSOR_0_0.value: props["inertia_tensor-0-0"],
                        ObjectSizeShapeFeatures.F_INERTIA_TENSOR_0_1.value: props["inertia_tensor-0-1"],
                        ObjectSizeShapeFeatures.F_INERTIA_TENSOR_1_0.value: props["inertia_tensor-1-0"],
                        ObjectSizeShapeFeatures.F_INERTIA_TENSOR_1_1.value: props["inertia_tensor-1-1"],
                        ObjectSizeShapeFeatures.F_INERTIA_TENSOR_EIGENVALUES_0.value: props[
                            "inertia_tensor_eigvals-0"
                        ],
                        ObjectSizeShapeFeatures.F_INERTIA_TENSOR_EIGENVALUES_1.value: props[
                            "inertia_tensor_eigvals-1"
                        ],
                    }
                )

            if calculate_zernikes:
                features_to_record.update(
                    {f"Zernike_{n}_{m}": zf[(n, m)] for n, m in zernike_numbers}
                )

    else:
        # 3D
        props = skimage.measure.regionprops_table(labels, properties=desired_properties)
        # SurfaceArea
        surface_areas = numpy.zeros(len(props["label"]))
        for index, label in enumerate(props["label"]):
            # this seems less elegant than you might wish, given that regionprops returns a slice,
            # but we need to expand the slice out by one voxel in each direction, or surface area freaks out
            volume = labels[
                max(props["bbox-0"][index] - 1, 0) : min(
                    props["bbox-3"][index] + 1, labels.shape[0]
                ),
                max(props["bbox-1"][index] - 1, 0) : min(
                    props["bbox-4"][index] + 1, labels.shape[1]
                ),
                max(props["bbox-2"][index] - 1, 0) : min(
                    props["bbox-5"][index] + 1, labels.shape[2]
                ),
            ]
            volume = volume == label
            verts, faces, _normals, _values = skimage.measure.marching_cubes(
                volume,
                method="lewiner",
                spacing=spacing,
                level=0,
            )
            surface_areas[index] = skimage.measure.mesh_surface_area(verts, faces)

        features_to_record = {
            ObjectSizeShapeFeatures.F_VOLUME.value: props["area"],
            ObjectSizeShapeFeatures.F_SURFACE_AREA.value: surface_areas,
            ObjectSizeShapeFeatures.F_MAJOR_AXIS_LENGTH.value: props["major_axis_length"],
            ObjectSizeShapeFeatures.F_MINOR_AXIS_LENGTH.value: props["minor_axis_length"],
            ObjectSizeShapeFeatures.F_CENTER_X.value: props["centroid-2"],
            ObjectSizeShapeFeatures.F_CENTER_Y.value: props["centroid-1"],
            ObjectSizeShapeFeatures.F_CENTER_Z.value: props["centroid-0"],
            ObjectSizeShapeFeatures.F_BBOX_VOLUME.value: props["bbox_area"],
            ObjectSizeShapeFeatures.F_MIN_X.value: props["bbox-2"],
            ObjectSizeShapeFeatures.F_MAX_X.value: props["bbox-5"],
            ObjectSizeShapeFeatures.F_MIN_Y.value: props["bbox-1"],
            ObjectSizeShapeFeatures.F_MAX_Y.value: props["bbox-4"],
            ObjectSizeShapeFeatures.F_MIN_Z.value: props["bbox-0"],
            ObjectSizeShapeFeatures.F_MAX_Z.value: props["bbox-3"],
            ObjectSizeShapeFeatures.F_EXTENT.value: props["extent"],
            ObjectSizeShapeFeatures.F_EULER_NUMBER.value: props["euler_number"],
            ObjectSizeShapeFeatures.F_EQUIVALENT_DIAMETER.value: props["equivalent_diameter"],
        }
        if calculate_advanced:
            features_to_record[ObjectSizeShapeFeatures.F_SOLIDITY.value] = props["solidity"]
    return features_to_record, props["label"], nobjects



########################################################
# MeasureColocalization
########################################################
# TODO: Decide if the code for image and object should be combined into a single function

# TODO: verify types
# TODO: move to image_processing.py
def get_thresholded_images_and_counts(
    first_image: Image2DGrayscale, 
    second_image: Image2DGrayscale,
    first_threshold_value: float,
    second_threshold_value: float,
) -> Tuple[float, float, NDArray[numpy.float64], NDArray[numpy.float64], NDArray[numpy.float64], NDArray[numpy.float64], NDArray[numpy.bool_]]:
    fi = first_image
    si = second_image
    thr_fi = first_threshold_value * numpy.max(fi) / 100
    thr_si = second_threshold_value * numpy.max(si) / 100
    combined_thresh = (fi > thr_fi) & (si > thr_si)
    fi_thresh = fi[combined_thresh]
    si_thresh = si[combined_thresh]
    tot_fi_thr = fi[(fi > thr_fi)].sum()
    tot_si_thr = si[(si > thr_si)].sum()
    return thr_fi, thr_si, fi_thresh, si_thresh, tot_fi_thr, tot_si_thr, combined_thresh

# TODO: verify types
def get_thresholded_images_and_counts_from_objects(
    first_pixels: np.ndarray,
    second_pixels: np.ndarray,
    im1_threshold: float,
    im2_threshold: float,
    labels: ObjectLabelsDense
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int32], NDArray[np.int32], NDArray[np.bool_]]:
    lrange = numpy.arange(labels.max(), dtype=numpy.int32) + 1
    # Threshold as percentage of maximum intensity of objects in each channel
    tff = (im1_threshold / 100) * fix(
        scipy.ndimage.maximum(first_pixels, labels, lrange)
    )
    tss = (im2_threshold / 100) * fix(
        scipy.ndimage.maximum(second_pixels, labels, lrange)
            )

    combined_thresh = (first_pixels >= tff[labels - 1]) & (second_pixels >= tss[labels - 1])
    fi_thresh = first_pixels[combined_thresh]
    si_thresh = second_pixels[combined_thresh]
    tot_fi_thr = scipy.ndimage.sum(
        first_pixels[first_pixels >= tff[labels - 1]],
        labels[first_pixels >= tff[labels - 1]],
        lrange,
    )
    tot_si_thr = scipy.ndimage.sum(
        second_pixels[second_pixels >= tss[labels - 1]],
        labels[second_pixels >= tss[labels - 1]],
        lrange,
    )
    fi_thresh = fi_thresh.astype(numpy.float64)
    si_thresh = si_thresh.astype(numpy.float64)
    tot_fi_thr = tot_fi_thr.astype(numpy.int32)
    tot_si_thr = tot_si_thr.astype(numpy.int32)
    combined_thresh = combined_thresh.astype(numpy.bool_)
    return fi_thresh, si_thresh, tot_fi_thr, tot_si_thr, combined_thresh

# TODO: verify types
def measure_correlation_and_slope(
        first_image: Image2DGrayscale, 
        second_image: Image2DGrayscale,
        first_image_name: str, 
        second_image_name: str, 
    ) -> Tuple[List[Tuple[str, str, str, str, str]], float, float]:
    result = []
    #
    # Perform the correlation, which returns:
    # [ [ii, ij],
    #   [ji, jj] ]
    #
    corr = np.corrcoef((first_image, second_image))[1, 0]
    #
    # Find the slope as a linear regression to
    # A * i1 + B = i2
    #
    least_squares_solution = lstsq(
        np.array((first_image, np.ones_like(first_image))).transpose(), 
        second_image)
    assert least_squares_solution is not None
    coeffs = least_squares_solution[0]
    slope = coeffs[0]
    assert slope is not None
    result += [
        [
            first_image_name,
            second_image_name,
            "-",
            "Correlation",
            "%.3f" % corr,
        ],
        [first_image_name, second_image_name, "-", "Slope", "%.3f" % slope],
    ]
    return result, corr, slope

# TODO: verify types
def measure_manders_coefficient(
        first_image: Image2DGrayscale, 
        second_image: Image2DGrayscale,
        first_image_name: str, 
        second_image_name: str, 
        first_threshold_value: Optional[float] = None,
        second_threshold_value: Optional[float] = None,
        fi_thresh: Optional[Image2DGrayscale] = None,
        si_thresh: Optional[Image2DGrayscale] = None,
        tot_fi_thr: Optional[int] = None,
        tot_si_thr: Optional[int] = None,
    ) -> Tuple[List[Tuple[str, str, str, str, str]], float, float]:
    if fi_thresh is None or si_thresh is None or tot_fi_thr is None or tot_si_thr is None:
        assert first_threshold_value is not None
        assert second_threshold_value is not None
        _, _, fi_thresh, si_thresh, tot_fi_thr, tot_si_thr, combined_thresh = get_thresholded_images_and_counts(
            first_image, second_image, first_threshold_value, second_threshold_value
        )
    assert fi_thresh is not None
    assert si_thresh is not None
    assert tot_fi_thr is not None
    assert tot_si_thr is not None
    result = []
    # Manders Coefficient
    M1 = 0
    M2 = 0
    M1 = fi_thresh.sum() / tot_fi_thr
    M2 = si_thresh.sum() / tot_si_thr

    result += [
        [
            first_image_name,
            second_image_name,
            "-",
            "Manders Coefficient",
            "%.3f" % M1,
        ],
        [
            second_image_name,
            first_image_name,
            "-",
            "Manders Coefficient",
            "%.3f" % M2,
        ],
    ]
    return result, M1, M2

# TODO: verify types
def measure_correlation_and_slope_from_objects(
    first_pixels: np.ndarray,
    second_pixels: np.ndarray,
    labels: ObjectLabelsDense,
    lrange: np.ndarray,
    first_image_name: str,
    second_image_name: str,
    object_name: str,
) -> Tuple[List[Tuple[str, str, str, str, str]], NDArray[np.float64]]:
    result = []
    #
    # The correlation is sum((x-mean(x))(y-mean(y)) /
    #                         ((n-1) * std(x) *std(y)))
    #

    mean1 = fix(scipy.ndimage.mean(first_pixels, labels, lrange))
    mean2 = fix(scipy.ndimage.mean(second_pixels, labels, lrange))
    #
    # Calculate the standard deviation times the population.
    #
    std1 = numpy.sqrt(
        fix(
            scipy.ndimage.sum(
                (first_pixels - mean1[labels - 1]) ** 2, labels, lrange
            )
        )
    )
    std2 = numpy.sqrt(
        fix(
            scipy.ndimage.sum(
                (second_pixels - mean2[labels - 1]) ** 2, labels, lrange
            )
        )
    )
    x = first_pixels - mean1[labels - 1]  # x - mean(x)
    y = second_pixels - mean2[labels - 1]  # y - mean(y)
    corr = fix(
        scipy.ndimage.sum(
            x * y / (std1[labels - 1] * std2[labels - 1]), labels, lrange
        )
    )
    # Explicitly set the correlation to NaN for masked objects
    corr[scipy.ndimage.sum(1, labels, lrange) == 0] = numpy.NaN
    col_order_1 = [first_image_name, second_image_name, object_name]
    result += [
        [ *col_order_1, "Mean Correlation coeff", "%.3f" % numpy.mean(corr)],
        [ *col_order_1, "Median Correlation coeff", "%.3f" % numpy.median(corr)],
        [ *col_order_1, "Min Correlation coeff", "%.3f" % numpy.min(corr)],
        [ *col_order_1, "Max Correlation coeff", "%.3f" % numpy.max(corr)],
    ]
    return result, corr

# TODO: verify types
def measure_manders_coefficient_from_objects(
    labels: ObjectLabelsDense,
    lrange: np.ndarray,
    first_image_name: str,
    second_image_name: str,
    object_name: str,
    fi_thresh: Optional[Image2DGrayscale] = None,
    si_thresh: Optional[Image2DGrayscale] = None,
    tot_fi_thr: Optional[NDArray[np.int32]] = None,
    tot_si_thr: Optional[NDArray[np.int32]] = None,
    combined_thresh: Optional[Image2DGrayscale] = None,
    ) -> Tuple[List[Tuple[str, str, str, str, str]], NDArray[np.float64], NDArray[np.float64]]:
    result = []
    # Manders Coefficient
    M1 = numpy.zeros(len(lrange))
    M2 = numpy.zeros(len(lrange))

    if combined_thresh is not None and numpy.any(combined_thresh):
        M1 = numpy.array(
            scipy.ndimage.sum(fi_thresh, labels[combined_thresh], lrange)
        ) / numpy.array(tot_fi_thr)
        M2 = numpy.array(
            scipy.ndimage.sum(si_thresh, labels[combined_thresh], lrange)
        ) / numpy.array(tot_si_thr)
    col_order_1 = [first_image_name, second_image_name, object_name]
    result += [
        [*col_order_1, "Mean Manders coeff", "%.3f" % numpy.mean(M1)],
        [*col_order_1, "Median Manders coeff", "%.3f" % numpy.median(M1)],
        [*col_order_1, "Min Manders coeff", "%.3f" % numpy.min(M1)],
        [*col_order_1, "Max Manders coeff", "%.3f" % numpy.max(M1)],
    ]
    col_order_2 = [second_image_name, first_image_name, object_name]
    result += [
        [*col_order_2, "Mean Manders coeff", "%.3f" % numpy.mean(M2)],
        [*col_order_2, "Median Manders coeff", "%.3f" % numpy.median(M2)],
        [*col_order_2, "Min Manders coeff", "%.3f" % numpy.min(M2)],
        [*col_order_2, "Max Manders coeff", "%.3f" % numpy.max(M2)],
    ]
    return result, M1, M2

# TODO: verify types
def measure_rwc_coefficient_from_objects(
    first_pixels: np.ndarray,
    second_pixels: np.ndarray,
    labels: ObjectLabelsDense,
    lrange: np.ndarray,
    first_image_name: str,
    second_image_name: str,
    object_name: str,
    fi_thresh: Optional[Image2DGrayscale] = None,
    si_thresh: Optional[Image2DGrayscale] = None,
    tot_fi_thr: Optional[NDArray[np.int32]] = None,
    tot_si_thr: Optional[NDArray[np.int32]] = None,
    combined_thresh: Optional[Image2DGrayscale] = None,
) -> Tuple[List[Tuple[str, str, str, str, str]], NDArray[np.float64], NDArray[np.float64]]:
    result = []
    # RWC Coefficient
    RWC1 = numpy.zeros(len(lrange))
    RWC2 = numpy.zeros(len(lrange))
    [Rank1] = numpy.lexsort(([labels], [first_pixels]))
    [Rank2] = numpy.lexsort(([labels], [second_pixels]))
    Rank1_U = numpy.hstack(
        [[False], first_pixels[Rank1[:-1]] != first_pixels[Rank1[1:]]]
    )
    Rank2_U = numpy.hstack(
        [[False], second_pixels[Rank2[:-1]] != second_pixels[Rank2[1:]]]
    )
    Rank1_S = numpy.cumsum(Rank1_U)
    Rank2_S = numpy.cumsum(Rank2_U)
    Rank_im1 = numpy.zeros(first_pixels.shape, dtype=int)
    Rank_im2 = numpy.zeros(second_pixels.shape, dtype=int)
    Rank_im1[Rank1] = Rank1_S
    Rank_im2[Rank2] = Rank2_S
    R = max(Rank_im1.max(), Rank_im2.max()) + 1
    Di = abs(Rank_im1 - Rank_im2)
    weight = (R - Di) * 1.0 / R
    weight_thresh = weight[combined_thresh]
    if combined_thresh is not None and numpy.any(combined_thresh):
        RWC1 = numpy.array(
            scipy.ndimage.sum(
                fi_thresh * weight_thresh, labels[combined_thresh], lrange
            )
        ) / numpy.array(tot_fi_thr)
        RWC2 = numpy.array(
            scipy.ndimage.sum(
                si_thresh * weight_thresh, labels[combined_thresh], lrange
            )
        ) / numpy.array(tot_si_thr)
    col_order_1 = [first_image_name, second_image_name, object_name]
    result += [
        [ *col_order_1, "Mean RWC coeff", "%.3f" % numpy.mean(RWC1)],
        [ *col_order_1, "Median RWC coeff", "%.3f" % numpy.median(RWC1)],
        [ *col_order_1, "Min RWC coeff", "%.3f" % numpy.min(RWC1)],
        [ *col_order_1, "Max RWC coeff", "%.3f" % numpy.max(RWC1)]
    ]
    col_order_2 = [second_image_name, first_image_name, object_name]
    result += [
        [ *col_order_2, "Mean RWC coeff", "%.3f" % numpy.mean(RWC2)],
        [ *col_order_2, "Median RWC coeff", "%.3f" % numpy.median(RWC2)],
        [ *col_order_2, "Min RWC coeff", "%.3f" % numpy.min(RWC2)],
        [ *col_order_2, "Max RWC coeff", "%.3f" % numpy.max(RWC2)]
    ]
    return result, RWC1, RWC2

# TODO: verify types
def measure_overlap_coefficient_from_objects(
    first_pixels: np.ndarray,
    second_pixels: np.ndarray,
    labels: ObjectLabelsDense,
    lrange: np.ndarray,
    first_image_name: str,
    second_image_name: str,
    object_name: str,
    combined_thresh: Optional[Image2DGrayscale] = None,
) -> Tuple[List[Tuple[str, str, str, str, str]], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    result = []
    # Overlap Coefficient
    if combined_thresh is not None and numpy.any(combined_thresh):
        fpsq = scipy.ndimage.sum(
            first_pixels[combined_thresh] ** 2,
            labels[combined_thresh],
            lrange,
        )
        spsq = scipy.ndimage.sum(
            second_pixels[combined_thresh] ** 2,
            labels[combined_thresh],
            lrange,
        )
        pdt = numpy.sqrt(numpy.array(fpsq) * numpy.array(spsq))

        overlap = fix(
            scipy.ndimage.sum(
                first_pixels[combined_thresh]
                * second_pixels[combined_thresh],
                labels[combined_thresh],
                lrange,
            )
            / pdt
        )
        K1 = fix(
            (
                scipy.ndimage.sum(
                    first_pixels[combined_thresh]
                    * second_pixels[combined_thresh],
                    labels[combined_thresh],
                    lrange,
                )
            )
            / (numpy.array(fpsq))
        )
        K2 = fix(
            scipy.ndimage.sum(
                first_pixels[combined_thresh]
                * second_pixels[combined_thresh],
                labels[combined_thresh],
                lrange,
            )
            / numpy.array(spsq)
        )
    else:
        overlap = K1 = K2 = numpy.zeros(len(lrange))

    col_order_1 = [first_image_name, second_image_name, object_name]
    result += [
        [*col_order_1, "Mean Overlap coeff", "%.3f" % numpy.mean(overlap)],
        [*col_order_1, "Median Overlap coeff", "%.3f" % numpy.median(overlap)],
        [*col_order_1, "Min Overlap coeff", "%.3f" % numpy.min(overlap)],
        [*col_order_1, "Max Overlap coeff", "%.3f" % numpy.max(overlap)],
    ]
    return result, overlap, K1, K2

# TODO: verify types
def measure_costes_coefficient_from_objects(
    first_pixels: np.ndarray,
    second_pixels: np.ndarray,
    labels: ObjectLabelsDense,
    lrange: np.ndarray,
    first_image_name: str,
    second_image_name: str,
    object_name: str,
    fi,
    si,
    first_image_scale,
    second_image_scale,
    costes_method: CostesMethod = CostesMethod.FAST,
    combined_thresh: Optional[Image2DGrayscale] = None,
) -> Tuple[List[Tuple[str, str, str, str, str]], NDArray[np.float64], NDArray[np.float64]]:
    result = []
    # Orthogonal Regression for Costes' automated threshold
    scale = get_scale(first_image_scale, second_image_scale)

    if costes_method == CostesMethod.FASTER:
        thr_fi_c, thr_si_c = bisection_costes(fi, si, scale)
    else:
        thr_fi_c, thr_si_c = linear_costes(fi, si, scale)

    # Costes' thershold for entire image is applied to each object
    fi_above_thr = first_pixels > thr_fi_c
    si_above_thr = second_pixels > thr_si_c
    combined_thresh_c = fi_above_thr & si_above_thr
    fi_thresh_c = first_pixels[combined_thresh_c]
    si_thresh_c = second_pixels[combined_thresh_c]
    if numpy.any(fi_above_thr):
        tot_fi_thr_c = scipy.ndimage.sum(
            first_pixels[first_pixels >= thr_fi_c],
            labels[first_pixels >= thr_fi_c],
            lrange,
        )
    else:
        tot_fi_thr_c = numpy.zeros(len(lrange))
    if numpy.any(si_above_thr):
        tot_si_thr_c = scipy.ndimage.sum(
            second_pixels[second_pixels >= thr_si_c],
            labels[second_pixels >= thr_si_c],
            lrange,
        )
    else:
        tot_si_thr_c = numpy.zeros(len(lrange))

    # Costes Automated Threshold
    C1 = numpy.zeros(len(lrange))
    C2 = numpy.zeros(len(lrange))
    if numpy.any(combined_thresh_c):
        C1 = numpy.array(
            scipy.ndimage.sum(
                fi_thresh_c, labels[combined_thresh_c], lrange
            )
        ) / numpy.array(tot_fi_thr_c)
        C2 = numpy.array(
            scipy.ndimage.sum(
                si_thresh_c, labels[combined_thresh_c], lrange
            )
        ) / numpy.array(tot_si_thr_c)
    col_order_1 = [first_image_name, second_image_name, object_name]
    result += [
        [*col_order_1, "Mean Manders coeff (Costes)", "%.3f" % numpy.mean(C1)],
        [*col_order_1, "Median Manders coeff (Costes)", "%.3f" % numpy.median(C1)],
        [*col_order_1, "Min Manders coeff (Costes)", "%.3f" % numpy.min(C1)],
        [*col_order_1, "Max Manders coeff (Costes)", "%.3f" % numpy.max(C1)],
    ]
    col_order_2 = [second_image_name, first_image_name, object_name]
    result += [
        [*col_order_2, "Mean Manders coeff (Costes)", "%.3f" % numpy.mean(C2)],
        [*col_order_2, "Median Manders coeff (Costes)", "%.3f" % numpy.median(C2)],
        [*col_order_2, "Min Manders coeff (Costes)", "%.3f" % numpy.min(C2)],
        [*col_order_2, "Max Manders coeff (Costes)", "%.3f" % numpy.max(C2)],
    ]

    return result, C1, C2

# TODO: verify types
def measure_rwc_coefficient(
        first_image: Image2DGrayscale, 
        second_image: Image2DGrayscale,
        first_image_name: str, 
        second_image_name: str, 
        first_threshold_value: Optional[float] = None,
        second_threshold_value: Optional[float] = None,
        fi_thresh: Optional[Image2DGrayscale] = None,
        si_thresh: Optional[Image2DGrayscale] = None,
        tot_fi_thr: Optional[int] = None,
        tot_si_thr: Optional[int] = None,
        combined_thresh: Optional[Image2DGrayscale] = None,
    ) -> Tuple[List[Tuple[str, str, str, str, str]], float, float]:
    fi = first_image
    si = second_image
    if fi_thresh is None or si_thresh is None or tot_fi_thr is None or tot_si_thr is None or combined_thresh is None:
        assert first_threshold_value is not None
        assert second_threshold_value is not None
        _, _, fi_thresh, si_thresh, tot_fi_thr, tot_si_thr, combined_thresh = get_thresholded_images_and_counts(
            first_image, second_image, first_threshold_value, second_threshold_value
        )
    assert fi_thresh is not None
    assert si_thresh is not None
    assert tot_fi_thr is not None
    assert tot_si_thr is not None
    assert combined_thresh is not None

    result = []
    # RWC Coefficient
    RWC1 = 0
    RWC2 = 0
    Rank1 = np.lexsort([fi])
    Rank2 = np.lexsort([si])
    Rank1_U = np.hstack([[False], fi[Rank1[:-1]] != fi[Rank1[1:]]])
    Rank2_U = np.hstack([[False], si[Rank2[:-1]] != si[Rank2[1:]]])
    Rank1_S = np.cumsum(Rank1_U)
    Rank2_S = np.cumsum(Rank2_U)
    Rank_im1 = np.zeros(fi.shape, dtype=int)
    Rank_im2 = np.zeros(si.shape, dtype=int)
    Rank_im1[Rank1] = Rank1_S
    Rank_im2[Rank2] = Rank2_S

    R = max(Rank_im1.max(), Rank_im2.max()) + 1
    Di = abs(Rank_im1 - Rank_im2)
    weight = ((R - Di) * 1.0) / R
    weight_thresh = weight[combined_thresh]
    RWC1 = (fi_thresh * weight_thresh).sum() / tot_fi_thr
    RWC2 = (si_thresh * weight_thresh).sum() / tot_si_thr
    result += [
        [
            first_image_name,
            second_image_name,
            "-",
            "RWC Coefficient",
            "%.3f" % RWC1,
        ],
        [
            second_image_name,
            first_image_name,
            "-",
            "RWC Coefficient",
            "%.3f" % RWC2,
        ],
    ]
    return result, RWC1, RWC2

# TODO: verify types
def measure_overlap_coefficient(
        first_image: Image2DGrayscale, 
        second_image: Image2DGrayscale,
        first_image_name: str, 
        second_image_name: str, 
        first_threshold_value: Optional[float] = None,
        second_threshold_value: Optional[float] = None,
        fi_thresh: Optional[Image2DGrayscale] = None,
        si_thresh: Optional[Image2DGrayscale] = None,
    ) -> Tuple[List[Tuple[str, str, str, str, str]], float, float, float]:
    fi = first_image
    si = second_image
    if fi_thresh is None or si_thresh is None:
        assert first_threshold_value is not None
        assert second_threshold_value is not None
        _, _, fi_thresh, si_thresh, _, _, _ = get_thresholded_images_and_counts(
            first_image, second_image, first_threshold_value, second_threshold_value
        )
    assert fi_thresh is not None
    assert si_thresh is not None
    result = []
    # Overlap Coefficient
    overlap = 0
    overlap = (fi_thresh * si_thresh).sum() / np.sqrt(
        (fi_thresh ** 2).sum() * (si_thresh ** 2).sum()
    )
    K1 = (fi_thresh * si_thresh).sum() / (fi_thresh ** 2).sum()
    K2 = (fi_thresh * si_thresh).sum() / (si_thresh ** 2).sum()
    result += [
        [
            first_image_name,
            second_image_name,
            "-",
            "Overlap Coefficient",
            "%.3f" % overlap,
        ]
    ]
    return result, overlap, K1, K2

# TODO: verify types
def measure_costes_coefficient(
        first_image: Image2DGrayscale, 
        second_image: Image2DGrayscale,
        first_image_name: str, 
        second_image_name: str, 
        first_image_scale: Optional[float] = None,
        second_image_scale: Optional[float] = None,
        first_threshold_value: Optional[float] = None,
        second_threshold_value: Optional[float] = None,
        fi_thresh: Optional[Image2DGrayscale] = None,
        si_thresh: Optional[Image2DGrayscale] = None,
        costes_method: CostesMethod = CostesMethod.FAST,
    ) -> Tuple[List[Tuple[str, str, str, str, str]], float, float]:
    fi = first_image
    si = second_image
    if fi_thresh is None or si_thresh is None:
        assert first_threshold_value is not None
        assert second_threshold_value is not None
        _, _, fi_thresh, si_thresh, _, _, _ = get_thresholded_images_and_counts(
            first_image, second_image, first_threshold_value, second_threshold_value
        )
    assert fi_thresh is not None
    assert si_thresh is not None
    result = []
    # Orthogonal Regression for Costes' automated threshold
    scale = get_scale(first_image_scale, second_image_scale)
    if costes_method == CostesMethod.FASTER:
        thr_fi_c, thr_si_c = bisection_costes(fi, si, scale)
    else:
        thr_fi_c, thr_si_c = linear_costes(fi, si, scale)

    # Costes' thershold calculation
    combined_thresh_c = (fi > thr_fi_c) & (si > thr_si_c)
    fi_thresh_c = fi[combined_thresh_c]
    si_thresh_c = si[combined_thresh_c]
    tot_fi_thr_c = fi[(fi > thr_fi_c)].sum()
    tot_si_thr_c = si[(si > thr_si_c)].sum()

    # Costes' Automated Threshold
    C1 = 0
    C2 = 0
    C1 = fi_thresh_c.sum() / tot_fi_thr_c
    C2 = si_thresh_c.sum() / tot_si_thr_c

    result += [
        [
            first_image_name,
            second_image_name,
            "-",
            "Manders Coefficient (Costes)",
            "%.3f" % C1,
        ],
        [
            second_image_name,
            first_image_name,
            "-",
            "Manders Coefficient (Costes)",
            "%.3f" % C2,
        ],
    ]
    return result, C1, C2


def get_scale(scale_1, scale_2):
    if scale_1 is not None and scale_2 is not None:
        return max(scale_1, scale_2)
    elif scale_1 is not None:
        return scale_1
    elif scale_2 is not None:
        return scale_2
    else:
        return 255

# TODO: verify types
def bisection_costes(fi: Image2DGrayscale, si: Image2DGrayscale, scale_max:float=255) -> Tuple[float, float]:
    """
    Finds the Costes Automatic Threshold for colocalization using a bisection algorithm.
    Candidate thresholds are selected from within a window of possible intensities,
    this window is narrowed based on the R value of each tested candidate.
    We're looking for the first point below 0, and R value can become highly variable
    at lower thresholds in some samples. Therefore the candidate tested in each
    loop is 1/6th of the window size below the maximum value (as opposed to the midpoint).
    """

    non_zero = (fi > 0) | (si > 0)
    xvar = np.var(fi[non_zero], axis=0, ddof=1)
    yvar = np.var(si[non_zero], axis=0, ddof=1)

    xmean = np.mean(fi[non_zero], axis=0)
    ymean = np.mean(si[non_zero], axis=0)

    z = fi[non_zero] + si[non_zero]
    zvar = np.var(z, axis=0, ddof=1)

    covar = 0.5 * (zvar - (xvar + yvar))

    denom = 2 * covar
    num = (yvar - xvar) + np.sqrt(
        (yvar - xvar) * (yvar - xvar) + 4 * (covar * covar)
    )
    a = num / denom
    b = ymean - a * xmean

    # Initialise variables
    left = 1
    right = scale_max
    mid = ((right - left) // (6/5)) + left
    lastmid = 0
    # Marks the value with the last positive R value.
    valid = 1

    while lastmid != mid:
        thr_fi_c = mid / scale_max
        thr_si_c = (a * thr_fi_c) + b
        combt = (fi < thr_fi_c) | (si < thr_si_c)
        if np.count_nonzero(combt) <= 2:
            # Can't run pearson with only 2 values.
            left = mid - 1
        else:
            try:
                costReg, _ = scipy.stats.pearsonr(fi[combt], si[combt])
                if costReg < 0:
                    left = mid - 1
                elif costReg >= 0:
                    right = mid + 1
                    valid = mid
            except ValueError:
                # Catch misc Pearson errors with low sample numbers
                left = mid - 1
        lastmid = mid
        if right - left > 6:
            mid = ((right - left) // (6 / 5)) + left
        else:
            mid = ((right - left) // 2) + left

    thr_fi_c = (valid - 1) / scale_max
    thr_si_c = (a * thr_fi_c) + b

    return thr_fi_c, thr_si_c

# TODO: verify types
def linear_costes(fi: Image2DGrayscale, si: Image2DGrayscale, scale_max:float=255, costes_method: CostesMethod = CostesMethod.FAST) -> Tuple[float, float]:
    """
    Finds the Costes Automatic Threshold for colocalization using a linear algorithm.
    Candiate thresholds are gradually decreased until Pearson R falls below 0.
    If "Fast" mode is enabled the "steps" between tested thresholds will be increased
    when Pearson R is much greater than 0.
    """
    i_step = 1 / scale_max
    non_zero = (fi > 0) | (si > 0)
    xvar = np.var(fi[non_zero], axis=0, ddof=1)
    yvar = np.var(si[non_zero], axis=0, ddof=1)

    xmean = np.mean(fi[non_zero], axis=0)
    ymean = np.mean(si[non_zero], axis=0)

    z = fi[non_zero] + si[non_zero]
    zvar = np.var(z, axis=0, ddof=1)

    covar = 0.5 * (zvar - (xvar + yvar))

    denom = 2 * covar
    num = (yvar - xvar) + np.sqrt(
        (yvar - xvar) * (yvar - xvar) + 4 * (covar * covar)
    )
    a = num / denom
    b = ymean - a * xmean

    # Start at 1 step above the maximum value
    img_max = max(fi.max(), si.max())
    i = i_step * ((img_max // i_step) + 1)

    num_true = None
    fi_max = fi.max()
    si_max = si.max()

    # Initialise without a threshold
    costReg, _ = scipy.stats.pearsonr(fi, si)
    thr_fi_c = i
    thr_si_c = (a * i) + b
    while i > fi_max and (a * i) + b > si_max:
        i -= i_step
    while i > i_step:
        thr_fi_c = i
        thr_si_c = (a * i) + b
        combt = (fi < thr_fi_c) | (si < thr_si_c)
        try:
            # Only run pearsonr if the input has changed.
            if (positives := np.count_nonzero(combt)) != num_true:
                costReg, _ = scipy.stats.pearsonr(fi[combt], si[combt])
                num_true = positives

            if costReg <= 0:
                break
            elif costes_method == CostesMethod.ACCURATE or i < i_step * 10:
                i -= i_step
            elif costReg > 0.45:
                # We're way off, step down 10x
                i -= i_step * 10
            elif costReg > 0.35:
                # Still far from 0, step 5x
                i -= i_step * 5
            elif costReg > 0.25:
                # Step 2x
                i -= i_step * 2
            else:
                i -= i_step
        except ValueError:
            break
    return thr_fi_c, thr_si_c
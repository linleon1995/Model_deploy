import nrrd
import numpy as np
import os


def save_nodule_in_nrrd(
    vol, mask_vol, direction, origin, spacing, save_dir, filename, space='left-posterior-superior'):
    """Save input volume and mask volume in nrrd and seg.nrrd format

    Args:
        vol ([D, H, W]): Input volume
        mask_vol ([D, H, W]): Mask volume
        direction (): Direction with same dimesnion order to input volume
        origin (): Origin with same dimesnion order to input volume
        spacing (): Spacing with same dimesnion order to input volume
        save_dir (str): Saving directory
        filename (str): Saving file name
    """
    os.makedirs(save_dir, exist_ok=True)

    vol = np.transpose(vol, (2, 1, 0))
    mask_vol = np.transpose(mask_vol, (2, 1, 0))
    spacing = np.array([spacing[1], spacing[2], spacing[0]])
    origin = np.array([origin[1], origin[2], origin[0]])
    
    raw_path = os.path.join(save_dir, f'{filename}.nrrd')
    seg_path = os.path.join(save_dir, f'{filename}.seg.nrrd')
    raw_nrrd_write(raw_path, vol, direction, spacing, origin, space)
    seg_nrrd_write(seg_path, mask_vol, direction, spacing, origin, space)

    
def seg_nrrd_write(filename, voxels, direction, spacing, origin, space='left-posterior-superior'):
    space_direction = direction * np.tile(spacing[:,np.newaxis], (1, 3))
    voxels = np.int32(voxels)
    seg_header = build_nrrd_seg_header(voxels, space_direction, origin, space)
    nrrd.write(f'{filename}.seg.nrrd', voxels, seg_header)
    # print('Seg nrrd data conversion completed.')


def raw_nrrd_write(filename, voxels, direction, spacing, origin, space='left-posterior-superior'):
    space_direction = direction * np.tile(spacing[:,np.newaxis], (1, 3))
    voxels = np.int32(voxels)
    nrrd_header = build_nrrd_header(voxels, space_direction, origin, space)
    nrrd.write(f'{filename}.nrrd', voxels, nrrd_header)
    # print('RAW nrrd data conversion completed.')


def build_nrrd_header(arr, direction, origin, space):
    header = {
        'type': 'unsigned char',
        'dimension': arr.ndim,
        'space': space,
        'sizes': arr.shape,
        'space directions': direction,
        'kinds': ['domain', 'domain', 'domain'],
        'endian': 'little',
        'encoding': 'gzip',
        'space origin': origin
    }
    return header


def build_nrrd_seg_header(arr, direction, origin, space, cmap=None):
    # slicer can decide color automatically
    if cmap is None:
        cmap = [(0.5, 0.7, 0), (0.7, 0.5, 0)]

    header = build_nrrd_header(arr, direction, origin, space)
    header.update(build_common_custom_field())
    
    nodule_ids = np.unique(arr)[1:]
    print(f'Nodule number {nodule_ids.size}')
    for idx, nodule_id in enumerate(nodule_ids):
        data = np.uint8(arr==nodule_id)
        color = cmap[idx%len(cmap)]
        seg_header = build_segment_custom_field(nodule_id, color, data)
        header.update(seg_header)
    return header


def build_common_custom_field():
    conversion_params = 'Collapse labelmaps|1|Merge the labelmaps \
                        into as few shared labelmaps as possible \
                        1 = created labelmaps will be shared if possible \
                        without overwriting each other.&Compute surface \
                        normals|1|Compute surface normals. 1 (default) = \
                        surface normals are computed. 0 = surface normals \
                        are not computed (slightly faster but produces less \
                        smooth surface display).&Crop to reference image \
                        geometry|0|Crop the model to the extent of reference \
                        geometry. 0 (default) = created labelmap will contain \
                        the entire model. 1 = created labelmap extent will be \
                        within reference image extent.&Decimation \
                        factor|0.0|Desired reduction in the total \
                        number of polygons. Range: 0.0 (no decimation)\
                        to 1.0 (as much simplification as possible). \
                        Value of 0.8 typically reduces data set size \
                        by 80% without losing too much details.&Fractional \
                        labelmap oversampling factor|1|Determines the \
                        oversampling of the reference image geometry. \
                        All segments are oversampled with the same value \
                        (value of 1 means no oversampling).&Joint \
                        smoothing|0|Perform joint smoothing.&Oversampling \
                        factor|1|Determines the oversampling of the \
                        reference image geometry. If it\'s a number, \
                        then all segments are oversampled with the \
                        same value (value of 1 means no oversampling). If it has the value "A", then automatic oversampling is calculated.&Reference image geometry|-0.64453125;0;0;166;0;-0.64453125;0;133.600006103516;0;0;2.49990081787109;74.0438003540039;0;0;0;1;0;511;0;511;0;130;|Image geometry description string determining the geometry of the labelmap that is created in course of conversion. Can be copied from a volume, using the button.&Smoothing factor|0.5|Smoothing factor. Range: 0.0 (no smoothing) to 1.0 (strong smoothing).&Threshold fraction|0.5|Determines the threshold that the closed surface is created at as a fractional value between 0 and 1.&'
    
    common_custom_field = {
        'Segmentation_ContainedRepresentationNames': 'Binary labelmap|',
        'Segmentation_ConversionParameters': conversion_params,
        'Segmentation_MasterRepresentation': 'Binary labelmap',
        'Segmentation_ReferenceImageExtentOffset': '0 0 0',
    }
    return common_custom_field


def build_segment_custom_field(id: int, color: tuple, data: np.array) -> dict:
    key = f'Segment{id}'
    color = f'{color[0]} {color[1]} {color[2]}'

    assert data.ndim == 3
    zs, ys, xs = np.where(data)
    extent = []
    for d in [xs, ys, zs]:
        extent.append(str(np.min(d)))
        extent.append(str(np.max(d)))

    segment_custom_field = {
        f'{key}_Color': color,
        f'{key}_ColorAutoGenerated': '0',
        f'{key}_Extent': ' '.join(extent),
        f'{key}_ID': f'ID_{id:03d}',
        f'{key}_LabelValue': id,
        f'{key}_Layer': '0',
        f'{key}_Name': f'Segment_{id:03d}',
        f'{key}_NameAutoGenerated': '0',
        f'{key}_Tags': 'Segmentation.Status:inprogress|TerminologyEntry:Segmentation category and type - 3D Slicer General Anatomy list~SCT^85756007^Tissue~SCT^272673000^Bone~^^~Anatomic codes - DICOM master list~^^~^^|',
    }
    return segment_custom_field

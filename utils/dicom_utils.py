import pydicom #for loading dicom
import SimpleITK as sitk #for loading mhd/raw
import os
import numpy as np
import scipy.ndimage

#DICOM: send path to any *.dcm file where containing dir has the other slices (dcm files), or path to the dir itself
#MHD/RAW: send path to the *.mhd file where containing die has the coorisponding *.raw file
def load_scan(path2scan):
    if (path2scan.split('.')[-1] == 'mhd') or (path2scan.split('.')[-1] == 'raw'):
        return load_mhd(path2scan)
    elif path2scan.split('.')[-1] == 'dcm':
        return load_dicom(os.path.split(path2scan)[0]) #pass containing directory
    elif os.listdir(path2scan)[0].split('.')[-1] == 'dcm':
        return load_dicom(path2scan)
    else:
        raise Exception('No valid scan [series] found in given file/directory')


def load_mhd(path2scan):
    itkimage = sitk.ReadImage(path2scan)
    scan = sitk.GetArrayFromImage(itkimage)
    spacing = np.flip(np.array(itkimage.GetSpacing()),axis=0)
    orientation = np.transpose(np.array(itkimage.GetDirection()).reshape((3, 3)))
    origin = np.flip(np.array(itkimage.GetOrigin()),axis=0) #origionally in yxz format (read xy in viewers but sanved as yx)
    return scan, spacing, orientation, origin, None #output in zyx format

def load_dicom(path2scan_dir):
    dicom_folder = path2scan_dir
    dcms = os.listdir(dicom_folder)
    first_slice_data = pydicom.read_file(path2scan_dir + '\\' + dcms[0])
    first_slice = first_slice_data.pixel_array
    orientation = np.transpose(first_slice_data.ImageOrientationPatient) #zyx format
    spacing_xy = np.array(first_slice_data.PixelSpacing, dtype=float)
    spacing_z = np.float(first_slice_data.SliceThickness)
    spacing = np.array([spacing_z, spacing_xy[1], spacing_xy[0]]) #zyx format

    scan = np.zeros((len(dcms),first_slice.shape[0],first_slice.shape[1]))
    raw_slices=[]
    indexes = []
    for dcm in dcms:
        slice_data = pydicom.read_file(dicom_folder + '\\' + dcm)
        slice_data.filename = dcm
        raw_slices.append(slice_data)
        indexes.append(float(slice_data.ImagePositionPatient[2]))
    indexes = np.array(indexes,dtype=float)

    raw_slices = [x for _, x in sorted(zip(indexes, raw_slices))]
    origin = np.array(raw_slices[0][0x00200032].value) #origin is assumed to be the image location of the first slice
    if origin is None:
        origin = np.zeros(3)
    else:
        origin = np.array([origin[2],origin[1],origin[0]]) #change from x,y,z to z,y,x

    for i, slice in enumerate(raw_slices):
        scan[i, :, :] = slice.pixel_array
    # for i, index in enumerate(indexes):
    #     for slice in raw_slices:
    #         if int(slice.InstanceNumber) == index:
    #             scan[i,:,:] = slice._pixel_data_numpy()
    return scan, spacing, orientation, origin, raw_slices


#point to directory of folders conting dicom scans only (subdirs only), runs aon all folders..
# ref directory is used to copy the m
def save_dicom(scan, origional_raw_slices, dst_directory): # \\dfds\\ format
    os.makedirs(dst_directory,exist_ok=True)
    for i, slice in enumerate(origional_raw_slices):
        slice.pixel_array.flat = scan[i,:,:].flat
        slice.PixelData = slice.pixel_array.tobytes()
        slice.save_as(os.path.join(dst_directory,slice.filename))#.dcmwrite(os.path.join(dst_directory,slice.filename),slice)

def scale_scan(scan,spacing,factor=1):
    resize_factor = factor * spacing
    new_real_shape = scan.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / scan.shape
    new_spacing = spacing / real_resize_factor
    scan_resized = scipy.ndimage.interpolation.zoom(scan, real_resize_factor, mode='nearest')
    return scan_resized, resize_factor

def scale_vox_coord(coord, spacing, factor=1):
    resize_factor = factor * spacing
    return (coord*resize_factor).astype(int)

def world2vox(world_coord, spacing, orientation, origin):
    world_coord = np.dot(np.linalg.inv(np.dot(orientation, np.diag(spacing))), world_coord - origin)
    if orientation[0, 0] < 0:
        vox_coord = (np.array([world_coord[0], world_coord[2], world_coord[1]])).astype(int)
    else:
        vox_coord = (np.array([world_coord[0], world_coord[1], world_coord[2]])).astype(int)
    return vox_coord
from config import * #user configurations
from keras.models import load_model
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = config['gpus']
from utils.equalizer import *
import pickle
import numpy as np
import time
import scipy.ndimage
from utils.dicom_utils import *
from utils.utils import *

# in this version: coords must be provided manually (for autnomaic candiate location selection, use[x])
# in this version: we scale the entire scan. For faster tampering, one should only scale the cube that is being tampred.
# in this version: dicom->dicom, dicom->numpy, mhd/raw->numpy supported

class scan_manipulator:
    def __init__(self,isInjector=True):
        print("===Init Tamperer===")
        self.scan = None
        self.malscan_resized = None # temporary storage for tamepring until time to scale back down
        self.isInjector = isInjector
        self.load_path = None
        self.m_zlims = config['mask_zlims']
        self.m_ylims = config['mask_ylims']
        self.m_xlims = config['mask_xlims']
        self.tamper_coords = []

        #load model and parameters
        if self.isInjector:
            self.model_path = config['modelpath_inject']
        else:
            self.model_path = config['modelpath_remove']

        #load generator
        print("Loading model")
        self.generator = load_model(os.path.join(self.model_path,"G_model.h5"))

        #load normalization params
        self.norm = np.load(os.path.join(self.model_path,'normalization.npy'))

        # load equalization params
        self.eq = histEq([], path = os.path.join(self.model_path,'equalization.pkl'))

    # loads dicom/mhd to be tampered
    # Provide path to a *.dcm file or the *mhd file. The contaitning folder should have the other slices)
    def load_target_scan(self, load_path):
        self.load_path = load_path
        print('Loading scan')
        self.scan, self.scan_spacing, self.scan_orientation, self.scan_origin, self.scan_raw_slices = load_scan(load_path)
        print("Scaling up scan...")
        self.scan_resized, self.resize_factor = scale_scan(self.scan, self.scan_spacing)
        self.mal_scan_resized = np.copy(self.scan_resized)

    # saves tampered scan as 'dicom' series or 'numpy' serialization
    def save_tampered_scan(self, save_dir, output_type='dicom'):
        if self.scan is None:
            print('Cannot save: load a target scan first.')
            return

        # scale scan back down and add noise touchups
        if len(self.tamper_coords) > 0:
            self._touch_up_scan()
            #self.tamper_coords.clear()

        print('Saving scan')
        if (output_type == 'dicom') and (self.load_path.split('.')[-1]=="mhd"):
            raise Exception('Save file error: mhd -> dicom conversion currently unsupported. Either supply a dicom scan or set the output type to numpy.')
            #save with same per slice metadata as source
        if output_type == "dicom":
            save_dicom(self.scan, origional_raw_slices=self.scan_raw_slices, dst_directory=save_dir)
        else:
            np.save(os.path.join(save_dir,'tampered_scan.np'),self.scan)
        print('Done.')


    # tamper loaded scan at given voxel (index) coordinate
    # coord: E.g. vox: slice_indx, y_indx, x_indx    world: -324.3, 23, -234
    def tamper(self, coord, isVox=True):
        if self.scan is None:
            print('Cannot tamper: load a target scan first.')
            return

        print('===Injecting Evidence===')
        if not isVox:
            coord = world2vox(coord, self.scan_spacing, self.scan_orientation, self.scan_origin)

        ### Scale coordinate
        vox_coord_s = scale_vox_coord(coord, self.scan_spacing)

        ### Cut Location
        print("Cutting out target region")
        clean_cube = cutCube(self.mal_scan_resized, vox_coord_s, config["cube_shape"])

        ### Normalize/Equalize Location
        print("Normalizing sample")
        clean_cube_eq = self.eq.equalize(clean_cube)
        clean_cube_norm = (clean_cube_eq - self.norm[0]) / ((self.norm[2] - self.norm[1]))

        ########  Inject Cancer   ##########

        ### Inject/Remove evidence
        if self.isInjector:
            print("Injecting evidence")
        else:
            print("Removing evidence")

        x = np.copy(clean_cube_norm)
        x[self.m_zlims[0]:self.m_zlims[1], self.m_xlims[0]:self.m_xlims[1], self.m_ylims[0]:self.m_ylims[1]] = 0
        x = x.reshape((1, config['cube_shape'][0], config['cube_shape'][1], config['cube_shape'][2], 1))
        x_mal = self.generator.predict([x])
        x_mal = x_mal.reshape(config['cube_shape'])

        ### De-Norm/De-equalize
        print("De-normalizing sample")
        x_mal[x_mal > .5] = .5  # fix boundry overflow
        x_mal[x_mal < -.5] = -.5
        mal_cube_eq = x_mal * ((self.norm[2] - self.norm[1])) + self.norm[0]
        mal_cube = self.eq.dequalize(mal_cube_eq)
        # Correct for pixel norm error
        # fix overflow
        bad = np.where(mal_cube > 2000)
        # mal_cube[bad] = np.median(clean_cube)
        for i in range(len(bad[0])):
            neiborhood = cutCube(mal_cube, np.array([bad[0][i], bad[1][i], bad[2][i]]), (np.ones(3)*5).astype(int),-1000)
            mal_cube[bad[0][i], bad[1][i], bad[2][i]] = np.mean(neiborhood)
        # fix underflow
        mal_cube[mal_cube < -1000] = -1000

        ### Paste Location
        print("Pasting sample into scan")
        self.mal_scan_resized = pasteCube(self.mal_scan_resized, mal_cube, vox_coord_s)
        self.tamper_coords.append(coord)
        print('Done.')


    def _touch_up_scan(self):
        ### Rescale
        print("Scaling down scan...")
        mal_scan, resize_factor = scale_scan(self.mal_scan_resized, 1 / self.scan_spacing)

        ### Noise Touch-ups
        print("Adding noise touch-ups...")
        for coord in self.tamper_coords:
            noise_map_dim = (config['cube_shape']*2).astype(int)
            ben_cube_ext = cutCube(self.scan, coord, noise_map_dim)
            mal_cube_ext = cutCube(mal_scan, coord, noise_map_dim)
            local_sample = cutCube(self.scan, coord, config["cube_shape"])

            # Init Touch-ups
            if self.isInjector:
                noisemap = np.random.randn(150, 200, 300) * np.std(local_sample[local_sample < -600]) * .6
                kernel_size = 3
                factors = sigmoid((mal_cube_ext + 700) / 70)
                k = kern01(mal_cube_ext.shape[0], kernel_size)
                for i in range(factors.shape[0]):
                    factors[i, :, :] = factors[i, :, :] * k
            else:
                noisemap = np.random.randn(150, 200, 200) * 30
                kernel_size = .1
                k = kern01(mal_cube_ext.shape[0], kernel_size)
                factors = None

            # Perform touch-ups
            if config['copynoise']:  # copying similar noise from hard coded location over this lcoation (usually more realistic)
                benm = cutCube(self.scan, np.array([int(self.scan.shape[0] / 2), int(self.scan.shape[1]*.43), int(self.scan.shape[2]*.27)]), noise_map_dim)
                x = np.copy(benm)
                x[x > -800] = np.mean(x[x < -800])
                noise = x - np.mean(x)
            else:  # gaussian interpolated noise is used
                rf = np.ones((3,)) * (60 / np.std(local_sample[local_sample < -600])) * 1.3
                np.random.seed(np.int64(time.time()))
                noisemap_s = scipy.ndimage.interpolation.zoom(noisemap, rf, mode='nearest')
                noise = noisemap_s[:noise_map_dim, :noise_map_dim, :noise_map_dim]
            mal_cube_ext += noise

            if self.isInjector:  # Injection
                final_cube_s = np.maximum((mal_cube_ext * factors + ben_cube_ext * (1 - factors)), ben_cube_ext)
            else: #Removal
                minv = np.min((np.min(mal_cube_ext), np.min(ben_cube_ext)))
                final_cube_s = (mal_cube_ext + minv) * k + (ben_cube_ext + minv) * (1 - k) - minv

            mal_scan = pasteCube(mal_scan, final_cube_s, coord)
        self.scan = mal_scan
        print('touch-ups complete')



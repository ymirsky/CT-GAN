# MIT License
# 
# Copyright (c) 2019 Yisroel Mirsky
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

if __name__ == '__main__':
    import argparse

    #Parse command line arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.description = 'CT-GAN medical evidence tamper pipeline. Uses a pre-trained injection/removal model to tamper '\
                         'evidence in a given 3D medical scan (mhd/raw or dicom series)'
    parser.epilog = 'To change other settings, check config.py\nNote, this version is significantly slower since it will scale (interpolate) the entire scan and not just the target cuboid. '+\
                    'For more information, please read our paper:\nCT-GAN: Malicious Tampering of 3D Medical Imagery using Deep Learning. \nYisroel Mirsky, Tom Mahler, Ilan Shelef, and Yuval Elovici'
    parser.add_argument('-t','--target',required=True,help="The path to the target scan to be tampered. The path should point to the directory containing a dicom series (*.dcm files) or a *.mhd file.")
    parser.add_argument('-d','--destination', required=True, help="The directory (path) to save the tampered scan.")
    parser.add_argument('-a','--action', choices=['inject','remove'], required=True, help="The directory (path) to save the tampered scan.")
    parser.add_argument('-c','--coord', required=True,nargs='*',help="The selected coordinate(s) in the target scan to inject evidence. You must provide one or more coordinates in z,y,x format with no spaces. \nExample (inject at two locations): python tamper.py -t patient.mhd -d outdir -c 123,324,401 53,201,441")
    parser.add_argument('-s', '--system', choices=['vox','world'], default='vox',help="Indicate the coordinate system of the supplied target coordinates: 'vox' or 'world'. Default is 'vox'.")
    parser.add_argument('-f', '--outformat',choices=['dicom','numpy'],default='dicom',help="The output format to save the tamepred scan: 'dicom' or 'numpy'. Default is 'dicom'.")
    args = parser.parse_known_args()[0]

    from procedures.attack_pipeline import *

    # Init pipeline
    injector = scan_manipulator()

    # Load target scan (provide path to dcm file/dir, or mhd file)
    injector.load_target_scan(load_path=args.target)#('path_to_target_scan')

    for coord in args.coord:
        coorda = np.array(coord.split(','),dtype=float)
        injector.tamper(coorda, action=args.action, isVox=(args.system=='vox'))

    # Save scan
    injector.save_tampered_scan(save_dir=args.destination,output_type=args.outformat) #output can be dicom iff input was dicom, otherwise only numpy save is supported

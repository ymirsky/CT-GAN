from procedures.attack_pipeline import *

# Init pipeline
injector = scan_manipulator(isInjector=True)

# Load target scan (provide path to dcm file/dir, or mhd file)
injector.load_target_scan('D:\\CT\\Experiments\\EXP2_fixed\\1796')#('path_to_target_scan')

# Inject at two locations (this version does not implement auto candidate location selection)
vox_coord1 = np.array([222,365,149]) #z, y , x
vox_coord2 = np.array([227,365,335])
injector.tamper(vox_coord1, isVox=True) #can supply realworld coord too
injector.tamper(vox_coord2, isVox=True)

# Save scan
injector.save_tampered_scan('C:\\tmp',output_type='dicom') #output can be dicom iff input was dicom, otherwise only numpy save is supported
#path_to_save_scan
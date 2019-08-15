from procedures.attack_pipeline import *

print('Removing Evidence...')

# Init pipeline
injector = scan_manipulator()

# Load target scan (provide path to dcm or mhd file)
injector.load_target_scan('path_to_target_scan')

# Inject at two locations (this version does not implement auto candidate location selection)
vox_coord1 = np.array([53,213,400]) #z, y , x (x-y should be flipped if the coordinates were obtained from an image viewer such as RadiAnt)
vox_coord2 = np.array([33,313,200])
injector.tamper(vox_coord1, action='inject', isVox=True) #can supply realworld coord too
injector.tamper(vox_coord2, action='inject', isVox=True)

# Save scan
injector.save_tampered_scan('path_to_save_scan',output_type='dicom') #output can be dicom or numpy

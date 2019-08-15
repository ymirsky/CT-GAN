from procedures.datasetBuilder import *

if __name__ == '__main__':
    # Init dataset builder for creating a dataset of evidence to inject
    print('Initializing Dataset Builder for Evidence Injection')
    builder = Extractor(is_healthy_dataset=False, parallelize=False)

    # Extract training instances
    # Source data location and save location is loaded from config.py
    print('Extracting instances...')
    builder.extract()

    print('Done.')
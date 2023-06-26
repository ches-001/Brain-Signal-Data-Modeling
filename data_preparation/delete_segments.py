import shutil, os, argparse

def delete_segments(root_dir):
    if not os.path.isdir(root_dir):
        return
    if "segments" in root_dir:
        return shutil.rmtree(root_dir)
    
    for sample_dir in os.listdir(root_dir):
        sample_dir = os.path.join(root_dir, sample_dir)
        delete_segments(sample_dir)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="\n%(asctime)s - %(levelname)s - %(message)s")

    nirs_dir = "data/samples/NIRS"
    eeg_dir = "data/samples/EEG"

    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    eeg_data_dir = os.path.join(script_dir, "../data/samples/EEG")
    nirs_data_dir = os.path.join(script_dir, "../data/samples/NIRS")

    # CLI Parsed parameters
    parser = argparse.ArgumentParser(description=f"Segments Generator")
    parser.add_argument("--eeg_path", type=str, default=eeg_data_dir, metavar='', help='EEG data path')
    parser.add_argument("--nirs_path", type=str, default=nirs_data_dir, metavar='', help='NIRS data path')

    args = parser.parse_args()

    eeg_dir = args.eeg_path
    nirs_dir = args.nirs_path

    try:
        logging.info("Deleting generated EEG segments")
        delete_segments(eeg_dir)
    except Exception as e:
        logging.error(e)

    try:
        logging.info("Deleting generated NIRS segments")
        delete_segments(nirs_dir)
    except Exception as e:
        logging.error(e)
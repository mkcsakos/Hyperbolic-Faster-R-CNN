from pathlib import Path
import requests
import yaml
import json
import glob as glob
import argparse
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.NOTSET)
mmdetection_logger = logging.getLogger('mmdetection')
download_weights_logger = mmdetection_logger.getChild('download_weights')
download_weights_logger.setLevel(logging.DEBUG)

if Path.cwd().name == 'mmdetection':
    repo_dir = Path('')
else:
    repo_dir = Path('mmdetection')

working_dir = 'test_folder'
weight_file = 'weights.json'
config_folder = 'configs'
checkpoints_folder = 'checkpoints'

root_meta_file_path = repo_dir.joinpath(config_folder)
weight_file_path = repo_dir.joinpath(working_dir, weight_file)


def download_weights(url, file_save_name):
    """
    Download weights for any model.
    :param url: Download URL for the weihgt file.
    :param file_save_name: String name to save the file on to disk.
    """

    checkpoints_directory_path = repo_dir.joinpath(working_dir, checkpoints_folder)
    checkpoints_directory_path.mkdir(parents = False, exist_ok = True)
    weights_file_path = checkpoints_directory_path.joinpath(file_save_name)

    if weights_file_path.exists():
        download_weights_logger.info(" File already present, skipping")
        return

    download_weights_logger.info(f"Downloading {file_save_name}")
    file = requests.get(url, stream=True)
    total_size = int(file.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)

    with weights_file_path.open('wb') as weights_file:
        for data in file.iter_content(block_size):
            progress_bar.update(len(data))
            weights_file.write(data)

    progress_bar.close()


def parse_meta_file():
    """
    Function to parse all the model meta files inside `mmdetection/configs`
    and return the download URLs for all available models.
    Returns:
        weights_dict: List containing URLs for all the downloadable models.
    """

    all_metal_file_paths = root_meta_file_path.glob('*/metafile.yml')
    weights_dict = {}

    for meta_file_path in all_metal_file_paths:
        with meta_file_path.open('r') as file:
            yaml_file = yaml.safe_load(file)

        try:
            models = yaml_file['Models']
        except:
            models = yaml_file

        for i in range(len(models)):
            model = models[i]

            try:
                weights_dict[model['Name']] = model['Weights']
            except:
                try:
                    weights_dict[model['Name']] = model['Results'][0]['Metrics']['Weights']
                except:
                    download_weights_logger.warning(f" Can't find Weights and Metrics for model: {model}")

    return weights_dict


def get_model(model_name):
    """
    Either downloads a model or loads one from local path if already
    downloaded using the weight file name (`model_name`) provided.
    :param model_name: Name of the weight file. Most likely in the format
        retinanet_ghm_r50_fpn_1x_coco. SEE `weights.txt` to know weight file
        name formats and downloadable URL formats.
    Returns:
        model: The loaded detection model.
    """

    weights_dict = load_weghts_from_file()
    download_url = None

    for model, weights in weights_dict.items():
        if isinstance(weights, list):
            weights = weights[0]
            download_weights_logger.info(f" There are multiple weight files for {model}. Selecting the first!")

        if model_name == model:
            download_weights_logger.info(f" Founds weights: {weights}\n")
            download_url = weights
            break

    assert download_url != None, f"{model_name} weight file not found!!!"

    download_weights(url=download_url, file_save_name=download_url.split('/')[-1])


def write_weights_to_file():
    """
    Write all the model URLs to `weights.txt` to have a complete list and choose one of them.
    """

    weights_dict = parse_meta_file()

    assert weights_dict
    assert repo_dir.joinpath(working_dir).is_dir()

    if weight_file_path.is_file():
        weight_file_path.unlink()

    with weight_file_path.open('w') as outfile:
        json.dump(weights_dict, outfile, indent=4)


def load_weghts_from_file():
    """
    Load all the model weights dictionary to memory
    """

    assert weight_file_path.exists()

    with weight_file_path.open('r') as weights_file:
        weights = json.load(weights_file)

    return weights



if __name__ == '__main__':
    write_weights_to_file()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-w', '--weights', default='faster-rcnn_r50_fpn_1x_coco',
        help='weight file name'
    )
    args = vars(parser.parse_args())
    get_model(args['weights'])

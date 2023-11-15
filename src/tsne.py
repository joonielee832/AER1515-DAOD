import argparse
import torch
from torchvision.models import resnet101, ResNet101_Weights
from torch.utils.data import DataLoader
from collections import OrderedDict
import os
import os.path as osp
import numpy as np
import random
from tqdm import tqdm
from tsnecuda import TSNE
import matplotlib.pyplot as plt

from tsne_utils import TSNEDataset

def load_all_cityscapes(root_dir):
    train_dir = osp.join(root_dir, 'leftImg8bit', 'train')
    cities = os.listdir(train_dir)
    filenames = []
    for city in cities:
        city_dir = osp.join(train_dir, city)
        files = os.listdir(city_dir)
        filenames.extend([osp.join(city_dir, file) for file in files])
    return filenames

def load_all_acdc(root_dir, filter_weather=False):
    rgb_dir = osp.join(root_dir, 'rgb_anon')
    filenames = []
    for weather in os.listdir(rgb_dir):
        weather_dir = osp.join(rgb_dir, weather)
        train_dir = osp.join(weather_dir, 'train')
        for img_dir in os.listdir(train_dir):
            imgs = os.listdir(osp.join(train_dir, img_dir))
            filenames.extend([osp.join(train_dir, img_dir, img) for img in imgs])
    return filenames

def load_all_cadc(root_dir):
    dates = os.listdir(root_dir)
    filenames = []
    for date in dates:
        date_dir = osp.join(root_dir, date)
        scenes = [_ for _ in os.listdir(date_dir) if _.startswith('0')]
        for scene in scenes:
            scene_dir = osp.join(date_dir, scene)
            frame_dir = osp.join(scene_dir, 'labeled')
            for cam in [c for c in os.listdir(frame_dir) if c.startswith('image')]:
                cam_dir = osp.join(frame_dir, cam, 'data')
                files = os.listdir(cam_dir)
                filenames.extend([osp.join(cam_dir, file) for file in files if file.endswith('.png')])
    return filenames

def extract_features(model, dataloader):
    all_features = torch.Tensor().to('cuda' if torch.cuda.is_available() else 'cpu')
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        if torch.cuda.is_available():
            batch = batch.to('cuda')
        with torch.no_grad():
            features = model(batch).squeeze()
        all_features = torch.cat([all_features, features], dim=0)
    return all_features

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/home/data/', help='Root directory of all datasets')
    parser.add_argument('--output', type=str, default='/home/results/', help='Output directory')
    parser.add_argument('--remove', action='store_true', help='Remove existing features', default=False)
    return parser.parse_args()

def main(root_dir, output_dir, remove=False):
    seed = 0
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    #? Load resnet101
    extractor = resnet101(weights=ResNet101_Weights.DEFAULT)
    extractor = torch.nn.Sequential(OrderedDict([*(list(extractor.named_children())[:-1])]))
    if torch.cuda.is_available():
        extractor = extractor.to('cuda')
    extractor.eval()
    
    #? Load all datasets
    batch_size = 128
    multiweather_filelist = load_all_cityscapes(osp.join(root_dir, 'cityscapes_multiweather'))
    # cadc_filelist = load_all_cadc(osp.join(root_dir, 'cadcd'))
    cityscapes_filelist = load_all_cityscapes(osp.join(root_dir, 'cityscapes'))
    acdc_filelist = load_all_acdc(osp.join(root_dir, 'acdc'))
    min_dataset_size = min([len(multiweather_filelist), len(cityscapes_filelist), len(acdc_filelist)])
    multiweather_filelist = multiweather_filelist[:min_dataset_size]
    # cadc_filelist = cadc_filelist[:min_dataset_size]
    cityscapes_filelist = cityscapes_filelist[:min_dataset_size]
    acdc_filelist = acdc_filelist[:min_dataset_size]
    
    multiweather_img_size = min(TSNEDataset.load_img(multiweather_filelist[0]).size)
    multiweather_dataset = TSNEDataset(multiweather_filelist, multiweather_img_size, center_crop=True)
    multiweather_dataloader = DataLoader(multiweather_dataset, batch_size=batch_size, shuffle=True)
    # cadc_dataset = TSNEDataset(cadc_filelist, multiweather_img_size, center_crop=True)
    # cadc_dataloader = DataLoader(cadc_dataset, batch_size=batch_size, shuffle=True)
    cityscapes_dataset = TSNEDataset(cityscapes_filelist, multiweather_img_size, center_crop=True)
    cityscapes_dataloader = DataLoader(cityscapes_dataset, batch_size=batch_size, shuffle=True)
    acdc_dataset = TSNEDataset(acdc_filelist, multiweather_img_size, center_crop=True)
    acdc_dataloader = DataLoader(acdc_dataset, batch_size=batch_size, shuffle=True)
    
    #? Extract features
    print('Extracting multiweather features...')
    multiweather_features_path = osp.join(output_dir, 'multiweather_features.npy')
    if not remove and osp.exists(multiweather_features_path):
        multiweather_features = np.load(multiweather_features_path)
    else:
        multiweather_features = extract_features(extractor, multiweather_dataloader).cpu().numpy()
        np.save(multiweather_features_path, multiweather_features)
    
    # print('Extracting CADC features...')
    # cadc_features_path = osp.join(output_dir, 'cadc_features.npy')
    # if not remove and osp.exists(cadc_features_path):
    #     cadc_features = np.load(cadc_features_path)
    # else:
    #     cadc_features = extract_features(extractor, cadc_dataloader).cpu().numpy()
    #     np.save(cadc_features_path, cadc_features)
    
    print('Extracting cityscapes features...')
    cityscapes_features_path = osp.join(output_dir, 'cityscapes_features.npy')
    if not remove and osp.exists(cityscapes_features_path):
        cityscapes_features = np.load(cityscapes_features_path)
    else:
        cityscapes_features = extract_features(extractor, cityscapes_dataloader).cpu().numpy()
        np.save(cityscapes_features_path, cityscapes_features)
    
    print('Extracting ACDC features...')
    acdc_features_path = osp.join(output_dir, 'acdc_features.npy')
    if not remove and osp.exists(acdc_features_path):
        acdc_features = np.load(acdc_features_path)
    else:
        acdc_features = extract_features(extractor, acdc_dataloader).cpu().numpy()
        np.save(acdc_features_path, acdc_features)
    
    # all_features = np.concatenate([multiweather_features[:min_dataset_size], 
    #                                cadc_features[:min_dataset_size],
    #                                acdc_features[:min_dataset_size]], axis=0)
    all_features = np.concatenate([multiweather_features[:min_dataset_size], 
                                   cityscapes_features[:min_dataset_size],
                                   acdc_features[:min_dataset_size]], axis=0)
    # all_labels = np.concatenate([np.zeros(min_dataset_size), 
    #                              np.ones(min_dataset_size),
    #                              np.ones(min_dataset_size)+1], axis=0)
    # perplexity_list = [500, 1000]
    perplexity_list = [400, 500, 600, 700]
    datasets = ['multiweather', 'cityscapes', 'acdc']
    colors = ['b', 'c', 'y', 'm', 'r']
    #? Remove all tsne plots in output directory
    for file in os.listdir(output_dir):
        if file.startswith('tsne') and file.endswith('.png'):
            os.remove(osp.join(output_dir, file))
    
    for perplexity in perplexity_list:
        tsne = TSNE(n_iter=10000, verbose=1, perplexity=perplexity, num_neighbors=1000) # 1000 seems to be the max before it errors out
        tsne_results = tsne.fit_transform(all_features)
        fig = plt.figure( figsize=(8,8))
        ax = fig.add_subplot(1, 1, 1, title='TSNE')

        # Create the scatter
        for i in range(tsne_results.shape[0] // min_dataset_size):
            ax.scatter(
                x=tsne_results[i*min_dataset_size:(i+1)*min_dataset_size,0],
                y=tsne_results[i*min_dataset_size:(i+1)*min_dataset_size,1],
                c=colors[i],
                s=0.5,
                label=f"{datasets[i]}")
        # Legend
        plt.legend()
        plt.savefig(osp.join(output_dir, f'tsne_{perplexity}.png'))

if __name__ == "__main__":
    args = parse_args()
    main(args.root, args.output, args.remove)
    
"""
TODO:
- tsne for cityscapes
- tsne for each weather type
"""
# Experiment runner for MobileFCMViTv3

import argparse

def main():
    parser = argparse.ArgumentParser(description='Run experiments for MobileFCMViTv3')
    parser.add_argument('--baseline', choices=['mobilenetv3', 'resnet50', 'vit'], default=None)
    parser.add_argument('--ablation', choices=['mobilevit', 'fcm', 'mobilefcmvitv3'], default=None)
    args = parser.parse_args()
    # ...experiment logic...
    print('Experiment setup:', args)

if __name__ == '__main__':
    main()

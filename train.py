import os
from argparse import ArgumentParser

import flash
import pytorch_lightning as pl
from torchvision.datasets import CIFAR100
from pytorch_lightning.callbacks import EarlyStopping
from flash.core.classification import Labels
from flash.core.finetuning import FreezeUnfreeze
from flash.core.data.transforms import merge_transforms
from flash.image.classification.transforms import default_transforms, train_default_transforms
from flash.image import ImageClassificationData, ImageClassifier

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument("--backbone", default="resnet18", type=str)
    parser.add_argument("--es_monitor", default="val_loss", type=str,)
    parser.add_argument("--es_stopping_threshold", default=None, type=float)
    parser.add_argument("--es_divergence_threshold", default=None, type=float)
    parser.add_argument("--es_check_finite",default=False, type=bool)
    parser.add_argument("--es_min_delta", default=0.0, type=float)
    parser.add_argument("--es_mode", default="min", type=str, choices=["min", "max"])
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()


    def unpack_torchvision(data):
        img_tuple = data['input']
        data['input'] = img_tuple[0]
        data['target'] = img_tuple[1]
        return data

    # 1. Load the data
    train_dataset = CIFAR100(train=True, download=True, root='.')

    datamodule = ImageClassificationData.from_datasets(train_dataset=train_dataset, 
                                                    train_transform=merge_transforms({"pre_tensor_transform":unpack_torchvision},
                                                                                        train_default_transforms([32, 32])),
                                                    val_transform=merge_transforms({"pre_tensor_transform":unpack_torchvision},
                                                                                        default_transforms([32, 32])),
                                                    val_split=.1,
                                                    batch_size=128)

    # 2. Build the model
    model = ImageClassifier(backbone=args.backbone, learning_rate=args.learning_rate, num_classes=100, serializer=Labels())

    # 3. Early stopping call back 
    early_stopping = EarlyStopping(
      monitor=args.es_monitor,
      mode=args.es_mode,
      min_delta=args.es_min_delta,
      stopping_threshold=args.es_stopping_threshold,
      divergence_threshold=args.es_divergence_threshold,
      check_finite=args.es_check_finite
    )

    # 4. Create the trainer
    trainer = flash.Trainer.from_argparse_args(args, callbacks=[early_stopping])

    # 5. Train the model
    trainer.finetune(model, datamodule=datamodule)

"""OpenPifPaf-based pose estimation module for TrackLab."""

import argparse
import sys
from typing import Any, List, Tuple
import torch
import numpy as np
import pandas as pd
from PIL import Image
from omegaconf.listconfig import ListConfig

import openpifpaf  # type: ignore


from tracklab.pipeline.imagelevel_module import ImageLevelModule
from tracklab.utils.coordinates import sanitize_keypoints, generate_bbox_from_keypoints

import logging

log = logging.getLogger(__name__)


def collate_images_anns_meta(
    batch: List[Tuple[Any, Any]],
) -> Tuple[Any, Tuple[Any, Any, Any]]:
    """Custom collate function for OpenPifPaf batch processing.

    Args:
        batch: List of tuples containing batch data.

    Returns:
        Collated batch data with indices, images, annotations, and metadata.
    """
    idxs = [b[0] for b in batch]
    batch_data = [b[1] for b in batch]
    anns = [b[-2] for b in batch]
    metas = [b[-1] for b in batch]

    processed_images = torch.utils.data.dataloader.default_collate(  # type: ignore
        [b[0] for b in batch_data]
    )
    idxs = torch.utils.data.dataloader.default_collate(idxs)  # type: ignore
    return idxs, (processed_images, anns, metas)


class OpenPifPaf(ImageLevelModule):
    """OpenPifPaf pose estimation module.

    This module integrates OpenPifPaf for bottom-up pose estimation,
    detecting all persons and their keypoints in images simultaneously.
    """

    collate_fn = collate_images_anns_meta
    input_columns: List[str] = []
    output_columns: List[str] = [
        "image_id",
        "id",
        "video_id",
        "category_id",
        "bbox_ltwh",
        "bbox_conf",
        "keypoints_xyc",
        "keypoints_conf",
    ]

    def __init__(self, cfg: Any, device: str, batch_size: int, **kwargs: Any) -> None:
        """Initialize the OpenPifPaf module.

        Args:
            cfg: Configuration object containing model settings.
            device: Device to run inference on.
            batch_size: Batch size for processing.
            **kwargs: Additional configuration parameters.
        """
        super().__init__(batch_size)
        self.cfg = cfg
        self.device = device
        self.id: int = 0

        if cfg.predict.checkpoint:
            old_argv = sys.argv
            sys.argv = self._hydra_to_argv(cfg.predict)
            tracklab_cli()
            predictor = openpifpaf.Predictor()
            sys.argv = old_argv

            self.model = predictor.model
            self.pifpaf_preprocess = predictor.preprocess
            self.processor = predictor.processor
            log.info(
                f"Loaded detection model from checkpoint: {cfg.predict.checkpoint}"
            )

    @torch.no_grad()
    def preprocess(
        self, image: Any, detections: pd.DataFrame, metadata: pd.Series
    ) -> Tuple[Any, Any, Any]:
        """Preprocess image for pose estimation.

        Args:
            image: Input image array.
            detections: Detection DataFrame (unused in preprocessing).
            metadata: Image metadata series.

        Returns:
            Tuple of processed image, annotations, and metadata.
        """
        image = Image.fromarray(image)
        processed_image, anns, meta = self.pifpaf_preprocess(image, [], {})
        return processed_image, anns, meta

    @torch.no_grad()
    def process(
        self,
        batch: Tuple[Any, Tuple[Any, Any, Any]],
        detections: pd.DataFrame,
        metadatas: pd.DataFrame,
    ) -> List[pd.Series]:
        """Process batch and extract pose detections.

        Args:
            batch: Preprocessed batch data.
            detections: Input detections DataFrame (unused).
            metadatas: Image metadata DataFrame.

        Returns:
            List of detection Series with pose information.
        """
        _, (processed_image_batch, _, metas) = batch
        pred_batch = self.processor.batch(
            self.model, processed_image_batch, device=self.device
        )
        detections_list = []
        for predictions, meta, (_, metadata) in zip(
            pred_batch, metas, metadatas.iterrows()
        ):
            for prediction in predictions:
                prediction = prediction.inverse_transform(meta)
                keypoints = sanitize_keypoints(prediction.data, meta["width_height"])
                bbox = generate_bbox_from_keypoints(
                    keypoints[keypoints[:, 2] > 0],
                    self.cfg.bbox.extension_factor,
                    meta["width_height"],
                )
                detections_list.append(
                    pd.Series(
                        dict(
                            image_id=metadata.name,
                            keypoints_xyc=keypoints,
                            keypoints_conf=prediction.score,
                            bbox_ltwh=bbox,
                            bbox_conf=prediction.score,
                            video_id=metadata.video_id,
                            category_id=1,  # `person` class in posetrack
                        ),
                        name=self.id,
                    )
                )
                self.id += 1

        return detections_list

    def train(self, *args: Any, **kwargs: Any) -> None:
        """Train the OpenPifPaf model.

        Args:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        old_argv = sys.argv
        sys.argv = self._hydra_to_argv(self.cfg.train)
        log.info("Starting training of the detection model")
        self.cfg.predict.checkpoint = openpifpaf.train.main()
        sys.argv = self._hydra_to_argv(self.cfg.predict)
        openpifpaf.predict.tracklab_cli()
        predictor = openpifpaf.Predictor()
        sys.argv = old_argv

        self.model = predictor.model
        self.pifpaf_preprocess = predictor.preprocess
        self.processor = predictor.processor
        log.info(
            f"Loaded trained detection model from file: {self.cfg.predict.checkpoint}"
        )

    @staticmethod
    def _hydra_to_argv(cfg: Any) -> List[str]:
        """Convert Hydra configuration to command line arguments.

        Args:
            cfg: Hydra configuration object.

        Returns:
            List of command line arguments.
        """
        new_argv = ["argv_from_hydra"]
        for k, v in cfg.items():
            new_arg = f"--{str(k)}"
            if isinstance(v, ListConfig):
                new_argv.append(new_arg)
                for item in v:
                    new_argv.append(f"{str(item)}")
            elif v is not None:
                new_arg += f"={str(v)}"
                new_argv.append(new_arg)
            else:
                new_argv.append(new_arg)
        return new_argv


from openpifpaf import decoder, logger, network, show, visualizer  # type: ignore


def tracklab_cli():  # for tracklab integration
    parser = argparse.ArgumentParser(
        prog="python3 -m openpifpaf.predict",
        usage="%(prog)s [options] images",
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version="OpenPifPaf {version}".format(version=openpifpaf.__version__),
    )

    decoder.cli(parser)
    logger.cli(parser)
    network.Factory.cli(parser)
    openpifpaf.Predictor.cli(parser)

    parser.add_argument(
        "--json-output",
        default=None,
        nargs="?",
        const=True,
        help="Whether to output a json file, "
        "with the option to specify the output path or directory",
    )
    parser.add_argument("--disable-cuda", action="store_true", help="disable CUDA")
    args = parser.parse_args()

    logger.configure(args, log)  # logger first

    # add args.device
    args.device = torch.device("cpu")
    args.pin_memory = False
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device("cuda")
        args.pin_memory = True
    log.info(
        "neural network device: %s (CUDA available: %s, count: %d)",
        args.device,
        torch.cuda.is_available(),
        torch.cuda.device_count(),
    )

    decoder.configure(args)
    network.Factory.configure(args)
    openpifpaf.Predictor.configure(args)

    return args

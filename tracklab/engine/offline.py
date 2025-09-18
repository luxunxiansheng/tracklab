import logging
from tqdm import tqdm

from tracklab.engine import TrackingEngine
from tracklab.utils.cv2 import cv2_load_image

log = logging.getLogger(__name__)


class OfflineTrackingEngine(TrackingEngine):
    def video_loop(self, tracker_state, video, video_id):
        for name, model in self.models.items():
            if hasattr(model, "reset"):
                model.reset()

        detections, image_pred = tracker_state.load()
        if len(self.module_names) == 0:
            return detections, image_pred
        image_filepaths = {idx: fn for idx, fn in image_pred["file_path"].items()}
        model_names = self.module_names
        log.info(f"🎯 Processing {len(model_names)} modules for video {video_id}")

        for model_name in tqdm(
            model_names, desc=f"Processing modules for video {video_id}", unit="module"
        ):
            if self.models[model_name].level == "video":
                detections = self.models[model_name].process(detections, image_pred)
                continue
            self.datapipes[model_name].update(image_filepaths, image_pred, detections)
            self.callback(
                "on_module_start",
                task=model_name,
                dataloader=self.dataloaders[model_name],
            )

            # Get total number of batches for progress bar
            total_batches = len(self.dataloaders[model_name])
            for batch_idx, batch in enumerate(
                tqdm(
                    self.dataloaders[model_name],
                    desc=f"Processing {model_name} batches",
                    unit="batch",
                    total=total_batches,
                    leave=False,
                )
            ):
                detections, image_pred = self.default_step(
                    batch, model_name, detections, image_pred
                )

            self.callback("on_module_end", task=model_name, detections=detections)
            if detections.empty:
                return detections, image_pred
        return detections, image_pred

from .module import Module, Pipeline, Skip
from .detectionlevel_module import DetectionLevelModule
from .evaluator import Evaluator
from .imagelevel_module import ImageLevelModule
from .videolevel_module import VideoLevelModule

# Import commonly used classes from subdirectories
from .dataset import *
from .eval import *
from .bbox_detector import *
from .pose_estimator import *
from .reid import *
from .team import *
from .tracklet_agg import *
from .jersey import *
from .calibration import *
from .track import *

__all__ = [
    # Base classes
    "Module",
    "Pipeline",
    "Skip",
    "DetectionLevelModule",
    "Evaluator",
    "ImageLevelModule",
    "VideoLevelModule",
    # Dataset classes
    "ExternalVideo",
    "DanceTrack",
    "MOT17",
    "MOT20",
    "SportsMOT",
    "Bee24",
    "PoseTrack18",
    "PoseTrack21",
    "SoccerNetGameState",
    "SoccerNetMOT",
    # Evaluator classes
    "PoseTrack18Evaluator",
    "PoseTrack21Evaluator",
    "TrackEvalEvaluator",
    # Detector classes
    # Detector classes
    "YOLOUltralytics",
    "RTMLibDetector",
    # 'RTDetr',  # Temporarily disabled due to transformers version compatibility
    "MMDetection",
    # Pose estimator classes
    # Pose estimator classes
    "YOLOUltralyticsPose",
    # 'VITPose',  # Temporarily disabled due to transformers version compatibility
    "RTMPose",
    "RTMO",
    # 'BottomUpMMPose',  # Temporarily disabled due to mmcv compatibility
    # 'TopDownMMPose',  # Temporarily disabled due to mmcv compatibility
    # 'OpenPifPaf',  # Temporarily disabled due to missing openpifpaf
    # ReID classes
    "PRTReId",
    # "KPReId",  # Temporarily disabled due to torchreid compatibility
    # Team classes
    "TrackletTeamClustering",
    "TrackletTeamSideLabeling",
    # Tracklet aggregation classes
    "MajorityVoteTracklet",
    # Jersey classes
    "MMOCR",
    "EasyOCR",
    "VotingTrackletJerseyNumber",
    # Calibration classes
    "NBJW_Calib_Keypoints",
    "NBJW_Calib",
    "PnLCalib_Keypoints",
    "PnLCalib",
    "BaselineCalibration",
    "Bbox2Pitch",
    "BaselinePitch",
    # "TVCalib_Segmentation",  # Temporarily disabled due to missing tvcalib
    # "TVCalib",  # Temporarily disabled due to missing tvcalib
    # Tracker classes
    "BotSORT",
    "BPBReIDStrongSORT",
    "ByteTrack",
    "DeepOCSORT",
    "OCSORT",
    "StrongSORT",
]

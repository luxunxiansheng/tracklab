from .module import Module, Pipeline, Skip
from .detectionlevel_module import DetectionLevelModule
from .imagelevel_module import ImageLevelModule
from .videolevel_module import VideoLevelModule

# Detector classes
from .bbox_detector import (
    YOLOUltralytics,
    RTMLibDetector,
    MMDetection,
)

# Pose estimator classes
from .pose_estimator import (
    YOLOUltralyticsPose,
    RTMPose,
    RTMO,
)

# ReID classes
from .reid import (
    PRTReId,
)

# Team classes
from .team import (
    TrackletTeamClustering,
    TrackletTeamSideLabeling,
)

# Tracklet aggregation classes
from .tracklet_agg import (
    MajorityVoteTracklet,
)

# Jersey classes
from .jersey import (
    MMOCR,
    EasyOCR,
    VotingTrackletJerseyNumber,
)

# Calibration classes
from .calibration import (
    NBJW_Calib_Keypoints,
    NBJW_Calib,
    PnLCalib_Keypoints,
    PnLCalib,
    BaselineCalibration,
    Bbox2Pitch,
    BaselinePitch,
)

# Tracker classes
from .track import (
    BotSORT,
    BPBReIDStrongSORT,
    ByteTrack,
    DeepOCSORT,
    OCSORT,
    StrongSORT,
)

__all__ = [
    # Base classes
    "Module",
    "Pipeline",
    "Skip",
    "DetectionLevelModule",
    "ImageLevelModule",
    "VideoLevelModule",
    # Detector classes
    "YOLOUltralytics",
    "RTMLibDetector",
    # 'RTDetr',  # Temporarily disabled due to transformers version compatibility
    "MMDetection",
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

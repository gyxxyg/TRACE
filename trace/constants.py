CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "./log_dir"

NUM_FRAMES = 8
MAX_FRAMES = 128
NUM_FRAMES_PER_SECOND = 1
Grids = [(2, 2), (1, 2), (1, 3), (1, 4), (2, 1), (3, 1), (4, 1)]

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"

#########################################################################################

# DEFAULT_TIME_PATCH_TOKEN = "<time_patch>"
# DEFAULT_SCORE_PATCH_TOKEN = "<score_patch>"

#########################################################################################





# MMODAL_TOKEN_INDEX = {"IMAGE": -200, "VIDEO": -201, "AUDIO": -202}
# MMODAL_INDEX_TOKEN = {v: k for k, v in MMODAL_TOKEN_INDEX.items()}
# MMODAL_START_TOKEN_INDEX = {"IMAGE": "<im_start>", "VIDEO": "<vid_start>", "AUDIO": "<ad_start>"}
# MMODAL_END_TOKEN_INDEX = {"IMAGE": "<im_end>", "VIDEO": "<vid_end>", "AUDIO": "<ad_end>"}


# DEFAULT_MMODAL_TOKEN = {"IMAGE": "<image>", "VIDEO": "<video>", "AUDIO": "<audio>"}
# DEFAULT_MMODAL_PATCH_TOKEN = {"IMAGE": "<im_patch>", "VIDEO": "<vid_patch>", "AUDIO": "<ad_patch>"}
# DEFAULT_MMODAL_START_TOKEN = {"IMAGE": "<Image>", "VIDEO": "<Video>", "AUDIO": "<ad_start>"}
# DEFAULT_MMODAL_END_TOKEN = {"IMAGE": "<\Image>", "VIDEO": "<\Video>", "AUDIO": "<\Audio>"}



#########################################################################################

MMODAL_TOKEN_INDEX = {"IMAGE": -200, "VIDEO": -201, "AUDIO": -202, "TIME": -203, "SCORE": -204, "SYNC": -205}
MMODAL_INDEX_TOKEN = {v: k for k, v in MMODAL_TOKEN_INDEX.items()}
MMODAL_START_TOKEN_INDEX = {"IMAGE": "<im_start>", "VIDEO": "<vid_start>", "AUDIO": "<ad_start>", "TIME": "<time_start>", "SCORE": "<score_start>"}
MMODAL_END_TOKEN_INDEX = {"IMAGE": "<im_end>", "VIDEO": "<vid_end>", "AUDIO": "<ad_end>", "TIME": "<time_end>", "SCORE": "<score_end>"}


DEFAULT_MMODAL_TOKEN = {"IMAGE": "<image>", "VIDEO": "<video>", "AUDIO": "<audio>", "TIME": "<time>", "SCORE": "<score>", "SYNC": "<sync>"}
DEFAULT_MMODAL_PATCH_TOKEN = {"IMAGE": "<im_patch>", "VIDEO": "<vid_patch>", "AUDIO": "<ad_patch>", "TIME": "<time_patch>", "SCORE": "<score_patch>"}
DEFAULT_MMODAL_START_TOKEN = {"IMAGE": "<Image>", "VIDEO": "<Video>", "AUDIO": "<ad_start>", "TIME": "<time_start>", "SCORE": "<score_start>"}
DEFAULT_MMODAL_END_TOKEN = {"IMAGE": "<\Image>", "VIDEO": "<\Video>", "AUDIO": "<\Audio>", "TIME": "<time_end>", "SCORE": "<score_end>"}

#########################################################################################
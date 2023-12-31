from exa.utils.logger import *
from exa.utils.logger import Logger
from exa.utils.metric_logger import Logging
from exa.utils.deploy import Deploy
from exa.utils.decoding_wrapper import real_time_decoding
from exa.quant.main import Quantize
from exa.inference.text_to_video import TextToVideo
from exa.inference.mmi import MultiModalInference
from exa.inference.kosmos import Kosmos
from exa.inference.hf import Inference
from exa.inference.gptq import GPTQInference
import os
import sentry_sdk
from exa.exa import exa

print(exa)

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "True"

telemetry = os.environ.get("TELEMETRY", "False")

if telemetry == "True":
    sentry_sdk.init(
        dsn="https://c85aae98407fc178f819e13cd8506084@o4504578305490944.ingest.sentry.io/4505943647191040",
        # Set traces_sample_rate to 1.0 to capture 100%
        # of transactions for performance monitoring.
        traces_sample_rate=1.0,
        # Set profiles_sample_rate to 1.0 to profile 100%
        # of sampled transactions.
        # We recommend adjusting this value in production.
        profiles_sample_rate=1.0,
    )
else:
    pass

# inference
# from exa.inference.ctransformers import CInference

# quant

# inference

# utils

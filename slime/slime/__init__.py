import os

if "ROCR_VISIBLE_DEVICES" in os.environ:
    _rocr = os.environ.pop("ROCR_VISIBLE_DEVICES")
    os.environ.setdefault("HIP_VISIBLE_DEVICES", _rocr)

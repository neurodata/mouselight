import boto3
from botocore import UNSIGNED
from botocore.client import Config
import os
from pathlib import Path

cwd = Path(os.path.abspath(__file__))
exp_dir = cwd.parents[1]
data_dir = os.path.join(exp_dir, "data")
figures_dir = os.path.join(exp_dir, "figures")
print(f"Downloading segments to {data_dir}")

# Make figures directory
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)

# Make data directory
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
# Make brains directories
brains = [1, 2]
for brain in brains:
    brain_name = "brain%d" % brain
    brain_dir = os.path.join(data_dir, brain_name)
    if not os.path.exists(brain_dir):
        os.makedirs(os.path.join(data_dir, brain_name))
    seg_dir = os.path.join(brain_dir, "segments_swc")
    trace_data_dir = os.path.join(brain_dir, "trace_data")
    if not os.path.exists(seg_dir):
        os.makedirs(seg_dir)
    if not os.path.exists(trace_data_dir):
        os.makedirs(trace_data_dir)
# Download segments
s3 = boto3.resource("s3", config=Config(signature_version=UNSIGNED))
bucket = s3.Bucket("open-neurodata")
for brain in brains:
    brain_name = "brain%d" % brain
    prefix = "brainlit/axon_geometry/" + brain_name + "/segments_swc"
    brain_dir = os.path.join(data_dir, brain_name)
    seg_dir = os.path.join(brain_dir, "segments_swc")
    seg_count = 0
    for _ in bucket.objects.filter(Prefix=prefix):
        seg_count += 1
    for i, seg_obj in enumerate(bucket.objects.filter(Prefix=prefix)):
        seg_name = os.path.basename(seg_obj.key)
        seg_path = os.path.join(seg_dir, seg_name.lower())
        bucket.download_file(seg_obj.key, seg_path)
        print("%s: downloaded segment %d/%d" % (brain_name, i + 1, seg_count))

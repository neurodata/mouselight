try:
    from urlparse import urlparse
except ImportError:
    from urllib.parse import urlparse
import contextlib
import joblib
import SimpleITK as sitk
import math
import boto3
import numpy as np
import os
from tqdm import tqdm


def get_bias_field(img, mask=None, scale=1.0, niters=[50, 50, 50, 50]):
    """Correct bias field in image using the N4ITK algorithm (http://bit.ly/2oFwAun)

    Parameters:
    ----------
    img : {SimpleITK.SimpleITK.Image}
        Input image with bias field.
    mask : {SimpleITK.SimpleITK.Image}, optional
        If used, the bias field will only be corrected within the mask. (the default is None, which results in the whole image being corrected.)
    scale : {float}, optional
        Scale at which to compute the bias correction. (the default is 0.25, which results in bias correction computed on an image downsampled to 1/4 of it's original size)
    niters : {list}, optional
        Number of iterations per resolution. Each additional entry in the list adds an additional resolution at which the bias is estimated. (the default is [50, 50, 50, 50] which results in 50 iterations per resolution at 4 resolutions)

    Returns
    -------
    SimpleITK.SimpleITK.Image
        Bias-corrected image that has the same size and spacing as the input image.
    """

    # do in case image has 0 intensities
    # add a small constant that depends on
    # distribution of intensities in the image
    minmaxfilter = sitk.MinimumMaximumImageFilter()
    minmaxfilter.Execute(img)
    minval = minmaxfilter.GetMinimum()
    img_rescaled = sitk.Cast(img, sitk.sitkFloat32) - minval + 1.0

    spacing = np.array(img_rescaled.GetSpacing()) / scale
    img_ds = imgResample(img_rescaled, spacing=spacing)
    img_ds = sitk.Cast(img_ds, sitk.sitkFloat32)

    # Calculate bias
    if mask is None:
        mask = sitk.Image(img_ds.GetSize(), sitk.sitkUInt8) + 1
        mask.CopyInformation(img_ds)
    else:
        if type(mask) is not sitk.SimpleITK.Image:
            mask_sitk = sitk.GetImageFromArray(mask)
            mask_sitk.CopyInformation(img)
            mask = mask_sitk
        mask = imgResample(mask, spacing=spacing)

    img_ds_bc = sitk.N4BiasFieldCorrection(img_ds, mask, 0.001, niters)
    bias_ds = img_ds_bc / sitk.Cast(img_ds, img_ds_bc.GetPixelID())

    # Upsample bias
    bias = imgResample(bias_ds, spacing=img.GetSpacing(), size=img.GetSize())

    return bias


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback:
        def __init__(self, time, index, parallel):
            self.index = index
            self.parallel = parallel

        def __call__(self, index):
            tqdm_object.update()
            if self.parallel._original_iterator is not None:
                self.parallel.dispatch_next()

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i : i + n]


def imgResample(img, spacing, size=[], useNearest=False, origin=None, outsideValue=0):
    """Resample image to certain spacing and size.

    Parameters:
    ----------
    img : {SimpleITK.SimpleITK.Image}
        Input 3D image.
    spacing : {list}
        List of length 3 indicating the voxel spacing as [x, y, z]
    size : {list}, optional
        List of length 3 indicating the number of voxels per dim [x, y, z] (the default is [], which will use compute the appropriate size based on the spacing.)
    useNearest : {bool}, optional
        If True use nearest neighbor interpolation. (the default is False, which will use linear interpolation.)
    origin : {list}, optional
        The location in physical space representing the [0,0,0] voxel in the input image. (the default is [0,0,0])
    outsideValue : {int}, optional
        value used to pad are outside image (the default is 0)

    Returns
    -------
    SimpleITK.SimpleITK.Image
        Resampled input image.
    """

    if origin is None:
        origin = [0] * 3
    if len(spacing) != img.GetDimension():
        raise Exception("len(spacing) != " + str(img.GetDimension()))

    # Set Size
    if size == []:
        inSpacing = img.GetSpacing()
        inSize = img.GetSize()
        size = [
            int(math.ceil(inSize[i] * (inSpacing[i] / spacing[i])))
            for i in range(img.GetDimension())
        ]
    else:
        if len(size) != img.GetDimension():
            raise Exception("len(size) != " + str(img.GetDimension()))

    # Resample input image
    interpolator = [sitk.sitkLinear, sitk.sitkNearestNeighbor][useNearest]
    identityTransform = sitk.Transform()

    return sitk.Resample(
        img,
        size,
        identityTransform,
        interpolator,
        origin,
        spacing,
        img.GetDirection(),
        outsideValue,
    )


# below code from https://stackoverflow.com/questions/42641315/s3-urls-get-bucket-name-and-path


class S3Url(object):
    """
    >>> s = S3Url("s3://bucket/hello/world")
    >>> s.bucket
    'bucket'
    >>> s.key
    'hello/world'
    >>> s.url
    's3://bucket/hello/world'

    >>> s = S3Url("s3://bucket/hello/world?qwe1=3#ddd")
    >>> s.bucket
    'bucket'
    >>> s.key
    'hello/world?qwe1=3#ddd'
    >>> s.url
    's3://bucket/hello/world?qwe1=3#ddd'

    >>> s = S3Url("s3://bucket/hello/world#foo?bar=2")
    >>> s.key
    'hello/world#foo?bar=2'
    >>> s.url
    's3://bucket/hello/world#foo?bar=2'
    """

    def __init__(self, url):
        self._parsed = urlparse(url, allow_fragments=False)

    @property
    def bucket(self):
        return self._parsed.netloc

    @property
    def key(self):
        if self._parsed.query:
            return self._parsed.path.lstrip("/") + "?" + self._parsed.query
        else:
            return self._parsed.path.lstrip("/")

    @property
    def url(self):
        return self._parsed.geturl()


def upload_file_to_s3(local_path, s3_bucket, s3_key):
    s3 = boto3.resource("s3")
    s3.meta.client.upload_file(local_path, s3_bucket, s3_key)


def download_file_from_s3(s3_bucket, s3_key, local_path):
    s3 = boto3.resource("s3")
    s3.meta.client.download_file(s3_bucket, s3_key, local_path)


def s3_object_exists(bucket, key):
    s3 = boto3.resource("s3")

    try:
        s3.Object(bucket, key).load()
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            # The object does not exist.
            return False
        else:
            # Something else has gone wrong.
            raise
    else:
        # The object does exist.
        return True


def download_terastitcher_files(s3_path, local_path):
    default_terastitcher_files = [
        "xml_import.xml",
        "xml_displcompute.xml",
        "xml_dislproj.xml",
        "xml_merging.xml",
        "xml_displthres.xml",
    ]
    s3 = boto3.resource("s3")
    s3_url = S3Url(s3_path)
    xml_paths = list(
        get_matching_s3_keys(s3_url.bucket, prefix=s3_url.key, suffix="xml")
    )
    xml_paths = [i for i in xml_paths if i in default_terastitcher_files]
    if len(xml_paths) == 0:
        # xml files were not at s3_path
        return False
    # download xml results to local_path
    for i in tqdm(xml_paths, desc="downloading xml files from S3"):
        fname = i.split("/")[-1]
        s3.meta.client.download_file(s3_url.bucket, i, f"{local_path}/{fname}")

    return True


# code from https://alexwlchan.net/2019/07/listing-s3-keys/
def get_matching_s3_keys(bucket, prefix="", suffix=""):
    """
    Generate the keys in an S3 bucket.

    :param bucket: Name of the S3 bucket.
    :param prefix: Only fetch keys that start with this prefix (optional).
    :param suffix: Only fetch keys that end with this suffix (optional).
    """
    s3 = boto3.client("s3")
    kwargs = {"Bucket": bucket, "Prefix": prefix}
    while True:
        resp = s3.list_objects_v2(**kwargs)
        try:
            resp["Contents"]
        except Exception as e:
            print(e)
            return None
        for obj in resp["Contents"]:
            key = obj["Key"]
            if key.endswith(suffix):
                yield key

        try:
            kwargs["ContinuationToken"] = resp["NextContinuationToken"]
        except KeyError:
            break


# below code from https://github.com/boto/boto3/issues/358#issuecomment-372086466
from awscli.clidriver import create_clidriver


def aws_cli(*cmd):
    old_env = dict(os.environ)
    try:

        # Environment
        env = os.environ.copy()
        env["LC_CTYPE"] = "en_US.UTF"
        os.environ.update(env)

        # Run awscli in the same process
        exit_code = create_clidriver().main(*cmd)

        # Deal with problems
        if exit_code > 0:
            raise RuntimeError("AWS CLI exited with code {}".format(exit_code))
    finally:
        os.environ.clear()
        os.environ.update(old_env)


def get_reorientations(in_orient, out_orient):
    """
    Gets reordering and flips require to convert from  in_orient to out_orient
    """

    dimension = len(in_orient)
    if len(in_orient) != dimension:
        raise Exception("in_orient must be a string of length {0}.".format(dimension))
    if len(out_orient) != dimension:
        raise Exception("out_orient must be a string of length {0}.".format(dimension))
    in_orient = str(in_orient).lower()
    out_orient = str(out_orient).lower()

    inDirection = ""
    outDirection = ""
    orientToDirection = {"r": "r", "l": "r", "s": "s", "i": "s", "a": "a", "p": "a"}
    for i in range(dimension):
        try:
            inDirection += orientToDirection[in_orient[i]]
        except BaseException:
            raise Exception("in_orient '{0}' is invalid.".format(in_orient))

        try:
            outDirection += orientToDirection[out_orient[i]]
        except BaseException:
            raise Exception("out_orient '{0}' is invalid.".format(out_orient))

    if len(set(inDirection)) != dimension:
        raise Exception("in_orient '{0}' is invalid.".format(in_orient))
    if len(set(outDirection)) != dimension:
        raise Exception("out_orient '{0}' is invalid.".format(out_orient))

    order = []
    flip = []
    for i in range(dimension):
        j = inDirection.find(outDirection[i])
        order += [j]
        flip += [-1 if in_orient[j] != out_orient[i] else 1]
    return order, flip


def start_ec2_instance(instance_id, instance_type):
    # get ec2 client
    ec2 = boto3.resource("ec2")

    # stop instance in case it is running
    ec2.meta.client.stop_instances(InstanceIds=[instance_id])
    waiter = ec2.meta.client.get_waiter("instance_stopped")
    waiter.wait(InstanceIds=[instance_id])
    # make sure instance is the right type
    ec2.meta.client.modify_instance_attribute(
        InstanceId=instance_id, Attribute="instanceType", Value=instance_type
    )
    # start instance
    ec2.meta.client.start_instances(InstanceIds=[instance_id])
    # wait until instance is started up
    waiter = ec2.meta.client.get_waiter("instance_status_ok")
    waiter.wait(InstanceIds=[instance_id])
    # get instance ip address
    instance = ec2.Instance(instance_id)
    return instance.public_ip_address


def calc_hierarchy_levels(img_size, lowest_res=1024):
    max_xy = max(img_size[0:1])
    # we add one because 0 is included in the number of downsampling levels
    num_levels = max(1, math.ceil(math.log(max_xy / lowest_res, 2)) + 1)
    return num_levels

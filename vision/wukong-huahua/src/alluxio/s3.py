import boto.s3.connection
import os
import logging
# from myapp import app
from src.alluxio.hw_obs import cube_bucket

# logging = app.logger

alluxio_host = "alluxio-proxy.infra"


def insure_s3_bucket():
    # return
    if os.getenv("STORAGE_MEDIA") != "MINIO":
        return

    alluxioConn = boto.connect_s3(
        aws_access_key_id='',
        aws_secret_access_key='',
        # host='alluxio-master-0.infra',
        host=alluxio_host,
        port=39999,
        path='/api/v1/s3',
        is_secure=False,
        calling_format=boto.s3.connection.OrdinaryCallingFormat(),
    )

    b = alluxioConn.lookup(cube_bucket)
    if b is None:
        logging.info("bucket " + cube_bucket + " 不存在， 现在新建")
        alluxioConn.create_bucket(cube_bucket)
        return alluxioConn.get_bucket(cube_bucket)
    else:
        # logging.info("bucket publish-data 确认存在")
        return b


s3_bucket = insure_s3_bucket()


def send_key_to(local_file_name, s3_key_name):
    key = s3_bucket.new_key(s3_key_name)
    with open(local_file_name, "rb") as f:
        try:
            key.set_contents_from_file(f)
        except:
            return


def send_directory_to(local_directory, s3_directory_name):
    for item in os.scandir(local_directory):
        p = item.path.replace(local_directory, s3_directory_name)
        if item.is_file():
            logging.info("upload file to: " + p)
            send_key_to(item.path, p)
        elif item.is_dir():
            send_directory_to(item.path, p)


def copy_dir(source_dir, dest_dir, only_image):
    rps = s3_bucket.list(prefix=source_dir, delimiter="")
    for r in rps:
        source_key_name = r.name
        dest_key_name = source_key_name.replace(source_dir, dest_dir)
        if only_image and not is_image(source_key_name):
            continue
        if source_key_name[-1] == "/":
            continue
        s3_bucket.copy_key(dest_key_name, cube_bucket, source_key_name)


def is_image(url):
    if url.endswith('.jpg') or url.endswith('png') or url.endswith('jpeg'):
        return True
    return False

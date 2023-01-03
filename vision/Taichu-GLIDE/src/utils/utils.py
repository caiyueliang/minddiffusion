import os
import logging

logger = logging.getLogger()


def print_dir(root_dir):
    logger.info("[print_dir] start : {} ...".format(root_dir))
    for parent, _, fileNames in os.walk(root_dir):
            for name in fileNames:
                if name.startswith('.'):  # 去除隐藏文件
                    continue
                else:
                    logger.info("{}, {}".format(parent, name))
    logger.info("[print_dir] end ...")

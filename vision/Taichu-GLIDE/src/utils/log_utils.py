#!/usr/bin/env python3
# coding=utf-8
"""
Author: caiyueliang
since: 2022-11-24 10:41:22
LastTime: 2022-11-24 10:43:23
LastAuthor: caiyueliang
message:
Copyright (c) 2022 Wuhan Artificial Intelligence Research. All Rights Reserved
"""
import logging
import time
import os
import fcntl

from logging.handlers import TimedRotatingFileHandler


class MultiCompatibleTimedRotatingFileHandler(TimedRotatingFileHandler):

    def doRollover(self):
        if self.stream:
            self.stream.close()
            self.stream = None
        # get the time that this sequence started at and make it a TimeTuple
        currentTime = int(time.time())
        dstNow = time.localtime(currentTime)[-1]
        t = self.rolloverAt - self.interval
        if self.utc:
            timeTuple = time.gmtime(t)
        else:
            timeTuple = time.localtime(t)
            dstThen = timeTuple[-1]
            if dstNow != dstThen:
                if dstNow:
                    addend = 3600
                else:
                    addend = -3600
                timeTuple = time.localtime(t + addend)
        dfn = self.baseFilename + "." + time.strftime(self.suffix, timeTuple)
        # 兼容多进程并发 LOG_ROTATE
        if not os.path.exists(dfn):
            f = open(self.baseFilename, 'a')
            fcntl.lockf(f.fileno(), fcntl.LOCK_EX)
            if not os.path.exists(dfn):
                os.rename(self.baseFilename, dfn)  # 释放锁 释放老log句柄
        if self.backupCount > 0:
            for s in self.getFilesToDelete():
                os.remove(s)
        if not self.delay:
            self.stream = self._open()
        newRolloverAt = self.computeRollover(currentTime)
        while newRolloverAt <= currentTime:
            newRolloverAt = newRolloverAt + self.interval
        # If DST changes and midnight or weekly rollover, adjust for this.
        if (self.when == 'MIDNIGHT' or self.when.startswith('W')) and not self.utc:
            dstAtRollover = time.localtime(newRolloverAt)[-1]
            if dstNow != dstAtRollover:
                if not dstNow:  # DST kicks in before next rollover, so we need to deduct an hour
                    addend = -3600
                else:  # DST bows out before next rollover, so we need to add an hour
                    addend = 3600
                newRolloverAt += addend
        self.rolloverAt = newRolloverAt


def parse_log_level(log_level: str):
    log_level = log_level.upper()
    if log_level == "DEBUG":
        return logging.DEBUG
    elif log_level == "INFO":
        return logging.INFO
    elif log_level == "WARN" or log_level == "WARNING":
        return logging.WARNING
    elif log_level == "ERROR":
        return logging.ERROR
    else:
        return logging.DEBUG


def set_log_file(logger_obj: logging.Logger, filename: str = None, backup_count: int = 7) -> None:
    """set log info """
    # 文件格式
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(pathname)s:%(lineno)d - [PID:%(process)d] %(message)s")

    # # 写到控制台
    # console_handler = logging.StreamHandler()
    # console_handler.setFormatter(formatter)
    # console_handler.setLevel(logger_obj.level)
    # logger_obj.addHandler(console_handler)

    # 写到文件
    if filename:
        log_dir = os.path.dirname(os.path.abspath(filename))
        os.makedirs(log_dir, exist_ok=True)

        file_handler = MultiCompatibleTimedRotatingFileHandler(
            filename=filename, when='midnight', interval=1, backupCount=backup_count)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logger_obj.level)

        # 设置到logger对象中
        logger_obj.addHandler(file_handler)


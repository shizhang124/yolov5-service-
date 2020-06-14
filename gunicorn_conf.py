#!/usr/bin/env python
#coding:utf-8
# gunicorn.py
import logging
import logging.handlers
from logging.handlers import WatchedFileHandler
import os
import multiprocessing

bind = '0.0.0.0:8000'      #绑定ip和端口号
backlog = 512                #监听队列
#chdir = '/home/test/server/bin'  #gunicorn要切换到的目的工作目录
timeout = 30      #超时
worker_class = 'sync' #使用gevent模式，还可以使用sync 模式，默认的是sync模式
#worker_class = 'gevent' #使用gevent模式，还可以使用sync 模式，默认的是sync模式

daemon = True
workers = 12
raw_env = ["OMP_NUM_THREADS=1"]
#workers = multiprocessing.cpu_count() * 2 + 1    #进程数
#threads = 2 #指定每个进程开启的线程数
loglevel = 'info' #日志级别，这个日志级别指的是错误日志的级别，而访问日志的级别无法设置
access_log_format = '%(t)s %(p)s %(h)s "%(r)s" %(s)s %(L)s %(b)s %(f)s" "%(a)s"'    #设置gunicorn访问日志格式，错误日志无法设置
debug=True

"""
其每个选项的含义如下：
h          remote address
l          '-'
u          currently '-', may be user name in future releases
t          date of the request
r          status line (e.g. ``GET / HTTP/1.1``)
s          status
b          response length or '-'
f          referer
a          user agent
T          request time in seconds
D          request time in microseconds
L          request time in decimal seconds
p          process ID
"""
accesslog = "./logs/gunicorn_access.log"      #访问日志文件
errorlog = "./logs/gunicorn_error.log"        #错误日志文件
pidfile = "./logs/gunicorn.pid"


def on_starting(server):
    """
    Attach a set of IDs that can be temporarily re-used.
    Used on reloads when each worker exists twice.
    """
    server._worker_id_overload = set()


def nworkers_changed(server, new_value, old_value):
    """
    Gets called on startup too.
    Set the current number of workers.  Required if we raise the worker count
    temporarily using TTIN because server.cfg.workers won't be updated and if
    one of those workers dies, we wouldn't know the ids go that far.
    """
    server._worker_id_current_workers = new_value


def _next_worker_id(server):
    """
    If there are IDs open for re-use, take one.  Else look for a free one.
    """
    if server._worker_id_overload:
        return server._worker_id_overload.pop()

    in_use = set(w._worker_id for w in server.WORKERS.values() if w.alive)
    free = set(range(1, server._worker_id_current_workers + 1)) - in_use

    return free.pop()


def on_reload(server):
    """
    Add a full set of ids into overload so it can be re-used once.
    """
    server._worker_id_overload = set(range(1, server.cfg.workers + 1))


def pre_fork(server, worker):
    """
    Attach the next free worker_id before forking off.
    """
    worker._worker_id = _next_worker_id(server)


def post_fork(server, worker):
    """
    Put the worker_id into an env variable for further use within the app.
    """
    os.environ["APP_WORKER_ID"] = str(worker._worker_id)

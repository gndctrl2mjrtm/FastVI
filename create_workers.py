import fcntl
import os
import socket
import struct
import time
import subprocess
import random

import ray


def get_ip_address(ifname):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return socket.inet_ntoa(fcntl.ioctl(
        s.fileno(),
        0x8915,
        struct.pack('256s', bytes(ifname[:15], 'utf-8'))
    )[20:24])


worker_task = """
ray start --redis-address={}
sleep {}
"""


@ray.remote
def get_workers():
    time.sleep(1e-4)
    return ray.services.get_node_ip_address()


class RayCluster(object):

    def __init__(self, port: int = None, n_workers: int = None, worker_time: int = 3600,
                 verbose: bool = True, ifname='eno1', stop_existing=True):
        if port is None:
            port = 44278-random.randint(0,100)
        else:
            try:
                int(port)
            except Exception:
                raise TypeError("Invalid port arg: {}".format(port))
        if n_workers is None:
            self.n_workers = 0
        else:
            assert isinstance(n_workers, int)
            self.n_workers = n_workers
        assert isinstance(verbose, bool)
        self.stop_existing = stop_existing
        self.ifname = ifname
        self.verbose = verbose
        self.worker_time = worker_time
        self.port = port
        self.host = get_ip_address(self.ifname)
        self.addr = '{}:{}'.format(self.host, self.port)
        worker_task_config = worker_task.format(self.addr, self.worker_time)
        self.worker_file = 'ray_worker_script'
        if os.path.exists(self.worker_file):
            print("Warning, worker file already exists: {}".format(self.worker_file))
        with open(self.worker_file, 'w') as f:
            f.write(worker_task_config)

    def _start_ray_head(self):
        if self.stop_existing:
            os.system('ray stop')
        if ray.is_initialized():
            if not self.stop_existing:
                raise Exception("Ray is already initialized")

        if self.verbose:
            print("Starting ray cluster on {}:{}".format(self.host, self.port))
        cmd = 'ray start --head --redis-port={}'.format(self.port)
        os.system(cmd)
        ray.init(redis_address=self.addr)
        print("IT CONNECTED")

    def _start_ray_workers(self):
        if self.verbose:
            print("Submitting {} worker jobs".format(self.n_workers))
        for i in range(self.n_workers):
            os.system('qsub {}'.format(self.worker_file))

    def get_worker_ips(self):
        return set(ray.get([get_workers.remote() for _ in range(1000)]))

    def wait_for_workers(self):
        current_n_workers = 0
        while True:
            ips = self.get_worker_ips()
            if self.verbose:
                if len(ips) > current_n_workers:
                    print("Current number of workers: {}".format(current_n_workers))
                    current_n_workers = len(ips)
                if current_n_workers >= self.n_workers+1:
                    break
            time.sleep(1e-3)

    def init_cluster(self, wait=True):
        self._start_ray_head()
        self._start_ray_workers()
        self.wait_for_workers()

if __name__ == "__main__":
    # rc = RayCluster(ifname='wlp2s0')
    rc = RayCluster(ifname='eno1')
    rc.init_cluster()

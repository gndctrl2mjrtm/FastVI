#!/bin/env/python
# -*- encoding: utf-8 -*-
"""

"""
from __future__ import print_function, division
import random
import socket
import struct
import fcntl
import time
import ray
import os

__author__ = "Stephen Offer"


def get_ip_address(ifname: str) -> str:
    """
    Get current ip address
    Credit: https://stackoverflow.com/questions/24196932/
        how-can-i-get-the-ip-address-of-eth0-in-python
    :param ifname: Network name (i.e. eno1)
    :return: str
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return socket.inet_ntoa(fcntl.ioctl(s.fileno(),
        0x8915,struct.pack('256s', bytes(ifname[:15], 'utf-8'))
        )[20:24])


# Script to send to qsub
worker_task = """
ray start --redis-address={}
sleep {}
"""


@ray.remote
def get_workers():
    """
    :return: list of connected ip addresses
    """
    time.sleep(1e-4)
    return ray.services.get_node_ip_address()


class RayCluster(object):

    def __init__(self, port: int = None, n_workers: int = None, worker_time: int = 3600,
                 verbose: bool = True, ifname='eno1', stop_existing: bool = True):
        """
        Simple Python API for starting Ray clusters on PBS/Torque clusters

        :param port: Redis port number
        :param n_workers: Number of worker nodes
        :param worker_time: The walltime for worker nodes in seconds
        :param verbose: Verbose (True/False)
        :param ifname: The network name (i.e. eno1)
        :param stop_existing: Stop existing Ray clusters on that node (True/False)
        """
        if port is None:
            # TODO: Verify port connectivity
            port = 44278 - random.randint(0, 100)
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

        # TODO: User argument verification
        assert isinstance(verbose, bool)
        self.stop_existing = stop_existing
        # TODO: Verify that ifname exists
        self.ifname = ifname
        self.verbose = verbose
        # TODO: warn user if walltime is low
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
        """
        Spin up a ray cluster head node and connect locally
        :return: None
        """
        # TODO: Subprocessing parsing for connectivity verification
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

    def _start_ray_workers(self):
        """
        Spin up the worker nodes
        :return: None
        """
        # TODO: Remove existing Ray worker jobs through qdel
        if not ray.is_initialized:
            print("Warning, Ray is not initialized on current node.")
        if self.verbose:
            print("Submitting {} worker jobs".format(self.n_workers))
        for i in range(self.n_workers):
            # TODO: Add custom Qsub resource allocation args
            os.system('qsub {}'.format(self.worker_file))

    @staticmethod
    def get_worker_ips():
        """
        Get a set of the current ip addresses that are connected
        :return: set
        """
        return set(ray.get([get_workers.remote() for _ in range(1000)]))

    def wait_for_workers(self):
        """
        Wait for all of the worker nodes to be connected
        :return: None
        """
        # TODO: Add timeout arg
        current_n_workers = 0
        while True:
            ips = self.get_worker_ips()
            if self.verbose:
                if len(ips) > current_n_workers:
                    current_n_workers = len(ips)
                    print("Current number of workers: {}".format(current_n_workers))
                if current_n_workers >= self.n_workers + 1:
                    break
            time.sleep(1e-3)

    def init_cluster(self, wait: bool = True):
        """
        Initialize the Ray cluster
        :param wait: Whether to wait for the worker nodes (True/False)
        :return:
        """
        self._start_ray_head()
        self._start_ray_workers()
        if wait:
            self.wait_for_workers()


if __name__ == "__main__":
    # rc = RayCluster(ifname='wlp2s0')
    rc = RayCluster(ifname='eno1')
    rc.init_cluster()

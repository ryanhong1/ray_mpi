import numpy as np
from mpi4py import MPI
import random

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
local_array = [rank] * random.randint(2, 5)
print("rank: {}, local_array: {}".format(rank, local_array))
sendbuf = np.array(local_array)

sendcounts = comm.gather(len(sendbuf), root=0)

#sendcounts = np.array(comm.gather(len(sendbuf), 0))
recvbuf = None
if rank == 0:
	print("sendcounts: {}, total: {}".format(sendcounts, sum(sendcounts)))
	recvbuf = np.empty(sum(sendcounts), dtype=int)
comm.Gatherv(sendbuf=sendbuf, recvbuf=(recvbuf, sendcounts), root=0)
if rank == 0:
	print("Gathered array: {}".format(recvbuf))
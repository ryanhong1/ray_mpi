from mpi4py import MPI
import math

comm = MPI.COMM_WORLD
size  = comm.Get_size()
rank = comm.Get_rank()

#This is a comment
#starting value of x
x=-1

#step size  CHANGE THIS
dx=0.0000001

iters=int(2/dx)
N = iters // size + (iters % size > rank)
start = comm.scan(N)-N

print (rank, start, N, size)


#the sum of all the areas - start at 0
A=0.
x=x+dx*start 
#now for a loop to go through and add up all the 
#tiny rectangle areas
for i in range(N):
  #each area is sqrt(1-x**2)*dx
  #add this area to the existing area
	A=A+math.sqrt(1-x**2)*dx
  #now move x forward by an amount dx
	x=x+dx
        

#end the loop
#pi is twice the area
if rank==0:
    tpi=2*A
    error = abs(tpi - math.pi)
    print ("pi is approximately %.16f, "
                "error is %.16f" % (tpi, error))
    

#약간 이런거임
#반원이 있으면 그걸 size 만큼 가로로 나눔. 그리고 iteration, 즉 총 반복 횟수를 rank마다 고루 분배해서 동시에 돌리는 거임.
#근데 각 rank가 같은 좌표를 돌리면 안되잖음? 그래서 start, 즉 시작 좌표 값을 다르게 해서 각 rank 마다 거기서 시작해서, 
#  N번 - 즉 자기에게 주어진 횟수만큼 반복하게 끝내게 하는 거임
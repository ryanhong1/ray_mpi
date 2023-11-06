from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt



def normalize(vector):
    return vector / np.linalg.norm(vector)

def reflected(vector, axis):
    return vector - 2 * np.dot(vector, axis) * axis

def sphere_intersect(center, radius, ray_origin, ray_direction):
    b = 2 * np.dot(ray_direction, ray_origin - center)
    c = np.linalg.norm(ray_origin - center) ** 2 - radius ** 2
    delta = b ** 2 - 4 * c
    if delta > 0:
        t1 = (-b + np.sqrt(delta)) / 2
        t2 = (-b - np.sqrt(delta)) / 2
        if t1 > 0 and t2 > 0:
            return min(t1, t2)
    return None

def nearest_intersected_object(objects, ray_origin, ray_direction):
    distances = [sphere_intersect(obj['center'], obj['radius'], ray_origin, ray_direction) for obj in objects]
    nearest_object = None
    min_distance = np.inf
    for index, distance in enumerate(distances):
        if distance and distance < min_distance:
            min_distance = distance
            nearest_object = objects[index]
    return nearest_object, min_distance

def ray_tracing(x, y):
	# screen is on origin 
	pixel = np.array([x, y, 0]) 
	origin = camera 
	direction = normalize(pixel - origin) 
	color = np.zeros((3)) 
	reflection = 1 
	for k in range(max_depth): 
		# check for intersections 
		nearest_object, min_distance = nearest_intersected_object(objects, origin, direction) 
		if nearest_object is None: 
			break 
		intersection = origin + min_distance * direction 
		normal_to_surface = normalize(intersection - nearest_object['center']) 
		shifted_point = intersection + 1e-5 * normal_to_surface 
		intersection_to_light = normalize(light['position'] - shifted_point) 
		_, min_distance = nearest_intersected_object(objects, shifted_point, intersection_to_light) 
		intersection_to_light_distance = np.linalg.norm(light['position'] - intersection) 
		is_shadowed = min_distance < intersection_to_light_distance 
		if is_shadowed: 
			break 
		illumination = np.zeros((3)) 
		# ambiant 
		illumination += nearest_object['ambient'] * light['ambient'] 
		# diffuse 
		illumination += nearest_object['diffuse'] * light['diffuse'] * np.dot(intersection_to_light, normal_to_surface) 
		# specular 
		intersection_to_camera = normalize(camera - intersection) 
		H = normalize(intersection_to_light + intersection_to_camera) 
		illumination += nearest_object['specular'] * light['specular'] * np.dot(normal_to_surface, H) ** (nearest_object['shininess'] / 4) 
		# reflection 
		color += reflection * illumination 
		reflection *= nearest_object['reflection'] 
		origin = shifted_point 
		direction = reflected(direction, normal_to_surface)
	return color

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

start_time = MPI.Wtime()

max_depth = 3

#### parameters
width = 300
height = 200
camera = np.array([0, 0, 1])
#camera = np.array([0, 1, 1])
light = { 'position': np.array([5, 5, 5]), 'ambient': np.array([1, 1, 1]), 'diffuse': np.array([1, 1, 1]), 'specular': np.array([1, 1, 1]) }
objects = [
    { 'center': np.array([-0.2, 0, -1]), 'radius': 0.2, 'ambient': np.array([0.1, 0, 0]), 'diffuse': np.array([0.7, 1, 0]), 'specular': np.array([1, 1, 1]), 'shininess': 80, 'reflection': 0.1 },
    { 'center': np.array([0.1, -0.3, 0]), 'radius': 0.1, 'ambient': np.array([0.1, 0, 0.1]), 'diffuse': np.array([0.7, 0, 0.7]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 },
    { 'center': np.array([0.5, 0, -1]), 'radius': 0.5, 'ambient': np.array([0.1, 0, 0.1]), 'diffuse': np.array([0.7, 0.7, 0.7]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 },
    { 'center': np.array([-0.3, 0, 0]), 'radius': 0.15, 'ambient': np.array([0, 0.1, 0]), 'diffuse': np.array([0, 0.6, 0]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 },
    { 'center': np.array([0, -9000, 0]), 'radius': 9000 - 0.7, 'ambient': np.array([0.1, 0.1, 0.1]), 'diffuse': np.array([0.6, 0.6, 0.6]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 }
]


N = height // size + (height % size > rank) #남은 iteration(여기선 height)를 적당한 rank에 분배
start = comm.scan(N)-N

ratio = float(width) / height 

screen = (-1, 1 / ratio, 1, -1 / ratio) # left, top, right, bottom

#linspace(시작, 끝, 개수) : 개수만큼 시작에서 끝까지 간격 채움.
distance_unit = (screen[3]-screen[1])/height
Y = np.linspace(screen[1]+start*distance_unit,screen[1]+(start+N)*distance_unit, N)
X = np.linspace(screen[0], screen[2], width)

local_image = np.zeros((N,width,3))

for i, y in enumerate(Y):
	for j, x in enumerate(X):
		color = ray_tracing(x,y) 
		local_image[i, j] = np.clip(color, 0, 1)#0보다 작거나 1보다 큰 값들을 0~1사이로 조정


sendcounts = comm.gather(N, root=0) #각 rank 당 N을 이어서 배열 만들기. gatherv에 필요함
image = None

#gatherv를 써야 됨. 왜냐? 각 rank마다 반복 횟수가 다를 수 있기 때문에 배열의 크기도 달라질 수 있음.
if rank==0:
    image = np.empty((sum(sendcounts), width, 3))
comm.Gatherv(sendbuf=local_image,recvbuf=(image, sendcounts), root=0)

if rank==0:
	plt.imsave('image3.png', image)

end_time = MPI.Wtime()
print("Overall elapsed time: " + str(end_time-start_time))
import trimesh 
obj_path = '../assets/objs/bottle/1ef68777bfdb7d6ba7a07ee616e34cd7/model.obj'
obj = trimesh.load(obj_path, force='mesh')

pc = obj.vertices
print(pc.shape)
print(pc.max(axis=0), pc.min(axis=0))



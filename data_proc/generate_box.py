# create box mesh (for collision mesh of franka fingers)

# scale = [0.003, 0.002, 0.0035]
# pos=[-0.0055, 0.002, 0.0395]

scale = [0.025, 0.025, 0.025]
pos=[0, 0, 0]

for i in range(2):
    for j in range(2):
        for k in range(2):
            v1 = scale[0] * (-2*k+1) + pos[0]
            v2 = scale[1] * (-2*j+1) + pos[1]
            v3 = scale[2] * (-2*i+1) + pos[2]
            print("v %.4f %.4f %.4f" % (v1, v2, v3))

    

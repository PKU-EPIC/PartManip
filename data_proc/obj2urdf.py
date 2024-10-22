import os
import sys
import trimesh 

"""
Simple script to wrap an .obj file into an .urdf file.
"""


def split_filename(string):
    path_to_file = os.path.dirname(string)
    filename, extension = os.path.splitext(os.path.basename(string))
    return path_to_file, filename, extension


def check_input():
    if len(sys.argv) == 1:
        print("Provide a <file.obj> as argument")
        sys.exit()
    elif len(sys.argv) > 2:
        print("Too many arguments")
        sys.exit()
    _, _, ext = split_filename(sys.argv[1])
    if ext != ".obj":
        print("Incorrect extension (<{}> instead of <.obj>)".format(ext))
        sys.exit()
    if not os.path.exists(sys.argv[1]):
        print("The file <{}> does not exist".format(sys.argv[1]))
        sys.exit()


def generate_output_name():
    path_to_file, filename, extension = split_filename(sys.argv[1])
    if path_to_file == "":
        new_name = filename + ".urdf"
    else:
        new_name = path_to_file + "/" + filename + ".urdf"
    return new_name


def check_output_file():
    output_name = generate_output_name()
    if os.path.exists(output_name):
        print("Warning: <{}> already exists. Do you want to continue and overwrite it? [y, n] > ".format(output_name), end="")
        ans = input().lower()
        if ans not in ["y", "yes"]:
            sys.exit()


def write_urdf_text(scale, center):
    x, y, z = center[:]
    output_name = generate_output_name()
    _, name, _ = split_filename(sys.argv[1])
    print("Creation of <{}>...".format(output_name), end="")
    with open(output_name, "w") as f:
        text = """<?xml version="1.0" ?>
<robot name="{}.urdf">
  <link name="baseLink">
    <inertial>
      <origin rpy="0 0 0" xyz="{} {} {}"/>
       <mass value="0.0"/>
       <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
    </inertial>
    <visual>
      <origin rpy="0 0 0" xyz="{} {} {}"/>
      <geometry>
        <mesh filename="{}.obj" scale="{} {} {}"/>
      </geometry>
      <material name="white">
       <color rgba="1 1 1 1"/>
     </material>
    </visual>
    <collision>
      <origin rpy="0 0 0" xyz="{} {} {}"/>
      <geometry>
        <mesh filename="{}.obj" scale="{} {} {}"/>
        <!-- You could also specify the collision (for the {}) with a "box" tag: -->
        <!-- <box size=".06 .06 .06"/> -->
      </geometry>
    </collision>
  </link>
</robot>
        """.format(name, x,y,z, x,y,z, name, scale, scale, scale, x,y,z, name, scale, scale, scale, name)
        f.write(text)
        print(" done")


if __name__ == "__main__":
  check_input()
  new_full_filename = generate_output_name()
  check_output_file()

  # move center to the origin

  scale = 1
  mesh = trimesh.load(sys.argv[1], force='mesh')
  center = scale * (mesh.vertices.max(axis=0) + mesh.vertices.min(axis=0)) /2

  write_urdf_text(scale, -center)
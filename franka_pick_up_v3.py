# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from isaacsim import SimulationApp
import omni

simulation_app = SimulationApp({"headless": False})

import omni.kit.commands
import argparse
import sys

import carb
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
import random
from PIL import Image, ImageDraw
import time
from isaacsim.core.api import World
import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.prims import GeometryPrim, RigidPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.examples.franka.controllers.pick_place_controller import PickPlaceController
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.storage.native import get_assets_root_path
from isaacsim.sensors.camera import Camera
from isaacsim.core.utils.viewports import set_camera_view
import omni.replicator.core as rep
from Jacquard_V2 import calculo_agarre


def picking(simulation_app, my_world, camera, n_test):

    x = random.uniform(0.38, 0.46) 
    y = random.uniform(0.40, 0.55)
    pos_trabajo = [x, y, 0.08] 
    #pos_trabajo = [0.48, 0.5, 0.08]
    #pos_trabajo = [0.38, 0.5, 0.08]
    path_modelos ="C:/TFM/modelos"

    # Anadimos objeto a la escena
    #piñon largo
    # asset_path=path_modelos + "/IPAGearShaft2.usd"
    # escala=[0.5, 0.5, 0.5]
    #Biela
    # asset_path= path_modelos + "/121003320R--H.usd"
    # escala=[0.0015, 0.0015, 0.0015]

    #casquillo
    asset_path= path_modelos + "/8201732034--B.usd"
    escala=[0.0025, 0.0025, 0.0025]

    ang = random.randint(0, 359)
    rotacion=[90, ang, 0]
    #rotacion=[ang, 0, 0]

    path_prim="/World/Objeto"
    add_reference_to_stage(usd_path=asset_path, prim_path=path_prim)

    grasp_obj = GeometryPrim(prim_paths_expr=path_prim, 
                            scales=np.tile(np.array(escala), (1, 1)),
                            orientations=np.tile(np.array(rot_utils.euler_angles_to_quats(np.array(rotacion), degrees=True,  extrinsic = False)), (1, 1)),
                            positions=np.tile(np.array([pos_trabajo]), (1, 1)))

    # Definimos objeto rigido para que adquiera fisicas
    grasp_obj_rigid = RigidPrim(path_prim) 
    grasp_obj_rigid.enable_rigid_body_physics()
    grasp_obj_rigid.set_masses(np.full(1,0.01)) # In kg

    # Inicializamos el objeto en la posicion deseada 
    grasp_obj.initialize()
    grasp_obj.set_world_poses()

    # Activamos las colisiones y las definimos para que se calculen con convexDecomposition
    grasp_obj.enable_collision()
    types = ["convexDecomposition"]
    grasp_obj.set_collision_approximations(types, indices=np.array([0]))

    my_world.scene.add_default_ground_plane()
    camera.add_distance_to_image_plane_to_frame()

    my_franka.gripper.set_default_state(my_franka.gripper.joint_opened_positions)
    my_world.reset()

    my_controller = PickPlaceController(
        name="pick_place_controller", gripper=my_franka.gripper, robot_articulation=my_franka
    )
    articulation_controller = my_franka.get_articulation_controller()

    for i in range(100):
        my_world.step(render=True)

    # Captura las imagenes con la cámara
    ruta_imagenes = "C:/isaacsim/standalone_examples/api/isaacsim.robot.manipulators/imagenes/"
    rgb_image = Image.fromarray(camera.get_rgb())
    rgb_image.save(ruta_imagenes + 'imagen_rgb.png')    
    depth_image = camera.get_depth()
    depth_aux = cv2.normalize(depth_image, None, 255, 0, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_normalized = Image.fromarray(depth_aux)
    depth_colored = Image.fromarray(cv2.applyColorMap(depth_aux, cv2.COLORMAP_JET))
    depth_normalized.save(ruta_imagenes + 'imagen_depth.tiff')   
    depth_colored.save(ruta_imagenes + 'imagen_depth_color.png')   

    # Carga la red y calcula el mejor agarre
    #network = "C:/isaacsim/standalone_examples/api/isaacsim.robot.manipulators/Jacquard_V2/output/models/250317_1453_training_ggcnn2_ADAM_0005/epoch_91_iou_0.99"
    network = "C:/isaacsim/standalone_examples/api/isaacsim.robot.manipulators/Jacquard_V2/output/models/training_ggcnn_ADAM_0005/epoch_83_iou_0.97"
    #network = "C:/isaacsim/standalone_examples/api/isaacsim.robot.manipulators/Jacquard_V2/output/models/250318_0657_training_resnet50_ADAM_0005/epoch_100_iou_0.94"
    punto, angulo, ancho = calculo_agarre.mejor_agarre(network, n_test)

    reset_needed = False
    agarre_parcial = False
    while simulation_app.is_running:

        my_world.step(render=True)

        if my_world.is_stopped() and not reset_needed:
            reset_needed = True
        if my_world.is_playing():
            if reset_needed:
                my_world.reset()
                my_controller.reset()
                reset_needed = False

            picking = np.ndarray.flatten(grasp_obj.get_local_poses(indices=[0])[0])
            picking[0] = 0.756 - punto[0] * 0.6 / 300
            picking[1] = 0.845 - punto[1] * 0.6 / 300
            picking[2] = 0.04
            #picking[2] = 0.009
            ang_picking = angulo * 180/np.pi

            actions = my_controller.forward(
                #picking_position=cube.get_local_pose()[0],
                picking_position=picking,
                #placing_position=np.array([-0.3, -0.3, 0.0515 / 2.0]),
                placing_position = np.array([0.2, -0.5, 0.05]),
                current_joint_positions=my_franka.get_joint_positions(),
                end_effector_offset=np.array([0, 0.005, 0]),
                end_effector_orientation=rot_utils.euler_angles_to_quats(np.array([0, 180, ang_picking]), degrees=True),
            )

            posicion = grasp_obj.get_local_poses()[0]
            if (posicion[0,2]) > 0.1:
                agarre_parcial = True
            if my_controller.is_done():
                print("done picking and placing")
                pos_actual = grasp_obj.get_local_poses()[0]
                pos_objetivo = np.array([-0.3, -0.3, 0.0515 / 2.0])
                a = [pos_actual[0,0], pos_actual[0,1]]
                b = [0.2, -0.5]
       
                agarre = False

                if np.allclose(a, b, atol=0.3):
                    agarre = True
                with open(folder_path_true + '/document2.txt', 'a') as f:
                    f.write("test:" + str(n_test))
                    f.write("\n")
                    f.write("x_pic:" + str(picking[0]))
                    f.write("\n")
                    f.write("y_pic:" + str(picking[1]))
                    f.write("\n")
                    f.write("ang_picking:" + str(ang_picking))
                    f.write("\n")
                    f.write("agarre:" + str(agarre))
                    f.write("\n")
            articulation_controller.apply_action(actions)
            if my_controller.is_done():
                break 
                       
    for _ in range(10):
        my_world.step(render=True)
    
    objeto_path ="/World/Objeto"
    omni.kit.commands.execute("DeletePrims", paths=[objeto_path])
    return agarre, agarre_parcial

assets_root_path = get_assets_root_path()
my_world = World(stage_units_in_meters=1.0)    
my_world.scene.add_default_ground_plane()

asset_path = assets_root_path + "/Isaac/Robots/Franka/franka_alt_fingers.usd"
add_reference_to_stage(usd_path=asset_path, prim_path="/World/Franka")

# Esperar a que el prim se cree
stage = omni.usd.get_context().get_stage()
prim_path = "/World/Franka"
while not stage.GetPrimAtPath(prim_path).IsValid():
    my_world.step(render=True)

gripper = ParallelGripper(
    end_effector_prim_path="/World/Franka/panda_leftfinger",
    joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
    joint_opened_positions=np.array([0.2, 0.2]),
    joint_closed_positions=np.array([0.001, 0.005]),
    #joint_closed_positions=np.array([0.0025, 0.005]),
    #action_deltas=np.array([0.025, 0.025]),
)

# Verificar si ya existe el prim y eliminarlo si es necesario
existing_prim = my_world.scene.get_object("/World/Franka/my_franka")

my_franka = my_world.scene.add(
    SingleManipulator(
        prim_path="/World/Franka", name="my_franka", end_effector_prim_name="panda_leftfinger", gripper=gripper
    )
)   

camera = Camera(
        prim_path="/World/camera",
        position=np.array([0.46, 0.54, 2.0]), # Posicion (x, y, z)
        orientation=rot_utils.euler_angles_to_quats(np.array([0, 90, 0]), degrees=True), # Orientacion (cuaternion)
        frequency=20.0, # Frecuencia de captura en Hz
        #resolution=(1024, 1024)
        resolution=(300, 300)
        ) # Resolucion de la camara (ancho xalto)

# Anadir la camara al mundo 
my_world.scene.add(camera)

camera.initialize()

# OpenCV camera matrix and width and height of the camera sensor, from the calibration file
width, height = 300, 300
camera_matrix = [[958.8, 0.0, 957.8], [0.0, 956.7, 589.5], [0.0, 0.0, 1.0]]

# Pixel size in microns, aperture and focus distance from the camera sensor specification
# Note: to disable the depth of field effect, set the f_stop to 0.0. This is useful for debugging.
pixel_size = 3 * 1e-3   # in mm, 3 microns is a common pixel size for high resolution cameras
f_stop = 1.8            # f-number, the ratio of the lens focal length to the diameter of the entrance pupil
focus_distance = 2.0    # in meters, the distance from the camera to the object plane

    # Calculate the focal length and aperture size from the camera matrix
((fx,_,cx),(_,fy,cy),(_,_,_)) = camera_matrix
horizontal_aperture =  pixel_size * width                   # The aperture size in mm
vertical_aperture =  pixel_size * height
focal_length_x  = fx * pixel_size
focal_length_y  = fy * pixel_size
focal_length = (focal_length_x + focal_length_y) / 2         # The focal length in mm

    # Set the camera parameters, note the unit conversion between Isaac Sim sensor and Kit
camera.set_focal_length(focal_length / 10.0)                # Convert from mm to cm (or 1/10th of a world unit)
camera.set_focus_distance(focus_distance)                   # The focus distance in meters
camera.set_lens_aperture(f_stop * 100.0)                    # Convert the f-stop to Isaac Sim units
camera.set_horizontal_aperture(horizontal_aperture / 10.0)  # Convert from mm to cm (or 1/10th of a world unit)
camera.set_vertical_aperture(vertical_aperture / 10.0)
camera.set_clipping_range(0.05, 1.0e5)

root_dir =  "C:/isaacsim/standalone_examples/api/isaacsim.robot.manipulators/Jacquard_V2/result_test"
folder_path_true = root_dir + '/true_image'
results_ok = 0
results_nok = 0
results_parcial = 0
for i in range(100):
    if not simulation_app.is_running:
        simulation_app = SimulationApp({"headless": True}) 
    agarre, agarre_parcial = picking(simulation_app, my_world, camera, i)

    if agarre:
        results_ok += 1
    elif agarre_parcial:
        results_parcial += 1
    else: 
        results_nok += 1        
    print(f"Iteracion {i} completada. Aciertos: {results_ok}, Parciales: {results_parcial} y Fallos: {results_nok}")  
    print(agarre, agarre_parcial) 
    with open(folder_path_true + '/document2.txt', 'a') as f:
                f.write("Resultados OK: " + str(results_ok) + ' de ' + str(i +1))
                f.write("\n") 
                f.write("Resultados Parcial: " + str(results_parcial) + ' de ' + str(i +1))
                f.write("\n") 
                f.write("\n") 
    
    my_world.stop()
    for i in range(60):
        my_world.step(render=True)
    # Esperar unos pasos antes de continuar

simulation_app.close()

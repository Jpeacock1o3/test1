# Script Name: UE_scene_csv
# Author: Feng Liu
# Date: 2025-02-09
# Version: 1.0.0

# Description: This script creates a tethernet scene with a camera and an object from the provided csv file.
#              Images from the camera are taken with camera/target position/orientation defined in the csv file starting from 3nd row
#              This script should be put in \Content\Python folder
#              The images are saved in .\Documents\Unreal Projects\MyProject\Saved\Screenshots\WindowsEditor (cannot be changed due to UE settings)
#              To run the script, in UE Output Log tab select Python, and use
#                 import UE_scene_csv
#              If changes are made to the script, use
#                 import importlib
#                 importlib.reload(UE_scene_csv)

# Contact: fliu23@buffalo.edu

import unreal
import os
import time
import csv

class ScreenshotTicker:
    def __init__(self, csv_path, output_dir="C:/Output"):
        # self.output_dir = os.path.abspath(output_dir)
        # if not os.path.exists(self.output_dir):
        #     os.makedirs(self.output_dir)
        
        # Load CSV data and initialize
        with open(csv_path, "r") as file:
            self.scenarios = list(csv.DictReader(file))
        
        self.clear_spawned_objects()
        self.current_scenario = 0
        self.camera_actor = None  # Initialize camera here if needed
        self.debris_actor = None
        # self.debris_asset = unreal.EditorAssetLibrary.load_asset("/Game/StarterContent/Props/SM_Statue.SM_Statue")
        self.debris_asset = unreal.EditorAssetLibrary.load_asset("/Game/Custom_obj/Custom_target.Custom_target")
        self.tick_handler = unreal.register_slate_pre_tick_callback(self.on_tick)
        point_light = self.add_point_light(unreal.Vector(0, 500, 700), 5000, 600, unreal.LinearColor(1, 1, 1))

    # Function to clear all spawned objects (optional)
    def clear_spawned_objects(self):
        editor_actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        actors = editor_actor_subsystem.get_all_level_actors()
        # for actor in actors:
        #     if actor.get_class() != unreal.CineCameraActor.static_class():  # Keep the camera
        #         unreal.EditorLevelLibrary.destroy_actor(actor)
        for actor in actors:
            unreal.EditorLevelLibrary.destroy_actor(actor)

    def add_point_light(self, location, intensity, radius, light_color):
        light_actor = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.PointLight, location, unreal.Rotator(0, 0, 0))
        if light_actor:
            light_actor.light_component.set_intensity(intensity)
            light_actor.light_component.set_attenuation_radius(radius)
            light_actor.light_component.set_light_color(light_color)
            unreal.log("Point light added.")
        return light_actor

    def on_tick(self, delta_time):
        # if self.current_scenario == 0:
        #     time.sleep(1)
        #     self.current_scenario += 1
        #     return
            
        if self.current_scenario >= len(self.scenarios):
            unreal.unregister_slate_pre_tick_callback(self.tick_handler)
            return

        # Spawn objects and position camera (from CSV)
        row = self.scenarios[self.current_scenario]
        debris_loc = unreal.Vector(float(row["TargetX"]), float(row["TargetY"]), float(row["TargetZ"]))
        debris_rot = unreal.Rotator(
            float(row["TargetPitch"]),
            float(row["TargetYaw"]),
            float(row["TargetRoll"])
        )

        camera_loc = unreal.Vector(float(row["CameraX"]), float(row["CameraY"]), float(row["CameraZ"]))
        camera_rot = unreal.Rotator(
            float(row["CameraPitch"]),
            float(row["CameraYaw"]),
            float(row["CameraRoll"])
        )   

        if not self.debris_actor:
            self.debris_actor = unreal.EditorLevelLibrary.spawn_actor_from_object(self.debris_asset, debris_loc, debris_rot)
        else:
            self.debris_actor.set_actor_location_and_rotation(debris_loc, debris_rot, False, False)

        # Create/reposition camera
        if not self.camera_actor:
            self.camera_actor = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.CineCameraActor, camera_loc, camera_rot)
            self.cine_camera = self.camera_actor.get_cine_camera_component()
        else:
            self.camera_actor.set_actor_location_and_rotation(camera_loc, camera_rot, False, False)

        # self.cine_camera.set_field_of_view(120.0)
        # self.cine_camera.filmback.sensor_width(1024.0)

        # Take screenshot
        # screenshot_path = os.path.join(self.output_dir, f"Scenario_{self.current_scenario}.png")
        screenshot_path = f"Scenario_{self.current_scenario}.png"
        print("==========>> ", screenshot_path)
        unreal.AutomationLibrary.take_high_res_screenshot(1920, 1080, screenshot_path, self.camera_actor, delay=0.1)
        self.current_scenario += 1
        time.sleep(.1)  # Adjust based on testing

# Usages
manager = ScreenshotTicker("C:/UE_Files/scenarios.csv")
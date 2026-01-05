# Copyright (c) Meta Platforms, Inc. and affiliates.
import sys
import time

# import inference code
sys.path.append("notebook")
from inference import Inference, load_image, load_single_mask

# load model
tag = "hf"
config_path = f"checkpoints/{tag}/pipeline.yaml"
inference = Inference(config_path, compile=False)


image = load_image("notebook/images/ref_views_4/ob_0000005/rgb/0000000.png")
mask = load_single_mask("notebook/images/ref_views_4/ob_0000005/mask",index=0)

# run model
print("\n" + "="*60)
print("Starting inference timing measurement...")
print("="*60)
start = time.time()

output = inference(image, mask, seed=42)

end = time.time()
elapsed = end - start
print("="*60)
print(f"Inference completed in: {elapsed:.3f} seconds")
print("="*60 + "\n")

# export gaussian splat
# output["gs"].save_ply(f"splat.ply")
# output["gs"].save_ply(f"ob_0000005.ply")
# print("Your reconstruction has been saved to splat.ply")

# Instead of saving the Gaussian splat:
# output["gs"].save_ply(f"ob_0000005.ply")

# Save the mesh as OBJ:
if "glb" in output and output["glb"] is not None:
    print("\n" + "="*60)
    print("Starting mesh export...")
    print("="*60)
    export_start = time.time()

    # output["glb"].export("ob_0000005.obj")
    
    import pymeshlab

    # First export the mesh temporarily
    mesh = output["glb"]
    mesh.export("temp_mesh.obj")

    # Load into PyMeshLab and decimate
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh("temp_mesh.obj")

    # Get current face count
    original_faces = ms.current_mesh().face_number()
    print(f"Original faces: {original_faces}")

    # Calculate target faces for 90% reduction (adjust as needed)
    target_faces = int(original_faces * 0.05)  # Keep 10% of faces

    # Apply quadric edge collapse decimation
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_faces)

    # Export the simplified mesh
    ms.save_current_mesh("ob_0000005.obj")

    print(f"Reduced to: {ms.current_mesh().face_number()} faces")


    export_end = time.time()
    export_elapsed = export_end - export_start
    print("="*60)
    print(f"Mesh export completed in: {export_elapsed:.3f} seconds")
    print("="*60)
    print("Your reconstruction has been saved to ob_0000005.obj\n")
else:
    print("No mesh found in the output.")




# object_placement/sam-3d-objects/notebook/images/ref_views_4/ob_0000005
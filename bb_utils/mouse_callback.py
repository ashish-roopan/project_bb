import cv2
import argparse


def check_src(src):
    # Check if the src is an image or video
    if src.endswith(".jpg") or src.endswith(".png"):
        return "image"
    elif src.endswith(".mp4") or src.endswith(".avi"):
        return "video"
    else:
        return "unknown fomat for src"

# Mouse callback function to print coordinates
def hover_event(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  # Check if the mouse is moving over the image
        param["coords"] = (x, y)
        print(f"X: {x}, Y: {y}")

def run_on_image(src):
    global image
    image = cv2.imread(src)

    # Check if image is loaded successfully
    if image is None:
        print("Error loading image")
        return

    # Display the image
    cv2.imshow("Image", image)

    data = {"coords": None}
    cv2.setMouseCallback("Image", hover_event, data)
    cv2.waitKey(0)

    # Print the final coordinates (if any)
    if data["coords"] is not None:
        print(f"Final coordinates: {data['coords']}")

    cv2.destroyAllWindows()
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True, help="Path to the image")
    return parser.parse_args()

args = parse_args()
src = args.src
src_type = check_src(src)

if src_type == "image":
    run_on_image(src)




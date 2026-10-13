import os
import pickle
import face_recognition

from attendance import mark_attendance


# Project folder
base_path = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


# Model file
encoding_file = os.path.join(
    base_path,
    "models",
    "encodings.pkl"
)


# Check encoding file
if not os.path.exists(encoding_file):

    print("Encoding file not found!")
    print("Please run face_encoding.py first.")

    exit()


# Load known face encodings
with open(encoding_file, "rb") as file:

    data = pickle.load(file)


known_encodings = data["encodings"]
known_names = data["names"]


print("Known faces loaded:", len(known_encodings))


# Check if encodings exist
if len(known_encodings) == 0:

    print("No face encodings found!")
    print("Please add student images to the dataset.")

    exit()


# Test image
image_path = os.path.join(
    base_path,
    "test.jpg"
)


# Check test image
if not os.path.exists(image_path):

    print("Test image not found!")
    print("Please place test.jpg inside the face_module folder.")

    exit()


# Load classroom photo
image = face_recognition.load_image_file(image_path)


print("\nProcessing classroom image...")


# Find all faces in the image
face_locations = face_recognition.face_locations(image)


print("Faces detected:", len(face_locations))


# Generate encodings for all detected faces
face_encodings = face_recognition.face_encodings(
    image,
    face_locations
)


# Store recognized names
recognized_names = []


# Recognize every detected face
for face_encoding in face_encodings:


    # Calculate distance from every known face
    face_distances = face_recognition.face_distance(
        known_encodings,
        face_encoding
    )


    # Find closest known face
    best_match_index = face_distances.argmin()


    # Default name
    name = "Unknown"


    # Get best distance
    best_distance = face_distances[best_match_index]


    # Check if face matches a known person
    if best_distance < 0.50:

        name = known_names[best_match_index]


    print(
        f"Best distance: {best_distance:.4f} | "
        f"Recognized as: {name}"
    )


    # Store recognized name
    recognized_names.append(name)


# Display recognition results
print("\n========== RECOGNITION RESULTS ==========")


if recognized_names:

    for name in recognized_names:

        print(name)

else:

    print("No faces recognized.")


# Mark attendance
present_students = mark_attendance(recognized_names)


print("\nPresent students:", present_students)
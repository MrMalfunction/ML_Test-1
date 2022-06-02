from PIL import Image, ImageDraw
from IPython.display import display
import os
package_dir = os.path.dirname(os.path.abspath(__file__))
print(os.listdir(os.path.join(package_dir,'./TrainSet')))

# The program we will be finding faces on the example below
# pil_im = Image.open('/work/TrainSet/Man.jpg')
# display(pil_im)
# pil_im = Image.open('/work/TrainSet/Woman.jpg')
# display(pil_im)

## block 2

train_files = os.listdir(os.path.join(package_dir,'./TrainSet'))
train_names = []
for i in train_files:
    train_names.append(i.split('.')[0])
    
## block 3
import face_recognition
import numpy as np
from PIL import Image, ImageDraw
from IPython.display import display

# This is an example of running face recognition on a single image
# and drawing a box around each person that was identified.
known_face_encodings = []
# Load a sample picture and learn how to recognize it.
for i in train_files:
    curr_image = face_recognition.load_image_file("/work/TrainSet/" + i)
    curr_face_encoding = face_recognition.face_encodings(curr_image)[0]
    known_face_encodings.append(curr_face_encoding)

known_face_names = train_names
print('Learned encoding for', len(known_face_encodings), 'images.')


## block 4
# Load an image with an unknown face
unknown_image = face_recognition.load_image_file((os.path.join(package_dir,'./TrainSet'))

face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

pil_image = Image.fromarray(unknown_image)
# Create a Pillow ImageDraw Draw instance to draw with
draw = ImageDraw.Draw(pil_image)

# Loop through each face found in the unknown image
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    # See if the face is a match for the known face(s)
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

    name = "Unknown"

    # Or instead, use the known face with the smallest distance to the new face
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances) # Returns the indices of the minimum values along an axis.
    if matches[best_match_index]:
        name = known_face_names[best_match_index]
    print(name)

    # Draw a box around the face using the Pillow module
#     draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

#     # Draw a label with a name below the face
#     text_width, text_height = draw.textsize(name)
#     draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
#     draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))


# Remove the drawing library from memory as per the Pillow docs
#del draw

# Display the resulting image
#display(pil_image)

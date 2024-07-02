import cv2
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
from efficientnet.tfkeras import preprocess_input

# Load the trained model
model_path = 'best_model.h5'
loaded_model = load_model(model_path)

# Set the input size
input_size = 128  # Adjust as needed

def preprocess_image(img):
    img = cv2.resize(img, (input_size, input_size))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def predict_face(image):
    img = preprocess_image(image)
    prediction = loaded_model.predict(img)
    class_label = "Real" if prediction > 0.5 else "Fake"
    confidence = prediction[0, 0] if class_label == "Fake" else 1 - prediction[0, 0]
    return class_label, confidence

def process_video(video_path, output_path='output.avi'):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    detector = MTCNN()
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))  # Adjust frame size if needed

    fake_confidences = []
    real_confidences = []

    while True:
        ret, dst_img = cap.read()
        if not ret:
            break
        
        detector = MTCNN()
        # Detect faces
        faces = detector.detect_faces(dst_img)

        # Loop through each detected face
        for face in faces:
            bounding_box = face['box']
            confidence = face['confidence']

            if confidence > 0.95:  # Adjust confidence threshold if needed
                x, y, w, h = bounding_box
                face_image = dst_img[y:y+h, x:x+w]

                # Predict using the trained model
                class_label, confidence = predict_face(face_image)

                # Accumulate confidence values for fake and real frames
                if class_label == 'Fake':
                    fake_confidences.append(confidence)
                else:
                    real_confidences.append(confidence)

                # Display result on the frame
                label = f"{class_label} ({confidence:.2%})"
                cv2.putText(dst_img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.rectangle(dst_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
##        writer.write(dst_img)
        cv2.imshow("Video", dst_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
##    writer.release()
    cv2.destroyAllWindows()
    # Calculate the average confidence for fake and real frames
    average_fake_confidence = sum(fake_confidences) / len(fake_confidences) if fake_confidences else 0.0
    average_real_confidence = sum(real_confidences) / len(real_confidences) if real_confidences else 0.0

    # Determine the final result based on the average confidence
    final_result = "Fake" if average_fake_confidence > average_real_confidence else "Real"

    return final_result

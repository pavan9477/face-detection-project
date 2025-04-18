from flask import Flask, render_template
import cv2

app = Flask(__name__)

# Load the pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect faces in the image
def detect_faces(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    num_faces = len(faces)
    return img, num_faces
    

# Route to serve the HTML page
@app.route('/')
def index():
    return render_template('Design.html')

# Route to handle face detection
@app.route('/detect_faces')
def detect_faces_route():
    # Access the camera and detect faces in real-time
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame, num_faces = detect_faces(frame)
        
        # Display the output
        cv2.putText(frame, f'Number of Faces shows in the Frame: {num_faces}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow('Face Detection', frame)
        
        # Exit the loop if 'e' is pressed
        if cv2.waitKey(1) & 0xFF == ord('e'):
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()
    
    
    return 'Face detection completed.'

if __name__ == '__main__':
    app.run(debug=True)

    
        
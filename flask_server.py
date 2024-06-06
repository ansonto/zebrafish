from flask import Flask, request, jsonify
import base64
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    data = request.json
    image_data = data['image'].split(",")[1]
    nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # Process the frame here if needed
    # For now, just save the frame to verify it works
    cv2.imwrite('received_frame.png', frame)
    return jsonify(success=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

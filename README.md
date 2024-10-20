# Face Recognition API

## Save a Face

- **Endpoint**: `POST /save`
- **Description**: Saves an uploaded face image into the `TrainedFaces` folder, associating it with the employee code provided.
- **Requirements**: 
  - The image file is uploaded via the request.
  - An employee code must be provided.
- **Example Request**:

  ```http
  POST /save
  Content-Type: multipart/form-data
  
  Body:
  - image: (file)
  - employeeCode: (string) 
  
 - Result: The image is stored in the TrainedFaces folder, named with the employee code

## Detect a Face
- Endpoint: POST /detect

- Description: Detects a face from an uploaded image and returns the employee code that matches the detected face.

- Requirements:
  - The image file is uploaded via the request.
  - The system works best with 5-6 trained images per employee.
- **Example Request**:

```http
POST /detect
Content-Type: multipart/form-data

Body:
  - image: (file)
```
- Response:
Returns the employee code if a match is found.
If no match is found, an appropriate message is returned.
- Notes
For optimal detection, ensure each employee has at least 5-6 trained images in the system.

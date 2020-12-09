import face_recognition


class Recognizer:
    def __init__(self, image):
        self.image = image

    def go(self):
        face_locations = face_recognition.face_locations(self.image)
        print(f'There is/are {len(face_locations)} person(s) in this image')

        image1 = face_recognition.load_image_file('./img/image1.jpg')
        encoding1 = face_recognition.face_encodings(image1)[0]

        image2 = face_recognition.load_image_file('./img/image2.jpg')
        encoding2 = face_recognition.face_encodings(image2)[0]

        image3 = face_recognition.load_image_file('./img/image3.jpg')
        encoding3 = face_recognition.face_encodings(image3)[0]

        self.compare(encoding1, encoding2)
        self.compare(encoding1, encoding3)

    def compare(self, person1, person2):
        comparison = face_recognition.compare_faces([person1], person2)
        if comparison[0]:
            print('Same person')
        else:
            print('Not same person')

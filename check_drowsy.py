import cv2
import dlib
from scipy.spatial import distance

cap = cv2.VideoCapture(0)
hogFaceD = dlib.get_frontal_face_detector() 
dlibFaceLandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat

def drowsy_func():
	while (True):
		
		ret, frame = cap.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		dfaces = hogFaceD(gray)
		#faces = face.detectmultiscale(gray, scalefactor=1.5, minneighbors=5)
		for face in dfaces:
			facelandmark = dlibFaceLandmark(gray, face)
			righteye = []
			lefteye = []

			#connecting the dots for left eye
			for i in range (36,42):
				x = facelandmark.part(i).x
				y = facelandmark.part(i).y
				lefteye.append((x,y))
				nextpoint = i+1
				
				if i == 41:
					nextpoint = 36
				x2 = facelandmark.part(nextpoint).x
				y2 = facelandmark.part(nextpoint).y
				cv2.line(frame,(x,y),(x2,y2),(255,255,255),1)

			#calculating the aspect ratio for left eye
			templeye = lefteye
			a = distance.euclidean(templeye[1], templeye[5])
			b = distance.euclidean(templeye[2], templeye[4])
			c = distance.euclidean(templeye[0], templeye[3])
			left_ear = (a+b)/(2.0*c)
			
			for i in range (42,48):
				x = facelandmark.part(i).x
				y = facelandmark.part(i).y
				righteye.append((x,y))
				nextpoint = i+1
				
				if i == 47:
					nextpoint = 42
				x2 = facelandmark.part(nextpoint).x
				y2 = facelandmark.part(nextpoint).y
				cv2.line(frame,(x,y),(x2,y2),(255,255,255),1)

			tempreye = righteye
			a = distance.euclidean(templeye[1], templeye[5])
			b = distance.euclidean(templeye[2], templeye[4])
			c = distance.euclidean(templeye[0], templeye[3])
			right_ear = (a+b)/(2.0*c)

			eyeaspectratio = (right_ear + left_ear)/2
			eyeaspectratio = round(eyeaspectratio,2)
			if eyeaspectratio < 0.35:
				cv2.rectangle(frame,(0,200), (640,300),(3,115,252),cv2.FILLED)
				cv2.putText(frame, "User is Drowsy!",(150,265),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,0,0),2 )
			print(eyeaspectratio)
		
		cv2.imshow("Drowsy", frame)
		key = cv2.waitKey(1)
		if key == 27:
			break
	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	drowsy_func()
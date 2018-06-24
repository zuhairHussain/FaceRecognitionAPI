from django.contrib.auth.models import User, Group
from rest_framework import viewsets
from api.serializers import UserSerializer, GroupSerializer, ClassesSerializer, StudentsSerializer, FileSerializer
from rest_framework.permissions import IsAdminUser, IsAuthenticated, DjangoModelPermissions
from .models import Classes, Students, File
from rest_framework.parsers import MultiPartParser, FormParser


from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.contrib.auth.models import User
import base64
from PIL import Image
import numpy as np
import urllib.request
import cv2
import os
import uuid
import datetime
import json

class CreateFolder():

	def dispatch(self, request, *args, **kwargs):
		training_folder = os.path.join(TRAINED_FACES_PATH, str(request.user))
		if not os.path.exists(training_folder):
			os.makedirs(training_folder)


class UserViewSet(viewsets.ModelViewSet):
	"""
	API endpoint that allows users to be viewed or edited.
	"""
	permission_classes = (IsAdminUser,)
	queryset = User.objects.all().order_by('-date_joined')
	serializer_class = UserSerializer


class GroupViewSet(viewsets.ModelViewSet):
	"""
	API endpoint that allows groups to be viewed or edited.
	"""
	permission_classes = (IsAdminUser,)
	queryset = Group.objects.all()
	serializer_class = GroupSerializer


class ClassesViewSet(viewsets.ModelViewSet):
	queryset = Classes.objects.all()
	serializer_class = ClassesSerializer


class StudentsViewSet(viewsets.ModelViewSet):
	permission_classes = (IsAuthenticated,)
	queryset = Students.objects.all()
	serializer_class = StudentsSerializer

class FileViewSet(viewsets.ModelViewSet):
	parser_classes = (MultiPartParser,)
	permission_classes = (IsAuthenticated,)
	queryset = File.objects.all()
	serializer_class = FileSerializer


# class ImageUploadForm(forms.Form):
#     """Image upload form."""
#     image = forms.ImageField()
#
# def FileView(request):
#     if request.method == 'POST' and request.FILES['file']:
#         myfiles = request.FILES['file']
#         fs = FileSystemStorage()
#         filename = fs.save(myfiles.name, myfiles)
#         uploaded_file_url = fs.url(filename)
#         return JsonResponse({uploaded_file_url : uploaded_file_url})

# define the path to the face detector and smile detector
FACE_DETECTOR_PATH = "{base_path}/cascades/haarcascade_frontalface_default.xml".format(
	base_path=os.path.abspath(os.path.dirname(__file__)))

SMILE_DETECTOR_PATH = "{base_path}/cascades/haarcascade_smile.xml".format(
	base_path=os.path.abspath(os.path.dirname(__file__)))

# path to trained faces and labels
TRAINED_FACES_PATH = "{base_path}/faces".format(
	base_path=os.path.abspath(os.path.dirname(__file__)))

# maximum distance between face and match
THRESHOLD = 75

# create the cascade classifiers
detector = cv2.CascadeClassifier(FACE_DETECTOR_PATH)
smiledetector = cv2.CascadeClassifier(SMILE_DETECTOR_PATH)

def get_images_and_labels(path):
	# images will contains face images
	images = []
	# labels will contains the label that is assigned to the image
	labels = []
	user_paths = [f for f in os.listdir(path) if not f.endswith('.DS_Store')]
	for user_path in user_paths:
		# Append all the absolute image paths in a list image_paths
		image_paths = [os.path.join(os.path.join(path, user_path), f) for f in os.listdir(os.path.join(path, user_path)) if not f.endswith('.DS_Store')]
		for image_path in image_paths:
			# Read the image and convert to grayscale
			image_pil = Image.open(image_path).convert('L')
			# Convert the image format into numpy array
			image = np.array(image_pil, 'uint8')
			# Detect the face in the image
			faces = detector.detectMultiScale(image)
			# If face is detected, append the face to images and the label to labels
			for (x, y, w, h) in faces:
				images.append(image[y: y + h, x: x + w])
				labels.append(int(user_path))
				#cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
				#cv2.waitKey(50)
		# return the images list and labels list
	return images, labels

recognizer = cv2.face.LBPHFaceRecognizer_create()
images, labels = get_images_and_labels(TRAINED_FACES_PATH)
recognizer.train(images, np.array(labels))

@csrf_exempt
def recognize(request):
	# initialize the data dictionary to be returned by the request
	data = {}
	data2 = {}
	# check to see if this is a get request
	if request.method == "POST":
		body_unicode = request.body.decode('utf-8')
		body = json.loads(body_unicode)

		# check to see if an image was uploaded
		if body['imageBase64'] is not None and body['student_class'] is not None :

			b64_image = body['imageBase64']  # key that is being used to send the data
			imgdata = base64.b64decode(b64_image)
			var = datetime.datetime.now().strftime(
				"%d%m%Y%H%M%S")  # This will give unique values everytime. Values are based on current datetime
			filename = "attendance_images/" + var + '.jpg'
			with open(filename, 'wb') as f:
				f.write(imgdata)
			# return a JSON response
			data2.update({"image": filename})
			# return JsonResponse(data2)


			# grab the uploaded image
			image = _grab_image(path=filename)

		# otherwise, assume that a URL was passed in
		else:
			# grab the URL from the request
			url = request.POST.get("url", None)

			# if the URL is None, then return an error
			if url is None:
				data["error"] = "No URL provided."
				return JsonResponse(data)

			print(url)
			# load the image and convert
			image = _grab_image(url=url)

		# convert the image to grayscale, load the face cascade detector,
		# and detect faces in the image
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		rects = detector.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5,
			minSize=(30, 30), flags=0)

		# construct a list of bounding boxes from the detection
		rects = [(int(x), int(y), int(x + w), int(y + h)) for (x, y, w, h) in rects]
		if len(rects) == 0:
			data.update({"detected": False})
		else :
			x, y, w, h = rects[0]
			recognizer.setThreshold(THRESHOLD)
			identity, confidence = recognizer.predict(
				image[y:h, x:w]
				)
			smile = smiledetector.detectMultiScale(
			image[y:h, x:w],
			scaleFactor= 1.7,
			minNeighbors=22,
			minSize=(25, 25),
			flags=0)
			smiling = False if len(smile) == 0 else True
			print("==========================" , identity)
			if identity > 0 :
			   try:
				   user = Students.objects.get(id=identity, student_class=body['student_class'])
				   user = {
					   "first_name": user.first_name,
					   "last_name": user.last_name,
					   "username": user.username,
					   "email": user.email,
					   "id": user.id,
				   }
			   except Students.DoesNotExist:
				   user = ""
			else :
			   user = ""

			# update the data dictionary with the faces detected
			data.update({"detected": True, "identity": identity, "user": user, "box": rects, "smiling": smiling})

	# return a JSON response
	return JsonResponse(data)

@csrf_exempt
def train(request):
	# check to see if this is a GET request
	if request.method == "GET":
		# check to see if an image was uploaded
		if request.GET.get("url", None) is not None and request.GET.get("user", None) is not None :
			print(request.GET.get("url", None))

			# grab the uploaded image
			image = _grab_image(url= request.GET.get("url", None))
			image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			rects = detector.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5,
				minSize=(30, 30), flags=0)

			# Create Folder
			training_folder = os.path.join(TRAINED_FACES_PATH, request.GET.get('user', None))
			if not os.path.exists(training_folder):
				os.makedirs(training_folder)

			# construct a list of bounding boxes from the detection
			rects = [(int(x), int(y), int(x + w), int(y + h)) for (x, y, w, h) in rects]
			if len(rects) == 0:
				return JsonResponse({"error" : "No faces detected"})
			else :
				x, y, w, h = rects[0]
				cv2.imwrite( TRAINED_FACES_PATH + "/" +  str(request.GET.get("user", None)) + "/" + str(uuid.uuid4()) + ".jpg", image[y:h, x:w] );

	return JsonResponse({"success" : True})

# @api_view(['GET'])
# @permission_classes((IsAdminUser, ))
# def new(request):
# 	if request.method == "GET":
# 		if request.GET.get("username", None) is not None and request.GET.get("email", None) is not None:
# 			user = Students.objects.create_user(request.GET.get("username", None), request.GET.get("email", None), '')
# 			user.first_name = request.GET.get("first_name", None)
# 			user.last_name = request.GET.get("last_name", None)
# 			user.age = request.GET.get("age", None)
#
# 			user.save()
#
# 			training_folder = os.path.join(TRAINED_FACES_PATH, str(user.pk))
# 			if not os.path.exists(training_folder):
# 				os.makedirs(training_folder)
# 	return JsonResponse({"sucess": True})
#


def _grab_image(path=None, base64_string=None, url=None):
	# if the path is not None, then load the image from disk
	if path is not None:
		image = cv2.imread(path)

	# otherwise, the image does not reside on disk
	else:
		# if the URL is not None, then download the image
		if url is not None:
			with urllib.request.urlopen(url) as resp:
				data = resp.read()
				image = np.asarray(bytearray(data), dtype="uint8")
				image = cv2.imdecode(image, cv2.IMREAD_COLOR)

		# if the stream is not None, then the image has been uploaded
		elif base64_string is not None:
			# sbuf = StringIO()
			# sbuf.write(base64.b64decode(base64_string))
			# pimg = Image.open(sbuf)
			# image = cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)

			image = base64.b64decode(base64_string)
			print(image, '----------')
			image = np.fromstring(image, dtype=np.uint8)
			print(image, '++++++++++')
			image = cv2.imdecode(image, 0)




		# convert the image to a NumPy array and then read it into
		# OpenCV format

	# return the image
	return image




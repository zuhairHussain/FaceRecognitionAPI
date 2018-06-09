from django.db import models
from django.core.files.storage import FileSystemStorage

fs = FileSystemStorage(location='/media')

# Create your models here.
class Classes(models.Model):
    name = models.CharField(max_length=200)

class Students(models.Model):
    #id =
    first_name = models.CharField(max_length=200)
    last_name = models.CharField(max_length=200)
    username = models.CharField(max_length=200)
    email = models.EmailField(max_length=200)
    student_class = models.ForeignKey(Classes, related_name='student_class', on_delete=models.CASCADE)


class File(models.Model):
  photo = models.ImageField(upload_to='')
  remark = models.CharField(max_length=20)
  timestamp = models.DateTimeField(auto_now_add=True)
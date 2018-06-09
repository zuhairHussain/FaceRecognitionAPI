"""FaceRecognitionAPI URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.conf.urls import url, include
from rest_framework import routers
from api import views
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
    TokenVerifyView,
)
from django.conf import settings
from django.conf.urls.static import static


router = routers.DefaultRouter()
router.register(r'users', views.UserViewSet)
router.register(r'groups', views.GroupViewSet)
router.register(r'api/classes', views.ClassesViewSet)
router.register(r'api/students', views.StudentsViewSet)
router.register(r'upload', views.FileViewSet)

# Wire up our API using automatic URL routing.
# Additionally, we include login URLs for the browsable API.
urlpatterns = [
    url('admin/', admin.site.urls),
    url(r'^', include(router.urls)),
    url('recognize/', views.recognize),
    url('train/', views.train),
    # url('new/', views.new),
    # url('users/', views.users),
    url(r'^api-auth/', include('rest_framework.urls', namespace='rest_framework')),
    url(r'^api/token/$', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    url(r'^api/token/refresh/$', TokenRefreshView.as_view(), name='token_refresh'),
    url(r'^api/token/verify/$', TokenVerifyView.as_view(), name='token_verify'),
]

if settings.DEBUG:
  urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

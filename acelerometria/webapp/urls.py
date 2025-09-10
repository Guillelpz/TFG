from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.auth import views as auth_views
from webapp.forms import CustomSetPasswordForm , CustomPasswordChangeForm
urlpatterns = [
    #Basic URLs
    path('webapp/', views.firstView, name='webapp'),
    path('test/', views.test, name='test'),
    path('', views.main, name='main'),
    path('register/', views.registerEmail, name='registerEmail'),
    path('register/end', views.registerEnd, name='registerEnd'),
    path('logOut', views.logOut, name='logOut'),
    path('exploreData/', views.exploreData, name='exploreData'), 
    #Password recovery URLs
    path('password_reset/', auth_views.PasswordResetView.as_view(), name='password_reset'),
    path('password_reset/done/', auth_views.PasswordResetDoneView.as_view(), name='password_reset_done'),
    path('reset/<uidb64>/<token>/', auth_views.PasswordResetConfirmView.as_view(form_class=CustomSetPasswordForm), name='password_reset_confirm'),
    path('reset/done/', auth_views.PasswordResetCompleteView.as_view(), name='password_reset_complete'),
    #Password change URLs
    path('password_change/',auth_views.PasswordChangeView.as_view(form_class=CustomPasswordChangeForm),name='password_change',),
    path('password_change/done/', auth_views.PasswordChangeDoneView.as_view(   template_name='registration/password_change_done.html'), name='password_change_done'),#Data visualization URLs
    #File visualization URL
    path("file/<int:file_id>/", views.file_detail, name="file_detail"),
    #File processing URLs
    path('select/<int:file_id>/', views.selectColumns, name='selectColumns'),
    path('uploadFile/', views.upload_file, name='uploadFile'),
    path('process_file/<int:file_id>/', views.process_file, name='process_file'),
    #Para acelerar visualizacion y etiquetado:
    path('get_graph_data/<int:file_id>/', views.get_graph_data, name='get_graph_data'),
    path('label/<int:file_id>/', views.label_data, name='label_data'),
    path('success', views.success, name='success'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
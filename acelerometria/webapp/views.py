import os
import pandas as pd
import subprocess
import plotly.graph_objs as go
import pyreadr
import traceback
import json
from django.contrib.admin.views.decorators import staff_member_required
from django.http import HttpResponse
from django.template import loader
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.models import User
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import login, logout
from django.contrib import messages
from .forms import UploadFileForm
from django.http import JsonResponse
from .models import File, Metric
from django.conf import settings
from django.views.decorators.http import require_POST
from pathlib import Path
from django.db import transaction
from django.utils.timezone import make_aware, is_aware, is_naive, make_naive, timezone
import datetime
from django.http import HttpResponseRedirect
from django.db import transaction
from django.db.models import Q
from django.core.serializers.json import DjangoJSONEncoder
from itertools import cycle
from itertools import groupby
from operator import itemgetter
from .utils import downsample

#pandas2ri.activate()  # Enable automatic R -> pandas conversion

def firstView(request):
    template = loader.get_template('myfirst.html')
    return HttpResponse(template.render())

def test(request):
    template = loader.get_template('test.html')
    return HttpResponse(template.render())

def main(request):
    # Log In Information
    if request.method == 'POST':
        form = AuthenticationForm(data=request.POST)
        if form.is_valid():
            login(request, form.get_user())
            return redirect(request.path)
    else:
        form = AuthenticationForm()
    # End of Log In Information
    return render(request, 'main.html', {"form": form})

def exploreData(request):
    files = File.objects.all()

    # Gender
    gender_filters = []
    if request.GET.get('Male'):
        gender_filters.append('Male')
    if request.GET.get('Female'):
        gender_filters.append('Female')
    if gender_filters:
        files = files.filter(gender__in=gender_filters)

    # Age
    min_age = request.GET.get('min_age')
    max_age = request.GET.get('max_age')
    if min_age:
        files = files.filter(age__gte=min_age)
    if max_age:
        files = files.filter(age__lte=max_age)

    # Weight
    min_weight = request.GET.get('min_weight')
    max_weight = request.GET.get('max_weight')
    if min_weight:
        files = files.filter(weight__gte=min_weight)
    if max_weight:
        files = files.filter(weight__lte=max_weight)

    # Height
    min_height = request.GET.get('min_height')
    max_height = request.GET.get('max_height')
    if min_height:
        files = files.filter(height__gte=min_height)
    if max_height:
        files = files.filter(height__lte=max_height)

    # Upload date
    uploadDateFrom = request.GET.get('uploadDateFrom')
    uploadDateTo = request.GET.get('uploadDateTo')
    if uploadDateFrom:
        files = files.filter(created_at__date__gte=uploadDateFrom)
    if uploadDateTo:
        files = files.filter(created_at__date__lte=uploadDateTo)

    # Activity date and type
    activityDateFrom = request.GET.get('activityDateFrom')
    activityDateTo = request.GET.get('activityDateTo')
    activity_names = [
        request.GET.get(f'activity{i}') for i in range(1, 10)
        if request.GET.get(f'activity{i}')
    ]

    if activity_names or activityDateFrom or activityDateTo:
        file_ids_matching = Metric.objects.values_list('file_id', flat=True)

        if activity_names:
            file_ids_matching = Metric.objects.filter(activity__in=activity_names).values_list('file_id', flat=True)

        if activityDateFrom:
            file_ids_matching = Metric.objects.filter(timestamp__date__gte=activityDateFrom).values_list('file_id', flat=True)

        if activityDateTo:
            file_ids_matching = Metric.objects.filter(timestamp__date__lte=activityDateTo).values_list('file_id', flat=True)

        if activity_names or activityDateFrom or activityDateTo:
            file_ids_matching = Metric.objects.filter(
                activity__in=activity_names if activity_names else Metric.objects.values_list('activity', flat=True),
                timestamp__date__gte=activityDateFrom if activityDateFrom else '1900-01-01',
                timestamp__date__lte=activityDateTo if activityDateTo else '2100-01-01'
            ).values_list('file_id', flat=True).distinct()

            files = files.filter(id__in=file_ids_matching)

    return render(request, 'exploreData.html', {
        'files': files
    })

def file_detail(request, file_id):
    file = get_object_or_404(File, id=file_id)
    metrics_qs = Metric.objects.filter(file=file).order_by('timestamp')
    available_metrics = [
        'ENMO', 'anglex', 'angley', 'anglez',
        'MAD', 'NeishabouriCount_x', 'NeishabouriCount_y',
        'NeishabouriCount_z', 'NeishabouriCount_vm'
    ]

    # Construir full_data desde los queryset
    full_data = []
    for m in metrics_qs:
        full_data.append({
            'timestamp': m.timestamp.isoformat(),
            'ENMO': m.ENMO,
            'anglex': m.anglex,
            'angley': m.angley,
            'anglez': m.anglez,
            'MAD': m.MAD,
            'NeishabouriCount_x': m.NeishabouriCount_x,
            'NeishabouriCount_y': m.NeishabouriCount_y,
            'NeishabouriCount_z': m.NeishabouriCount_z,
            'NeishabouriCount_vm': m.NeishabouriCount_vm,
            'activity': m.activity
        })

    total_points = len(full_data)
    try:
        threshold = int(request.GET.get('max_points', 500))
    except ValueError:
        threshold = total_points // 5

    downsampled = []

    for metric_name in available_metrics:
        series = [(i, d[metric_name]) for i, d in enumerate(full_data) if d[metric_name] is not None]
        reduced = downsample(series, threshold)
        for j, (index, value) in enumerate(reduced):
            timestamp = full_data[index]['timestamp']
            activity = full_data[index]['activity']
            if j >= len(downsampled):
                downsampled.append({'timestamp': timestamp, metric_name: value, 'activity': activity})
            else:
                downsampled[j][metric_name] = value

    # Construcción de bloques de actividad
    activity_blocks = []
    prev_activity = None
    start_time = None
    prev_metric = None

    for metric in metrics_qs:
        if metric.activity != prev_activity:
            if prev_activity and start_time:
                activity_blocks.append({
                    'start': start_time.isoformat(),
                    'end': prev_metric.timestamp.isoformat(),
                    'activity': prev_activity
                })
            start_time = metric.timestamp
            prev_activity = metric.activity
        prev_metric = metric

    if prev_activity and start_time:
        activity_blocks.append({
            'start': start_time.isoformat(),
            'end': prev_metric.timestamp.isoformat(),
            'activity': prev_activity
        })

    context = {
        'file': file,
        'filename': os.path.basename(file.upload.name),
        'downsampled_metrics_json': json.dumps(downsampled, cls=DjangoJSONEncoder),
        'available_metrics': available_metrics,
        'available_metrics_json': json.dumps(available_metrics),
        'activity_blocks_json': json.dumps(activity_blocks, cls=DjangoJSONEncoder),
        'debug': True,
    }
    
    return render(request, 'fileDetail.html', context)

def selectColumns(request, file_id):
    uploaded = File.objects.get(id=file_id)
    file_path = os.path.join(settings.MEDIA_ROOT, uploaded.file.name)
    df = pd.read_csv(file_path, nrows=20)  # Only first rows
    columns = df.columns.tolist()
    return render(request, 'selectColumns.html', {
        'file_id': file_id,
        'columns': columns
    })

###################################ITEM UPLOAD AND PROCESSING###############################################

def load_rdata_pyreadr(rdata_path):
    result = pyreadr.read_r(rdata_path)

    if 'metashort' not in result:
        raise ValueError("No 'metashort' found in RData file")

    df = result['metashort']

    if 'timestamp' in df.columns:
        try:
            # Intentar convertir directamente
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        except Exception as e:
            print(f"[WARN] Error al convertir 'timestamp': {e}")
            # Convertir a string y luego a datetime (más seguro)
            df['timestamp'] = df['timestamp'].astype(str)
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

        # Eliminar zona horaria si queda alguna
        if pd.api.types.is_datetime64tz_dtype(df['timestamp']):
            df['timestamp'] = df['timestamp'].dt.tz_localize(None)

    return df

def generate_graph(df, metric_column, label_file_path=None):
    fig = go.Figure()

    # Add the main metric line
    fig.add_trace(go.Scattergl(
        x=df["time"],
        y=df[metric_column],
        mode="lines+markers",
        name=metric_column
    ))

    # Define a color palette and cycle through it
    color_palette = cycle([
        'rgba(135, 206, 250, 0.3)',  # Light Blue
        'rgba(144, 238, 144, 0.3)',  # Light Green
        'rgba(255, 140, 0, 0.3)',    # Orange
        'rgba(255, 99, 132, 0.3)',   # Pink/Red
        'rgba(160, 160, 255, 0.3)',  # Light Purple
        'rgba(255, 206, 86, 0.3)',   # Yellow
        'rgba(173, 216, 230, 0.3)',  # Pale Blue
    ])
    activity_colors = {}

    # If activity column exists, visualize with vrects
    if 'activity' in df.columns:
        df['activity'] = df['activity'].fillna(method='ffill')
        df['group'] = (df['activity'] != df['activity'].shift()).cumsum()

        for _, group_df in df.groupby('group'):
            label = group_df['activity'].iloc[0]
            if pd.isna(label) or label == '':
                continue
            start = group_df['time'].iloc[0]
            end = group_df['time'].iloc[-1]

            # Assign new color to label if not already done
            if label not in activity_colors:
                activity_colors[label] = next(color_palette)

            fig.add_vrect(
                x0=start, x1=end,
                annotation_text=label,
                annotation_position="top left",
                fillcolor=activity_colors[label],
                opacity=0.3,
                line_width=0
            )

    # Optional: If label file was saved separately
    if label_file_path and Path(label_file_path).exists():
        labels_df = pd.read_csv(label_file_path)
        labels_df = labels_df.sort_values('start_time')

        for _, row in labels_df.iterrows():
            label = row['label']
            if label not in activity_colors:
                activity_colors[label] = next(color_palette)

            fig.add_vrect(
                x0=row['start_time'], x1=row['end_time'],
                annotation_text=label,
                annotation_position="top left",
                fillcolor=activity_colors[label],
                opacity=0.3,
                line_width=0
            )

    fig.update_layout(
        title=f"Metric: {metric_column}",
        xaxis_title="Time",
        yaxis_title=metric_column,
        xaxis_rangeslider_visible=True,
        dragmode="select",
        selectdirection="h"
    )

    return fig.to_html(full_html=False, include_plotlyjs='cdn').replace('<div', '<div id="plotly-graph"', 1)

@staff_member_required
def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            file = form.save(commit=False)
            file.user = request.user  # Associate the uploaded file with the current user
            file.save()  # Save the file object in the database
            return redirect('process_file', file_id=file.id)
    else:
        form = UploadFileForm()

    return render(request, 'upload.html', {'form': form})
def remove_tz_from_df(df):
    for col in df.columns:
        if pd.api.types.is_datetime64tz_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)
    return df

def process_with_r(input_path):
    input_path = Path(input_path).resolve()
    script_path = Path(settings.BASE_DIR) / 'scripts' / 'script.r'

    result = subprocess.run(
        ['Rscript', str(script_path), str(input_path)],
        check=False,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        error_msg = f"""
        ❌ Error executing R (exit code {result.returncode})
        ---- STDOUT ----
        {result.stdout}
        ---- STDERR ----
        {result.stderr}
        """
        raise Exception(error_msg)

    output_filename = f"metashort_{input_path.name}.RData"
    output_path = Path(settings.BASE_DIR) / 'media' / 'outputs' / output_filename

    if not output_path.exists():
        raise FileNotFoundError(f"Generated file not found: {output_path}")

    return str(output_path)

def process_file(request, file_id):
    uploaded_file = get_object_or_404(File, id=file_id)

    try:
        # Obtener nombre base del archivo subido (sin ruta)
        input_name = Path(uploaded_file.upload.name).name  # ej. "miarchivo.gt3x"

        # Construir ruta esperada del .RData
        rdata_path = Path(settings.MEDIA_ROOT) / 'outputs' / f"metashort_{input_name}.RData"

        # Si no existe, se ejecuta el script de R para generarlo
        if not rdata_path.exists():
            rdata_path = Path(process_with_r(uploaded_file.upload.path))
            print(f"RData generado en: {rdata_path}")

        # Cargar DataFrame desde el .RData
        df = load_rdata_pyreadr(rdata_path)
        print(f"Columnas cargadas del RData: {df.columns}")

        # Normalizar columna tiempo
        if 'time' not in df.columns:
            if 'timestamp' in df.columns:
                df['time'] = pd.to_datetime(df['timestamp'])
            elif 't' in df.columns:
                df['time'] = pd.to_datetime(df['t'])
            else:
                raise ValueError("No se encontró ninguna columna de tiempo válida ('time', 'timestamp' o 't')")
        else:
            # Forzar que 'time' sea datetime
            df['time'] = pd.to_datetime(df['time'])

        # ✅ Eliminar zona horaria en todas las columnas datetime con tz
        df = remove_tz_from_df(df)

        print(f"Ejemplo de tiempos en df['time']:\n{df['time'].head()}")

        # Guardar métricas en base de datos si no existen
        if not uploaded_file.metrics.exists():
            save_metrics_from_df(df, uploaded_file)
            print("Métricas guardadas en base de datos")

        available_metrics = [col for col in df.columns if col not in ['time', 'timestamp']]
        selected_metric = request.GET.get('metric', 'ENMO')
        if selected_metric not in available_metrics:
            selected_metric = available_metrics[0] if available_metrics else None
        if not selected_metric:
            raise ValueError("No hay métricas disponibles para mostrar")

        label_file_path = Path(settings.MEDIA_ROOT) / 'outputs' / f"{input_name}_labels.csv"
        labels = []
        if label_file_path.exists():
            df_labels = pd.read_csv(label_file_path)
            print(f"Etiquetas cargadas: {len(df_labels)}")
            for _, row in df_labels.iterrows():
                labels.append({
                    'start': row.get('start_time'),
                    'end': row.get('end_time'),
                    'activity': row.get('activity')
                })

    except Exception as e:
        print(f"Error en process_file: {e}")
        return render(request, 'error.html', {'message': f'Error: {str(e)}'})

    return render(request, 'graph.html', {
        #'graph': graph_html,
        'metrics': available_metrics,
        'selected_metric': selected_metric,
        'file_id': file_id,
        'labels': labels,
    })

def get_graph_data(request, file_id):
    uploaded_file = get_object_or_404(File, id=file_id)

    try:
        # Obtener nombre base del archivo subido
        input_name = Path(uploaded_file.upload.name).name  # ej. "archivo.gt3x"
        rdata_path = Path(settings.MEDIA_ROOT) / 'outputs' / f"metashort_{input_name}.RData"

        # Si no existe, procesarlo con R
        if not rdata_path.exists():
            rdata_path = Path(process_with_r(uploaded_file.upload.path))

        df = load_rdata_pyreadr(rdata_path)

        # Asegurar columna de tiempo
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
        elif 'timestamp' in df.columns:
            df['time'] = pd.to_datetime(df['timestamp'])
        elif 't' in df.columns:
            df['time'] = pd.to_datetime(df['t'])
        else:
            return JsonResponse({'error': 'No time-related column found'}, status=400)

        metric = request.GET.get('metric', 'ENMO')
        if metric not in df.columns:
            return JsonResponse({'error': f"Invalid metric: {metric}"}, status=400)

        # Validar que el dataframe tenga datos
        if df.empty:
            return JsonResponse({'error': 'Data is empty'}, status=400)

        data = {
            'x': df['time'].astype(str).tolist(),
            'y': df[metric].tolist()
        }
        return JsonResponse(data)

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@require_POST
def label_data(request, file_id):
    uploaded_file = get_object_or_404(File, id=file_id)

    try:
        input_name = Path(uploaded_file.upload.name).name
        rdata_path = Path(settings.MEDIA_ROOT) / 'outputs' / f"metashort_{input_name}.RData"

        if not rdata_path.exists():
            return JsonResponse({"status": "error", "message": "Archivo .RData no encontrado."}, status=404)

        df = load_rdata_pyreadr(rdata_path)

        # Asegurar columna de tiempo
        if 'time' not in df.columns:
            if 'timestamp' in df.columns:
                df['time'] = pd.to_datetime(df['timestamp'], errors='coerce')
            elif 't' in df.columns:
                df['time'] = pd.to_datetime(df['t'], errors='coerce')
            else:
                return JsonResponse({"status": "error", "message": "No se encontró una columna de tiempo válida."}, status=400)
        else:
            df['time'] = pd.to_datetime(df['time'], errors='coerce')

        # Comprobar y normalizar zona horaria de df['time']
        if df['time'].dt.tz is None:
            # Si es naive, localizar como UTC
            df['time'] = df['time'].dt.tz_localize('UTC')
        else:
            # Si tiene zona horaria, convertir a UTC
            df['time'] = df['time'].dt.tz_convert('UTC')

        print("📨 POST recibido:")
        print("start_time:", request.POST.get("start_time"))
        print("end_time:", request.POST.get("end_time"))
        print("activity:", request.POST.get("activity"))
        start_str = request.POST.get('start_time')
        end_str = request.POST.get('end_time')
        activity = request.POST.get('activity')

        if not all([start_str, end_str, activity]):
            return JsonResponse({"status": "error", "message": "Campos incompletos."}, status=400)
        start_time = pd.to_datetime(start_str)
        end_time = pd.to_datetime(end_str)
        if is_naive(start_time):
            start_time = make_aware(start_time)
        if is_naive(end_time):
            end_time = make_aware(end_time)
        print("➡ df['time'].dtype:", df['time'].dtype)
        print("➡ start_time tz:", start_time.tzinfo)
        print("➡ end_time tz:", end_time.tzinfo)
        print("➡ Ejemplo de df['time']:", df['time'].iloc[0])

        mask = (df['time'] >= start_time) & (df['time'] <= end_time)
        if mask.sum() == 0:
            return JsonResponse({"status": "error", "message": "No se encontraron datos en el intervalo seleccionado."}, status=400)

        if 'activity' not in df.columns:
            df['activity'] = ''

        df.loc[mask, 'activity'] = activity

        # Convertir columna activity a string limpia
        df['activity'] = df['activity'].astype(str).fillna("")

        with transaction.atomic():
            Metric.objects.filter(
                file=uploaded_file,
                timestamp__gte=start_time,
                timestamp__lte=end_time
            ).update(activity=activity)

        save_labeled_data(df, uploaded_file.upload.name, activity)

        save_rdata(df, rdata_path)

        return JsonResponse({
            "status": "success",
            "labeled_count": int(mask.sum()),
            "activity": activity
        })

    except Exception as e:
        print("=== Exception occurred in label_data view ===")
        traceback.print_exc()
        return JsonResponse({
            "status": "error",
            "message": str(e)
        }, status=500)

def save_labeled_data(df, original_filename, activity):
    # Define output directory for labeled data
    output_dir = Path(settings.MEDIA_ROOT) / 'outputs' / 'labels'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Safely format the activity name for file naming
    safe_activity = "".join(c if c.isalnum() else "_" for c in activity)
    output_name = f"labeled_{safe_activity}_{Path(original_filename).name}.csv"
    output_path = output_dir / output_name

    # Save the full labeled DataFrame
    df.to_csv(output_path, index=False)

    return output_path

def save_rdata(df, path):
    # Importa aquí: a estas alturas warmup_r() ya corrió en el arranque
    from rpy2.robjects import r, globalenv, pandas2ri
    from rpy2.robjects.conversion import localconverter
    import pandas as pd
    import os

    # Trabaja con una copia...
    df = df.copy()

    # Convertir datetime con tz a string ISO
    if 'time' in df.columns and pd.api.types.is_datetime64_any_dtype(df['time']):
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df['time'] = df['time'].dt.strftime('%Y-%m-%dT%H:%M:%S%z').fillna("")

    # Normaliza tipos
    for col in df.columns:
        if pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].astype(str)
        elif df[col].dtype == 'object':
            df[col] = df[col].astype(str).fillna("")

    df = df.fillna(0)

    # Conversión pandas -> R
    with localconverter(pandas2ri.converter):
        r_df = pandas2ri.py2rpy(df)
        globalenv['metashort'] = r_df
        rpath = str(path).replace(os.sep, '/')
        r(f"save(metashort, file='{rpath}')")

def update_labels(df, start_time, end_time, activity):
    # Update activity values directly in the DataFrame based on the provided time range
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    mask = (df['timestamp'] >= pd.to_datetime(start_time)) & (df['timestamp'] <= pd.to_datetime(end_time))
    df.loc[mask, 'activity'] = activity
    return df

def save_metrics_from_df(df, file_instance):
    # Ensure timestamp is timezone-aware
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].apply(make_aware)

    metric_objs = []
    for row in df.itertuples(index=False):
        timestamp = make_aware(row.timestamp) if is_naive(row.timestamp) else row.timestamp
        metric = Metric(
            file=file_instance,
            timestamp=timestamp,
            ENMO=getattr(row, 'ENMO', None),
            anglex=getattr(row, 'anglex', None),
            angley=getattr(row, 'angley', None),
            anglez=getattr(row, 'anglez', None),
            MAD=getattr(row, 'MAD', None),
            NeishabouriCount_x=getattr(row, 'NeishabouriCount_x', None),
            NeishabouriCount_y=getattr(row, 'NeishabouriCount_y', None),
            NeishabouriCount_z=getattr(row, 'NeishabouriCount_z', None),
            NeishabouriCount_vm=getattr(row, 'NeishabouriCount_vm', None),
            activity=getattr(row, 'activity', None)
        )
        metric_objs.append(metric)

    # Bulk insert for performance
    with transaction.atomic():
        Metric.objects.bulk_create(metric_objs, batch_size=1000)

#####################################USER REGISTRATION#############################################

def registerEmail(request):
    if request.method == "POST":
        email = request.POST.get('email')
        if User.objects.filter(email=email).exists():
            messages.error(request, 'Email is already registered.')
            return redirect('registerEmail')
        request.session['email'] = email
        return redirect('registerEnd')
    return render(request, 'registerEmail.html')

def registerEnd(request):
    email = request.session.get('email', '')

    if request.method == "POST":
        username = request.POST.get('username')
        password = request.POST.get('password')
        confirm_password = request.POST.get('confirm_password')

        if password != confirm_password:
            messages.error(request, "Passwords do not match", extra_tags='inline')
            return render(request, 'registerEnd.html', {'email': email})

        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already taken")
            return redirect('registerEnd')

        if User.objects.filter(email=email).exists():
            messages.error(request, "Email already registered")
            return redirect('registerEmail')

        user = User.objects.create_user(username=username, password=password, email=email)
        user.save()
        messages.success(request, "User successfully registered")
        return redirect('main')  

    return render(request, 'registerEnd.html', {'email': email})

def logOut(request):
  logout(request)
  return redirect('main')

def success(request):
    labeled_count = request.GET.get('labeled_count')
    activity = request.GET.get('activity')
    output_path = request.GET.get('output_path')
    
    return render(request, 'success.html', {
        'labeled_count': labeled_count,
        'activity': activity,
        'output_path': output_path
    })

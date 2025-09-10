document.getElementById('uploadForm').addEventListener('submit', async function(event) {
    event.preventDefault();

    const formData = new FormData();
    const fileInput = document.getElementById('fileInput');
    formData.append('file', fileInput.files[0]);

    const response = await fetch('/upload/', {
        method: 'POST',
        headers: {
            'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value
        },
        body: formData
    });

    const data = await response.json();
    console.log(data);

    if (data.error) {
        alert(data.error);
        return;
    }

    const keys = Object.keys(data);
    const trace = {
        x: data[keys[0]],
        y: data[keys[1]],
        type: 'scatter',
        mode: 'lines+markers'
    };

    Plotly.newPlot('graph', [trace]);
});

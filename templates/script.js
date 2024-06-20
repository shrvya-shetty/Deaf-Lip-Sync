document.addEventListener('DOMContentLoaded', function() {
    const predictBtn = document.getElementById('predict-btn');
    const predictionDiv = document.getElementById('prediction');

    predictBtn.addEventListener('click', function() {
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            predictionDiv.innerText = data.prediction;
        })
        .catch(error => console.error('Error:', error));
    });
});

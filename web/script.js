const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const clearBtn = document.getElementById('clear');
const predictBtn = document.getElementById('predict');
const resultDiv = document.getElementById('result');

let isDrawing = false;
let lastX = 0;
let lastY = 0;

// Drawing setup
ctx.strokeStyle = 'white';
ctx.lineWidth = 20;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';

function startDrawing(e) {
    isDrawing = true;
    [lastX, lastY] = [e.offsetX, e.offsetY];
}

function draw(e) {
    if (!isDrawing) return;
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();
    [lastX, lastY] = [e.offsetX, e.offsetY];
}

function stopDrawing() {
    isDrawing = false;
}

// Mouse events
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

// Touch events for mobile
canvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    const touch = e.touches[0];
    const rect = canvas.getBoundingClientRect();
    startDrawing({ offsetX: touch.clientX - rect.left, offsetY: touch.clientY - rect.top });
});
canvas.addEventListener('touchmove', (e) => {
    e.preventDefault();
    const touch = e.touches[0];
    const rect = canvas.getBoundingClientRect();
    draw({ offsetX: touch.clientX - rect.left, offsetY: touch.clientY - rect.top });
});
canvas.addEventListener('touchend', stopDrawing);

clearBtn.addEventListener('click', () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    resultDiv.textContent = '';
});

predictBtn.addEventListener('click', async () => {
    canvas.toBlob(async (blob) => {
        const file = new File([blob], 'digit.png', { type: 'image/png' });
        const formData = new FormData();
        formData.append('file', file);  // Matches UploadFile in FastAPI

        try {
            const response = await fetch('http://localhost:8000/predict_drawing', {  // Replace with your API URL
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            console.log(data);
            resultDiv.textContent = `Predicted: ${data.pred}`;
        } catch (error) {
            resultDiv.textContent = 'Error: ' + error.message;
        }
    }, 'image/png');
});

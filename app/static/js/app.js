// Lógica Frontend para RadiografIA

document.addEventListener('DOMContentLoaded', () => {
    // API URL Base (relativa al host actual para despliegue local sencillo)
    const API_BASE = "http://localhost:8000";

    // Estado local de la aplicación
    let selectedFile = null;
    let lastAnalyzedFile = null;
    let predictedClassId = null;

    // Elementos del DOM - Conexión & Estado
    const statusDot = document.getElementById('status-dot');
    const statusText = document.getElementById('status-text');
    const deviceBadge = document.getElementById('device-badge');
    const deviceText = document.getElementById('device-text');

    // Elementos del DOM - Configuración
    const thresholdSlider = document.getElementById('threshold-slider');
    const thresholdVal = document.getElementById('threshold-val');
    const layerSelect = document.getElementById('layer-select');

    // Elementos del DOM - Carga de Imagen
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const dropZonePreview = document.getElementById('drop-zone-preview');
    const imagePreview = document.getElementById('image-preview');
    const removePreviewBtn = document.getElementById('remove-preview-btn');
    const analyzeBtn = document.getElementById('analyze-btn');

    // Elementos del DOM - Resultados
    const resultsCard = document.getElementById('results-card');
    const resultPlaceholder = document.getElementById('result-placeholder');
    const resultContent = document.getElementById('result-content');
    const resultBadge = document.getElementById('result-badge');
    const resultIcon = document.getElementById('result-icon');
    const resultText = document.getElementById('result-text');
    const confidencePercentage = document.getElementById('confidence-percentage');
    const confidenceFill = document.getElementById('confidence-fill');
    const inferenceTime = document.getElementById('inference-time');
    const appliedThreshold = document.getElementById('applied-threshold');

    // Elementos del DOM - Explicabilidad
    const explainCard = document.getElementById('explain-card');
    const imgPreprocessed = document.getElementById('img-preprocessed');
    const imgHeatmap = document.getElementById('img-heatmap');
    const heatmapSpinner = document.getElementById('heatmap-spinner');
    const prepSpinner = document.getElementById('prep-spinner');

    // 1. Verificar Conexión y Estado del Servidor
    async function checkHealth() {
        try {
            const response = await fetch(`${API_BASE}/health`);
            const data = await response.json();
            
            if (response.ok && data.status === 'ok') {
                statusDot.className = 'status-indicator online';
                statusText.textContent = 'Servidor en Línea';
                
                // Mostrar dispositivo de hardware
                const isCuda = data.device.includes('cuda');
                deviceBadge.style.display = 'flex';
                deviceText.textContent = isCuda ? `GPU: ${data.device.toUpperCase()}` : 'CPU (Inferencia Local)';
                deviceBadge.className = isCuda ? 'device-info gpu-active' : 'device-info';
                
                return true;
            } else {
                setServerOffline(data.message || 'Error en inicio');
                return false;
            }
        } catch (error) {
            setServerOffline('Desconectado');
            return false;
        }
    }

    function setServerOffline(msg) {
        statusDot.className = 'status-indicator offline';
        statusText.textContent = msg;
        deviceBadge.style.display = 'none';
        analyzeBtn.disabled = true;
        console.error('El servidor no está disponible:', msg);
    }

    // Inicializar verificación de salud
    checkHealth();
    // Re-verificar salud cada 10 segundos
    setInterval(checkHealth, 10000);

    // 2. Controladores de Eventos de Configuración
    thresholdSlider.addEventListener('input', (e) => {
        thresholdVal.textContent = parseFloat(e.target.value).toFixed(2);
    });

    // Cambiar capa dinámicamente si ya se ha analizado un archivo
    layerSelect.addEventListener('change', () => {
        if (lastAnalyzedFile && predictedClassId !== null) {
            fetchGradCAM(lastAnalyzedFile, layerSelect.value, predictedClassId);
        }
    });

    // 3. Manejo de Carga de Imagen (Drag and Drop / Explorador)
    const preventDefaults = (e) => {
        e.preventDefault();
        e.stopPropagation();
    };

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('drag-over'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('drag-over'), false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFileSelection(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelection(e.target.files[0]);
        }
    });

    removePreviewBtn.addEventListener('click', (e) => {
        preventDefaults(e);
        resetUploadZone();
    });

    function handleFileSelection(file) {
        if (!file.type.startsWith('image/')) {
            alert('Por favor, selecciona únicamente archivos de imagen.');
            return;
        }

        selectedFile = file;
        
        // Cargar previsualización
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onloadend = () => {
            imagePreview.src = reader.result;
            dropZonePreview.style.display = 'flex';
            analyzeBtn.disabled = false;
        };
    }

    function resetUploadZone() {
        selectedFile = null;
        fileInput.value = '';
        imagePreview.src = '';
        dropZonePreview.style.display = 'none';
        analyzeBtn.disabled = true;
        
        // Resetear vistas de análisis
        resultPlaceholder.style.display = 'flex';
        resultContent.style.display = 'none';
        explainCard.style.display = 'none';
        lastAnalyzedFile = null;
        predictedClassId = null;
        
        resultsCard.className = 'card results-card';
    }

    // 4. Ejecución de Análisis e Inferencia
    analyzeBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        // Validar salud primero
        const isHealthy = await checkHealth();
        if (!isHealthy) {
            alert('El servidor backend no responde. Por favor, verifica que uvicorn esté corriendo.');
            return;
        }

        lastAnalyzedFile = selectedFile;
        analyzeBtn.disabled = true;
        analyzeBtn.querySelector('span').textContent = 'Analizando...';

        // Mostrar estado de carga en la UI de resultados
        resultPlaceholder.style.display = 'none';
        resultContent.style.display = 'block';
        
        // Resetear clases de resultados
        resultsCard.className = 'card results-card';
        resultBadge.className = 'result-badge';
        resultText.textContent = 'Calculando Inferencia...';
        confidencePercentage.textContent = '0%';
        confidenceFill.style.width = '0%';
        inferenceTime.textContent = '-- ms';
        appliedThreshold.textContent = parseFloat(thresholdSlider.value).toFixed(2);

        // Ocultar explicabilidad anterior y mostrar loaders
        explainCard.style.display = 'block';
        imgHeatmap.style.opacity = '0.3';
        imgPreprocessed.style.opacity = '0.3';
        heatmapSpinner.style.display = 'block';
        prepSpinner.style.display = 'block';

        // Establecer imagen original de inmediato en el panel izquierdo (preprocesamiento)
        // La API preprocesa la imagen de la misma forma, para la visualización lado a lado mostramos el preview cargado.
        imgPreprocessed.src = imagePreview.src;
        imgPreprocessed.onload = () => {
            imgPreprocessed.style.opacity = '1';
            prepSpinner.style.display = 'none';
        };

        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('threshold', thresholdSlider.value);

        try {
            // 1. Llamar a endpoint /predict
            const response = await fetch(`${API_BASE}/predict`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Error en la inferencia: ${response.statusText}`);
            }

            const data = await response.json();
            
            // 2. Renderizar Resultados
            predictedClassId = data.class_id;
            renderResults(data);

            // 3. Llamar a /explain para generar el mapa Grad-CAM
            await fetchGradCAM(selectedFile, layerSelect.value, predictedClassId);

        } catch (error) {
            console.error('Error durante el análisis:', error);
            alert(`Error procesando la solicitud: ${error.message}`);
            resetUploadZone();
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.querySelector('span').textContent = 'Ejecutar Análisis';
        }
    });

    // 5. Renderizar los resultados de Inferencia en la pantalla
    function renderResults(data) {
        const isFractura = data.class_id === 1;
        
        // Actualizar clases CSS y tarjeta de resultados
        resultsCard.className = `card results-card ${isFractura ? 'fractura' : 'no-fractura'}`;
        resultBadge.className = `result-badge ${isFractura ? 'fractura' : 'no-fractura'}`;

        const predictionText = data.prediction || (isFractura ? 'Fractura Detectada' : 'Normal');
        resultBadge.innerHTML = `
            <i data-lucide="${isFractura ? 'alert-triangle' : 'shield-check'}"></i>
            <span id="result-text">${predictionText}</span>
        `;

        lucide.createIcons({
            root: resultBadge
        });
        
        const confValue = data.confidence > 1 ? data.confidence : (data.confidence * 100);

        // Actualizar métricas
        confidencePercentage.textContent = `${confValue.toFixed(1)}%`;
        confidenceFill.style.width = `${Math.min(confValue, 100)}%`; 
        inferenceTime.textContent = `${data.inference_time_ms || '--'} ms`;
        appliedThreshold.textContent = parseFloat(thresholdSlider.value).toFixed(2);
    }

    // 6. Obtener Explicación Grad-CAM del Backend
    async function fetchGradCAM(file, layerName, classId) {
        heatmapSpinner.style.display = 'block';
        imgHeatmap.style.opacity = '0.3';

        const explainData = new FormData();
        explainData.append('file', file);
        explainData.append('layer_name', layerName);
        explainData.append('target_class', classId);

        try {
            const response = await fetch(`${API_BASE}/explain`, {
                method: 'POST',
                body: explainData
            });

            if (!response.ok) {
                throw new Error(`Error en Grad-CAM: ${response.statusText}`);
            }

            const blob = await response.blob();
            const imageUrl = URL.createObjectURL(blob);
            
            // Asignar imagen resultante
            imgHeatmap.src = imageUrl;
            imgHeatmap.onload = () => {
                imgHeatmap.style.opacity = '1';
                heatmapSpinner.style.display = 'none';
            };

        } catch (error) {
            console.error('Error cargando explicación:', error);
            heatmapSpinner.style.display = 'none';
            imgHeatmap.alt = 'Error cargando Grad-CAM';
        }
    }
});

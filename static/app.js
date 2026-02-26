/**
 * StableFusion — Client-side logic
 * Handles form interaction, image upload, API calls, and gallery.
 * Modes: Text-to-Image (no image) or Image-to-Image (with reference image)
 */

(function () {
    'use strict';

    // ====================
    // DOM References
    // ====================
    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);

    const els = {
        // GPU
        gpuStatus: $('#gpuStatus'),
        statusDot: $('#gpuStatus .status-dot'),
        statusText: $('#gpuStatus .status-text'),

        // Prompt
        prompt: $('#prompt'),
        negativePrompt: $('#negativePrompt'),

        // Generation
        width: $('#width'), widthVal: $('#widthVal'),
        height: $('#height'), heightVal: $('#heightVal'),
        steps: $('#steps'), stepsVal: $('#stepsVal'),
        cfgScale: $('#cfgScale'), cfgScaleVal: $('#cfgScaleVal'),
        seed: $('#seed'),
        sampler: $('#sampler'),
        randomSeed: $('#randomSeed'),

        // Image-to-Image
        strength: $('#strength'), strengthVal: $('#strengthVal'),
        faceStrength: $('#faceStrength'), faceStrengthVal: $('#faceStrengthVal'),
        strengthSection: $('#strengthSection'),

        // Advanced Settings
        advancedToggle: $('#advancedToggle'),
        advancedSettings: $('#advancedSettings'),

        // Quality Enhancement
        qualityBoost: $('#qualityBoost'),
        hiresFix: $('#hiresFix'),
        hiresSettings: $('#hiresSettings'),
        hiresScale: $('#hiresScale'), hiresScaleVal: $('#hiresScaleVal'),
        hiresStrength: $('#hiresStrength'), hiresStrengthVal: $('#hiresStrengthVal'),

        // Upload
        uploadZone: $('#uploadZone'),
        imageInput: $('#imageInput'),
        uploadPlaceholder: $('#uploadPlaceholder'),
        uploadPreview: $('#uploadPreview'),
        previewImg: $('#previewImg'),
        removeImage: $('#removeImage'),
        imgOptionalBadge: $('#imgOptionalBadge'),

        // Mode indicator
        modeIndicator: $('#modeIndicator'),
        modeText2Img: $('#modeText2Img'),

        // Camera
        openCameraBtn: $('#openCameraBtn'),
        cameraModal: $('#cameraModal'),
        closeCameraBtn: $('#closeCameraBtn'),
        cameraVideo: $('#cameraVideo'),
        cameraCanvas: $('#cameraCanvas'),
        cameraNoAccess: $('#cameraNoAccess'),
        captureBtn: $('#captureBtn'),
        switchCameraBtn: $('#switchCameraBtn'),
        cancelCameraBtn: $('#cancelCameraBtn'),

        // Generate
        generateBtn: $('#generateBtn'),
        btnText: $('#generateBtn .btn-text'),
        btnLoading: $('#generateBtn .btn-loading'),

        // Output
        outputSection: $('#outputSection'),
        outputImg: $('#outputImg'),
        downloadBtn: $('#downloadBtn'),
        seedInfo: $('#seedInfo'),

        // Error
        errorBanner: $('#errorBanner'),
        errorText: $('#errorText'),
        errorClose: $('#errorClose'),

        // History
        historyGallery: $('#historyGallery'),
        historyEmpty: $('#historyEmpty'),
    };

    let uploadedFile = null;
    let lastResultUrl = null;
    let lastResultFilename = null;

    // ====================
    // Init
    // ====================
    function init() {
        setupRangeSliders();
        setupAdvancedToggle();
        setupQualityToggles();
        setupUpload();
        setupCamera();
        setupGenerate();
        setupDownload();
        setupErrorClose();
        setupRandomSeed();
        fetchGpuStatus();
        fetchHistory();
        updateModeIndicator();
    }

    // ====================
    // Range Sliders
    // ====================
    function setupRangeSliders() {
        const ranges = [
            { input: els.width, display: els.widthVal, fmt: (v) => v },
            { input: els.height, display: els.heightVal, fmt: (v) => v },
            { input: els.steps, display: els.stepsVal, fmt: (v) => v },
            { input: els.cfgScale, display: els.cfgScaleVal, fmt: (v) => parseFloat(v).toFixed(1) },
            { input: els.strength, display: els.strengthVal, fmt: (v) => parseFloat(v).toFixed(2) },
            { input: els.faceStrength, display: els.faceStrengthVal, fmt: (v) => parseFloat(v).toFixed(2) },
            { input: els.hiresScale, display: els.hiresScaleVal, fmt: (v) => parseFloat(v).toFixed(2) + 'x' },
            { input: els.hiresStrength, display: els.hiresStrengthVal, fmt: (v) => parseFloat(v).toFixed(2) },
        ];

        ranges.forEach(({ input, display, fmt }) => {
            input.addEventListener('input', () => {
                display.textContent = fmt(input.value);
            });
        });
    }

    // ====================
    // Mode Indicator
    // ====================
    function updateModeIndicator() {
        if (uploadedFile) {
            els.modeText2Img.textContent = '';
            els.modeText2Img.innerHTML = '<span class="mode-dot img2img"></span> Image-to-Image';
            els.modeText2Img.classList.add('img2img-mode');
            els.strengthSection.style.opacity = '1';
            els.strengthSection.style.pointerEvents = 'auto';
        } else {
            els.modeText2Img.textContent = '';
            els.modeText2Img.innerHTML = '<span class="mode-dot"></span> Text-to-Image';
            els.modeText2Img.classList.remove('img2img-mode');
            els.strengthSection.style.opacity = '0.5';
            els.strengthSection.style.pointerEvents = 'none';
        }
    }

    // ====================
    // Advanced Settings Toggle
    // ====================
    function setupAdvancedToggle() {
        els.advancedToggle.addEventListener('click', () => {
            const isOpen = els.advancedSettings.style.display !== 'none';
            if (isOpen) {
                els.advancedSettings.style.display = 'none';
                els.advancedToggle.classList.remove('open');
            } else {
                els.advancedSettings.style.display = 'block';
                els.advancedToggle.classList.add('open');
            }
        });
    }

    // ====================
    // Quality Toggles
    // ====================
    function setupQualityToggles() {
        // Hi-Res Fix toggle
        els.hiresFix.addEventListener('change', () => {
            if (els.hiresFix.checked) {
                els.hiresSettings.classList.add('open');
            } else {
                els.hiresSettings.classList.remove('open');
            }
        });
    }

    // ====================
    // Image Upload
    // ====================
    function setupUpload() {
        // Click to open file dialog
        els.uploadZone.addEventListener('click', (e) => {
            if (e.target === els.removeImage || e.target.closest('.btn-remove')) return;
            els.imageInput.click();
        });

        // File selected
        els.imageInput.addEventListener('change', () => {
            if (els.imageInput.files.length > 0) {
                handleFile(els.imageInput.files[0]);
            }
        });

        // Drag & Drop
        els.uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            els.uploadZone.classList.add('dragover');
        });

        els.uploadZone.addEventListener('dragleave', () => {
            els.uploadZone.classList.remove('dragover');
        });

        els.uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            els.uploadZone.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                handleFile(e.dataTransfer.files[0]);
            }
        });

        // Remove image
        els.removeImage.addEventListener('click', (e) => {
            e.stopPropagation();
            clearUpload();
        });
    }

    function handleFile(file) {
        // Validate type
        if (!['image/jpeg', 'image/png'].includes(file.type)) {
            showError('Format file tidak valid. Gunakan JPG atau PNG.');
            return;
        }
        // Validate size
        if (file.size > 5 * 1024 * 1024) {
            showError('Ukuran file melebihi batas 5 MB!');
            return;
        }

        uploadedFile = file;

        // Preview
        const reader = new FileReader();
        reader.onload = (e) => {
            els.previewImg.src = e.target.result;
            els.uploadPlaceholder.style.display = 'none';
            els.uploadPreview.style.display = 'block';
        };
        reader.readAsDataURL(file);

        updateModeIndicator();
    }

    function clearUpload() {
        uploadedFile = null;
        els.imageInput.value = '';
        els.previewImg.src = '';
        els.uploadPreview.style.display = 'none';
        els.uploadPlaceholder.style.display = 'block';
        updateModeIndicator();
    }

    // ====================
    // Camera
    // ====================
    let cameraStream = null;
    let currentFacingMode = 'environment'; // 'user' for front, 'environment' for back

    function setupCamera() {
        els.openCameraBtn.addEventListener('click', openCamera);
        els.closeCameraBtn.addEventListener('click', closeCamera);
        els.cancelCameraBtn.addEventListener('click', closeCamera);
        els.captureBtn.addEventListener('click', capturePhoto);
        els.switchCameraBtn.addEventListener('click', switchCamera);

        // Close on overlay click (outside modal)
        els.cameraModal.addEventListener('click', (e) => {
            if (e.target === els.cameraModal) closeCamera();
        });

        // Close on Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && els.cameraModal.style.display !== 'none') {
                closeCamera();
            }
        });
    }

    async function openCamera() {
        els.cameraModal.style.display = 'flex';
        els.cameraNoAccess.style.display = 'none';
        els.cameraVideo.style.display = 'block';

        try {
            const constraints = {
                video: {
                    facingMode: currentFacingMode,
                    width: { ideal: 1280 },
                    height: { ideal: 960 },
                },
                audio: false,
            };
            cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
            els.cameraVideo.srcObject = cameraStream;
        } catch (err) {
            console.error('Camera access error:', err);
            els.cameraVideo.style.display = 'none';
            els.cameraNoAccess.style.display = 'flex';
        }
    }

    function closeCamera() {
        // Stop all tracks
        if (cameraStream) {
            cameraStream.getTracks().forEach((track) => track.stop());
            cameraStream = null;
        }
        els.cameraVideo.srcObject = null;
        els.cameraModal.style.display = 'none';
    }

    async function switchCamera() {
        currentFacingMode = currentFacingMode === 'user' ? 'environment' : 'user';
        // Stop current stream, reopen with new facing
        if (cameraStream) {
            cameraStream.getTracks().forEach((track) => track.stop());
        }
        try {
            cameraStream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: currentFacingMode, width: { ideal: 1280 }, height: { ideal: 960 } },
                audio: false,
            });
            els.cameraVideo.srcObject = cameraStream;
            els.cameraNoAccess.style.display = 'none';
            els.cameraVideo.style.display = 'block';
        } catch {
            els.cameraVideo.style.display = 'none';
            els.cameraNoAccess.style.display = 'flex';
        }
    }

    function capturePhoto() {
        if (!cameraStream) return;

        const video = els.cameraVideo;
        const canvas = els.cameraCanvas;

        // Set canvas to video's actual resolution
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        const ctx = canvas.getContext('2d');

        // Mirror the image to match the viewfinder
        ctx.translate(canvas.width, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        ctx.setTransform(1, 0, 0, 1, 0, 0); // reset transform

        // Flash animation
        const viewport = document.querySelector('.camera-viewport');
        const flash = document.createElement('div');
        flash.className = 'camera-flash';
        viewport.appendChild(flash);
        setTimeout(() => flash.remove(), 500);

        // Convert canvas to blob -> File
        canvas.toBlob((blob) => {
            if (!blob) return;
            const file = new File([blob], `camera_${Date.now()}.png`, { type: 'image/png' });
            handleFile(file);
            closeCamera();
        }, 'image/png');
    }

    // ====================
    // Generate
    // ====================
    function setupGenerate() {
        els.generateBtn.addEventListener('click', doGenerate);
    }

    async function doGenerate() {
        // Validate
        const prompt = els.prompt.value.trim();
        if (!prompt) {
            showError('Prompt tidak boleh kosong!');
            els.prompt.focus();
            return;
        }

        // Build form data
        const formData = new FormData();
        formData.append('prompt', prompt);
        formData.append('negative_prompt', els.negativePrompt.value.trim());
        formData.append('width', els.width.value);
        formData.append('height', els.height.value);
        formData.append('steps', els.steps.value);
        formData.append('cfg_scale', els.cfgScale.value);
        formData.append('seed', els.seed.value);
        formData.append('sampler', els.sampler.value);
        formData.append('strength', els.strength.value);
        formData.append('face_strength', els.faceStrength.value);

        if (uploadedFile) {
            formData.append('image', uploadedFile);
        }

        // Quality Enhancement params
        formData.append('quality_boost', els.qualityBoost.checked);
        formData.append('hires_fix', els.hiresFix.checked);
        formData.append('hires_scale', els.hiresScale.value);
        formData.append('hires_strength', els.hiresStrength.value);

        // UI → loading
        setLoading(true);
        hideError();

        try {
            const resp = await fetch('/api/generate', {
                method: 'POST',
                body: formData,
            });

            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.detail || `HTTP ${resp.status}`);
            }

            const data = await resp.json();

            // Show result
            lastResultUrl = data.url;
            lastResultFilename = data.filename;
            els.outputImg.src = data.url + '?t=' + Date.now();
            els.seedInfo.textContent = `Seed: ${data.seed}`;
            els.outputSection.style.display = 'block';

            // Scroll to result
            els.outputSection.scrollIntoView({ behavior: 'smooth', block: 'center' });

            // Refresh history
            fetchHistory();

        } catch (err) {
            showError(err.message || 'Terjadi kesalahan saat generate.');
        } finally {
            setLoading(false);
        }
    }

    function setLoading(loading) {
        els.generateBtn.disabled = loading;
        els.btnText.style.display = loading ? 'none' : 'inline';
        els.btnLoading.style.display = loading ? 'flex' : 'none';
    }

    // ====================
    // Download
    // ====================
    function setupDownload() {
        els.downloadBtn.addEventListener('click', () => {
            if (!lastResultUrl) return;
            const a = document.createElement('a');
            a.href = lastResultUrl;
            a.download = lastResultFilename || 'generated.png';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        });
    }

    // ====================
    // Random Seed
    // ====================
    function setupRandomSeed() {
        els.randomSeed.addEventListener('click', () => {
            els.seed.value = -1;
        });
    }

    // ====================
    // Error
    // ====================
    function showError(msg) {
        els.errorText.textContent = msg;
        els.errorBanner.style.display = 'flex';
    }

    function hideError() {
        els.errorBanner.style.display = 'none';
    }

    function setupErrorClose() {
        els.errorClose.addEventListener('click', hideError);
    }

    // ====================
    // GPU Status
    // ====================
    async function fetchGpuStatus() {
        try {
            const resp = await fetch('/api/status');
            const data = await resp.json();

            if (data.device === 'cuda') {
                els.statusDot.classList.add('active');
                els.statusText.textContent = `${data.gpu} · ${data.vram}`;
            } else {
                els.statusDot.classList.add('cpu');
                els.statusText.textContent = 'CPU Mode (No CUDA)';
            }
        } catch {
            els.statusText.textContent = 'Server offline';
        }
    }

    // ====================
    // History
    // ====================
    async function fetchHistory() {
        try {
            const resp = await fetch('/api/history');
            const items = await resp.json();

            if (items.length === 0) {
                els.historyEmpty.style.display = 'block';
                return;
            }

            els.historyEmpty.style.display = 'none';

            // Clear existing items (keep empty message)
            els.historyGallery.querySelectorAll('.history-item').forEach((el) => el.remove());

            items.forEach((item) => {
                const div = document.createElement('div');
                div.className = 'history-item';
                div.innerHTML = `
                    <img src="${item.url}" alt="${item.filename}" loading="lazy">
                    <div class="history-label">${formatDate(item.created)}</div>
                `;
                div.addEventListener('click', () => {
                    els.outputImg.src = item.url;
                    lastResultUrl = item.url;
                    lastResultFilename = item.filename;
                    els.seedInfo.textContent = '';
                    els.outputSection.style.display = 'block';
                    els.outputSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
                });
                els.historyGallery.appendChild(div);
            });

        } catch {
            // History not critical, fail silently
        }
    }

    function formatDate(iso) {
        try {
            const d = new Date(iso);
            return d.toLocaleTimeString('id-ID', { hour: '2-digit', minute: '2-digit' });
        } catch {
            return '';
        }
    }

    // ====================
    // Start
    // ====================
    document.addEventListener('DOMContentLoaded', init);
})();

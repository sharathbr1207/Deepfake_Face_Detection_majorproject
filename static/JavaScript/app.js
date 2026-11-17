// ==========================================================
// DeepFake Detection - JavaScript Client - FIXED VERSION
// ==========================================================

// Use IIFE to avoid global scope pollution
(function() {
    // DOM Elements
    const fileInput = document.getElementById('fileInput');
    const uploadBtn = document.getElementById('uploadBtn');
    const fileName = document.getElementById('fileName');
    const fileTypeIndicator = document.getElementById('fileTypeIndicator');
    const submitBtn = document.getElementById('submitBtn');
    const loading = document.getElementById('loading');
    const loadingText = document.getElementById('loadingText');
    const uploadArea = document.getElementById('uploadArea');
    const fileRequirements = document.getElementById('fileRequirements');
    const typeButtons = document.querySelectorAll('.type-btn');
    const uploadForm = document.getElementById('uploadForm'); // Added form reference

    let currentFileType = 'image';

    // Initialize file type selection
    function initializeFileType() {
        typeButtons.forEach(btn => {
            btn.addEventListener('click', function() {
                // Remove active class from all buttons
                typeButtons.forEach(b => b.classList.remove('active'));
                // Add active class to clicked button
                this.classList.add('active');
                
                // Update current file type
                currentFileType = this.dataset.type;
                
                // Update file input accept attribute
                if (currentFileType === 'image') {
                    fileInput.accept = 'image/*';
                    fileRequirements.textContent = 'Supported formats: JPG, PNG, WEBP (Maximum 5MB)';
                    uploadBtn.textContent = 'Select Image';
                    submitBtn.textContent = 'Analyze Image';
                } else {
                    fileInput.accept = 'video/*';
                    fileRequirements.textContent = 'Supported formats: MP4, AVI, MOV (Maximum 50MB)';
                    uploadBtn.textContent = 'Select Video';
                    submitBtn.textContent = 'Analyze Video';
                }
                
                // Reset file input and info
                fileInput.value = '';
                fileName.textContent = '';
                fileTypeIndicator.textContent = '';
                submitBtn.style.display = 'none';
            });
        });
    }

    // Event Listeners
    uploadBtn.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', function(e) {
        if (e.target.files.length) {
            const file = e.target.files[0];
            
            // Validate file type
            if (currentFileType === 'image' && !file.type.startsWith('image/')) {
                alert('Please select an image file (JPG, PNG, WEBP)');
                fileInput.value = ''; // Clear invalid file
                return;
            }
            
            if (currentFileType === 'video' && !file.type.startsWith('video/')) {
                alert('Please select a video file (MP4, AVI, MOV)');
                fileInput.value = ''; // Clear invalid file
                return;
            }
            
            // Validate file size
            const maxSize = currentFileType === 'image' ? 5 * 1024 * 1024 : 50 * 1024 * 1024;
            if (file.size > maxSize) {
                alert(`File size exceeds maximum limit. Maximum ${currentFileType === 'image' ? '5MB' : '50MB'} allowed.`);
                fileInput.value = ''; // Clear oversized file
                return;
            }
            
            fileName.textContent = `Selected: ${file.name}`;
            
            // Show file type indicator
            if (currentFileType === 'image') {
                fileTypeIndicator.textContent = '📷 Image File';
                fileTypeIndicator.style.color = '#2d74da';
                loadingText.textContent = 'Analyzing image... This may take a few moments.';
            } else {
                fileTypeIndicator.textContent = '🎥 Video File';
                fileTypeIndicator.style.color = '#e53e3e';
                loadingText.textContent = 'Analyzing video... This may take longer as we process each frame.';
            }
            
            submitBtn.style.display = 'none';
        }
    });

    // Drag and drop functionality
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = 'var(--secondary)';
        uploadArea.style.backgroundColor = 'rgba(255, 255, 255, 0.9)';
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.style.borderColor = 'var(--border)';
        uploadArea.style.backgroundColor = 'rgba(255, 255, 255, 0.6)';
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = 'var(--border)';
        uploadArea.style.backgroundColor = 'rgba(255, 255, 255, 0.6)';
        
        if (e.dataTransfer.files.length) {
            const file = e.dataTransfer.files[0];
            
            // Auto-detect file type
            if (file.type.startsWith('image/')) {
                currentFileType = 'image';
                typeButtons[0].click();
            } else if (file.type.startsWith('video/')) {
                currentFileType = 'video';
                typeButtons[1].click();
            } else {
                alert('Unsupported file type. Please select an image or video file.');
                return;
            }
            
            fileInput.files = e.dataTransfer.files;
            
            // Trigger change event
            const event = new Event('change');
            fileInput.dispatchEvent(event);
        }
    });

    // FIXED: Form submission handler
    uploadForm.addEventListener('submit', function(e) {
        // Prevent default form submission until we validate
        e.preventDefault();
        
        if (!fileInput.files.length) {
            alert('Please select a file first');
            return;
        }

        const file = fileInput.files[0];
        
        // Validate file type again
        if (currentFileType === 'image' && !file.type.startsWith('image/')) {
            alert('Please select an image file (JPG, PNG, WEBP)');
            return;
        }
        
        if (currentFileType === 'video' && !file.type.startsWith('video/')) {
            alert('Please select a video file (MP4, AVI, MOV)');
            return;
        }

        // Validate file size again
        const maxSize = currentFileType === 'image' ? 5 * 1024 * 1024 : 50 * 1024 * 1024;
        if (file.size > maxSize) {
            alert(`File size exceeds maximum limit. Maximum ${currentFileType === 'image' ? '5MB' : '50MB'} allowed.`);
            return;
        }

        // Show loading, hide submit button
        loading.style.display = 'block';
        submitBtn.style.display = 'none';
        submitBtn.disabled = true;

        // Now submit the form normally - this will cause page reload to results
        // The form will submit to /upload and Flask will return the result.html page
        this.submit();
    });

    // Initialize the application
    initializeFileType();

})();
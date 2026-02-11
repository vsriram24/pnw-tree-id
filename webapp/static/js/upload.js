document.addEventListener("DOMContentLoaded", () => {
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const previewSection = document.getElementById("preview-section");
    const previewImage = document.getElementById("preview-image");
    const predictBtn = document.getElementById("predict-btn");
    const clearBtn = document.getElementById("clear-btn");
    const loading = document.getElementById("loading");
    const results = document.getElementById("results");
    const predictionsList = document.getElementById("predictions-list");
    const errorDiv = document.getElementById("error");

    // Exit early on pages without the upload UI (e.g. /about)
    if (!dropZone) return;

    let selectedFile = null;

    // Click to browse
    dropZone.addEventListener("click", () => fileInput.click());

    // Keyboard accessibility for drop zone
    dropZone.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            fileInput.click();
        }
    });

    // File input change
    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // Drag and drop
    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("drag-over");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("drag-over");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("drag-over");
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    // Clear button
    clearBtn.addEventListener("click", () => {
        selectedFile = null;
        fileInput.value = "";
        dropZone.classList.remove("hidden");
        previewSection.classList.add("hidden");
        predictBtn.classList.add("hidden");
        results.classList.add("hidden");
        errorDiv.classList.add("hidden");
        dropZone.focus();
    });

    // Predict button
    predictBtn.addEventListener("click", () => {
        if (!selectedFile) return;
        uploadAndPredict(selectedFile);
    });

    function handleFile(file) {
        const validTypes = ["image/jpeg", "image/png", "image/webp"];
        if (!validTypes.includes(file.type)) {
            showError("Please select a JPG, PNG, or WebP image.");
            return;
        }

        if (file.size > 16 * 1024 * 1024) {
            showError("File is too large. Maximum size is 16MB.");
            return;
        }

        selectedFile = file;

        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            previewImage.alt = `Preview of uploaded image: ${file.name}`;
            dropZone.classList.add("hidden");
            previewSection.classList.remove("hidden");
            predictBtn.classList.remove("hidden");
            results.classList.add("hidden");
            errorDiv.classList.add("hidden");
            predictBtn.focus();
        };
        reader.readAsDataURL(file);
    }

    function uploadAndPredict(file) {
        const formData = new FormData();
        formData.append("file", file);

        predictBtn.classList.add("hidden");
        loading.classList.remove("hidden");
        results.classList.add("hidden");
        errorDiv.classList.add("hidden");

        fetch("/predict", {
            method: "POST",
            body: formData,
        })
            .then((response) => response.json())
            .then((data) => {
                loading.classList.add("hidden");

                if (data.error) {
                    showError(data.error);
                    predictBtn.classList.remove("hidden");
                    return;
                }

                displayResults(data.predictions);
            })
            .catch(() => {
                loading.classList.add("hidden");
                showError("Network error. Please try again.");
                predictBtn.classList.remove("hidden");
            });
    }

    function displayResults(predictions) {
        predictionsList.innerHTML = "";

        predictions.forEach((pred, i) => {
            const item = document.createElement("div");
            item.className = "prediction-item";
            item.setAttribute("role", "listitem");

            const pct = pred.confidence_pct;
            const displayName = pred.display_name;
            const slug = pred.species;

            item.innerHTML = `
                <span class="prediction-rank" aria-hidden="true">${i + 1}.</span>
                <div class="prediction-info">
                    <div class="prediction-name">${displayName}</div>
                    <div class="prediction-slug">${slug}</div>
                </div>
                <div class="prediction-confidence">
                    <div class="confidence-bar" role="progressbar" aria-valuenow="${pct}" aria-valuemin="0" aria-valuemax="100" aria-label="${displayName}: ${pct}% confidence">
                        <div class="confidence-fill" style="width: 0%"></div>
                    </div>
                    <div class="confidence-value">${pct}%</div>
                </div>
            `;
            predictionsList.appendChild(item);

            // Animate confidence bar
            requestAnimationFrame(() => {
                const fill = item.querySelector(".confidence-fill");
                fill.style.width = `${pct}%`;
            });
        });

        results.classList.remove("hidden");
        predictBtn.classList.remove("hidden");

        // Move focus to results for screen reader announcement
        results.setAttribute("tabindex", "-1");
        results.focus();
    }

    function showError(message) {
        errorDiv.textContent = message;
        errorDiv.classList.remove("hidden");
    }
});

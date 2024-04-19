/*function previewImage(input) {
    const imageContainer = document.getElementById('image-container');
    const selectedImage = document.getElementById('selected-image');

    if (input.files && input.files[0]) {
        const reader = new FileReader();

        reader.onload = function (e) {
            selectedImage.src = e.target.result;
            selectedImage.style.display = 'block'; // Display the selected image
        };

        reader.readAsDataURL(input.files[0]);
    } else {
        // No file selected, hide the selected image
        selectedImage.style.display = 'none';
        selectedImage.src = ''; // Clear any previous image
    }
} */
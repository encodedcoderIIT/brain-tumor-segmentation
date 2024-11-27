document.addEventListener("DOMContentLoaded", function () {
  document
    .getElementById("folderInput")
    .addEventListener("change", function (event) {
      const fileList = document.getElementById("fileList");
      fileList.innerHTML = "";
      const files = event.target.files;
      for (let i = 0; i < files.length; i++) {
        if (files[i].name.endsWith("t2f.nii.gz")) {
          const li = document.createElement("li");
          li.textContent = files[i].name;
          li.classList.add("list-group-item");
          fileList.appendChild(li);
        }
      }
    });

  document
    .getElementById("uploadButton")
    .addEventListener("click", function () {
      const files = document.getElementById("folderInput").files;
      const formData = new FormData();
      for (let i = 0; i < files.length; i++) {
        if (files[i].name.endsWith("t2f.nii.gz")) {
          formData.append("files", files[i]);
        }
      }

      fetch("/upload", {
        method: "POST",
        body: formData,
      })
        .then((response) => response.json())
        .then((data) => {
          // Clear previous images
          for (let i = 1; i <= 1; i++) {
            document.getElementById(`originalImage${i}`).innerHTML = "";
          }

          // Display converted images in the original image boxes
          data.images.forEach((image, index) => {
            if (index < 1) {
              const img = document.createElement("img");
              img.src = `data:image/png;base64,${image}`;
              img.classList.add("img-fluid", "mt-3");
              document
                .getElementById(`originalImage${index + 1}`)
                .appendChild(img);
            }
          });
        })
        .catch((error) => console.error("Error:", error));
    });

  document
    .getElementById("startAnalysisButton")
    .addEventListener("click", function () {
      const files = document.getElementById("folderInput").files;
      const formData = new FormData();

      for (let i = 0; i < files.length; i++) {
        if (files[i].name.endsWith("t2f.nii.gz")) {
          formData.append("files", files[i]);
        }
      }

      fetch("/analyze", {
        method: "POST",
        body: formData,
      })
        .then((response) => response.json())
        .then((data) => {
          // Clear previous images
          for (let i = 1; i <= 1; i++) {
            document.getElementById(`processedImage${i}`).innerHTML = "";
          }

          // Display the returned image in the processed image box
          const img = document.createElement("img");
          img.src = `data:image/png;base64,${data.prediction}`;
          img.classList.add("img-fluid", "mt-3");

          document.getElementById("processedImage1").appendChild(img);
        })
        .catch((error) => console.error("Error:", error));
    });
});

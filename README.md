📦 Deep Learning for Image Compression using Autoencoders

This project focuses on building a "Deep Learning-based Image Compression" model using a "Convolutional Autoencoder (CAE)". The model compresses images by reducing their size while maintaining acceptable visual quality, enabling efficient storage and faster transmission.

---

🎯 Objective

To develop a "Convolutional Autoencoder" that learns efficient image representations and reconstructs them with minimal loss, achieving compression without major degradation in quality.

---

🔬 Scope

* Compress images using a deep learning model
* Enable faster image transmission and optimized storage
* Provide an easy-to-use web interface using **Streamlit**
* Support real-time image compression and download
* Allow future enhancements for higher-resolution image support

---

🧠 Problem Statement

The increasing size of image data demands efficient compression techniques. Traditional methods often compromise quality or performance. This project leverages **Convolutional Autoencoders** to extract essential image features and reconstruct compressed versions with minimal loss, making it ideal for storage and bandwidth optimization.

---

🛠️ Features

* CAE Model Training:

  * Images resized to 128×128 and normalized
  * Trained using MAE loss and Adam optimizer for 500 epochs
  * Model saved as `image_compression_model.h5`

* Streamlit Web App:

  * Upload `.jpg`, `.jpeg`, or `.png` files
  * Compress images and compare original vs. compressed side-by-side
  * Option to download the compressed image

---

🚀 How to Run

1. Navigate to the project folder
2. Run the app with:

   ```bash
   streamlit run app.py
   ```
3. Open the browser at `http://localhost:8501`
4. Upload an image, compress, and download the result

---

✅ Conclusion

This project demonstrates how **deep learning can be used for image compression** while maintaining visual quality. With a simple UI, it provides a practical tool for efficient image handling. Future work may include support for high-res images and adjustable compression levels.


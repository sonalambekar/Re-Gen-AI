import { initializeApp } from "https://www.gstatic.com/firebasejs/11.4.0/firebase-app.js";
import { getFirestore, collection, addDoc } from "https://www.gstatic.com/firebasejs/11.4.0/firebase-firestore.js";

// Firebase configuration
const firebaseConfig = {
    apiKey: "AIzaSyA2ycXtES3tqzyK4nVMmKoXsniYAcqyBd8",
    authDomain: "my-project-635e4.firebaseapp.com",
    projectId: "my-project-635e4",
    storageBucket: "my-project-635e4.appspot.com",
    messagingSenderId: "423533185631",
    appId: "1:423533185631:web:23a88a1302e9517042e9ea"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const firestore = getFirestore(app);

document.getElementById('uploadForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const name = document.getElementById('name').value;
    const address = document.getElementById('address').value;
    const imageInput = document.getElementById('imageInput').files[0];

    if (!name || !address || !imageInput) {
        alert('Please fill all fields and choose an image.');
        return;
    }

    try {
        // Convert image to base64 string
        const reader = new FileReader();
        reader.readAsDataURL(imageInput);
        reader.onloadend = async () => {
            const base64String = reader.result;

            // Save land details to Firestore
            await addDoc(collection(firestore, 'lands'), {
                title: name,
                Address: address,
                imageUrl: base64String
            });

            alert('Land details uploaded successfully!');
        };
    } catch (error) {
        console.error('Error uploading land details:', error);
        alert('Error uploading land details. Please try again.');
    }
});
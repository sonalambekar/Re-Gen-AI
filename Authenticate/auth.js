// Import Firebase
import { initializeApp } from "https://www.gstatic.com/firebasejs/11.4.0/firebase-app.js";
import { getAuth, createUserWithEmailAndPassword, signInWithEmailAndPassword } from "https://www.gstatic.com/firebasejs/11.4.0/firebase-auth.js";
import { getFirestore, doc, setDoc, getDoc } from "https://www.gstatic.com/firebasejs/11.4.0/firebase-firestore.js";

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
const auth = getAuth();
const firestore = getFirestore(app);

// Get buttons
const submit = document.getElementById("submit");

// Function to check if user exists
async function checkUserExists(uid) {
  const userDoc = await getDoc(doc(firestore, "users", uid));
  return userDoc.exists();
}

// Function to handle signup
async function handleSignup(username, email, password, profileImageUrl) {
  try {
    const userCredential = await createUserWithEmailAndPassword(auth, email, password);
    console.log("User Created:", userCredential.user);
    alert("Account Created Successfully!");
    await setDoc(doc(firestore, "users", userCredential.user.uid), {
      name: username,
      email: email,
      
    });
    localStorage.setItem('userName', username); // Store the user's name in local storage
    window.location.href = "main.html";
  } catch (error) {
    console.error("Error creating user:", error.message);
    alert(error.message); // Display error message to user
  }
}

// Function to handle login
async function handleLogin(email, password) {
  try {
    const userCredential = await signInWithEmailAndPassword(auth, email, password);
    console.log("User Logged In:", userCredential.user);
    alert("Login Successful!");

    // Fetch user data from Firestore
    const userDoc = await getDoc(doc(firestore, "users", userCredential.user.uid));
    if (userDoc.exists()) {
      const userData = userDoc.data();
      localStorage.setItem('userName', userData.name); // Store the user's name in local storage
    }

    window.location.href = "main.html";
  } catch (error) {
    console.error("Error signing in:", error.message);
    alert(error.message); // Display error message to user
  }
}

// Function to upload image to Cloudinary


// Signup or Login process
submit.addEventListener("click", async function (e) {
  e.preventDefault();
  const username = document.getElementById('username') ? document.getElementById('username').value.trim() : null;
  const email = document.getElementById("email").value.trim();
  const password = document.getElementById("password").value.trim();
 

  // Check if email and password are empty
  if (!email || !password) {
    alert("Email and Password cannot be empty!");
    return;
  }

  // Validate email format
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  if (!emailRegex.test(email)) {
    alert("Invalid email format!");
    return;
  }

  if (window.location.pathname.includes("signup.html")) {
    // Handle signup
    await handleSignup(username, email, password);
  } else if (window.location.pathname.includes("auth.html")) {
    // Handle login
    await handleLogin(email, password);
  }
});
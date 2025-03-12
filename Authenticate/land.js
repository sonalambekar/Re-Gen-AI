import { initializeApp } from "https://www.gstatic.com/firebasejs/11.4.0/firebase-app.js";
import { getFirestore, collection, getDocs, getDoc, doc, onSnapshot, addDoc, query, where } from "https://www.gstatic.com/firebasejs/11.4.0/firebase-firestore.js";

// Firebase configuration
const firebaseConfig = {
    apiKey: "AIzaSyA2ycXtES3tqzyK4nVMmKoXsniYAcqyBd8",
    authDomain: "my-project-635e4.firebaseapp.com",
    projectId: "my-project-635e4",
    storageBucket: "my-project-635e4.appspot.com",
    messagingSenderId: "423533185631",
    appId: "1:423533185631:web:23a88a1302e9517042e9ea"
};

const app = initializeApp(firebaseConfig);
const firestore = getFirestore(app);
const db = firestore; // Define db variable

async function FetchLandDetails() {
    const container = document.querySelector(".container");

    try {
        const querySnapshot = await getDocs(collection(firestore, 'lands'));
        querySnapshot.forEach((doc) => {
            const lands = doc.data();
            const landId = doc.id; // Get the document ID

            const card = document.createElement("div");
            card.classList.add("card");

            const cardImage = document.createElement("img");
            cardImage.classList.add("card-img-top");
            cardImage.src = lands.imageUrl; // Assuming you have an imageUrl field in your Firestore document
            cardImage.alt = "Card Image";

            const cardBody = document.createElement("div");
            cardBody.classList.add("card-body");

            const cardTitle = document.createElement("h5");
            cardTitle.classList.add("card-title");
            cardTitle.textContent = lands.title;

            const cardText = document.createElement("p");
            cardText.classList.add("card-text");
            cardText.textContent = lands.Address;
 
            const cardButton = document.createElement("a");
            cardButton.classList.add("btn");
            cardButton.href = "#"; // No need to redirect to another page
            cardButton.textContent = "start bidding";
            cardButton.dataset.id = landId; // Store the land ID in a data attribute
            cardButton.addEventListener('click', () => {
                localStorage.setItem(lands.title,JSON.stringify(landId))
                showPopup(lands, landId);
                onsnapshot_create_BId_container(landId)
            });
      
            
             

            const timer = document.createElement("div");
            timer.classList.add("timer");
            updateTimer(timer, lands.endTime);

            cardBody.appendChild(cardTitle);
            cardBody.appendChild(cardText);
            cardBody.appendChild(cardButton);
            card.appendChild(cardImage);
            card.appendChild(cardBody);
            card.appendChild(timer);
            container.appendChild(card);

            setInterval(() => updateTimer(timer, lands.endTime), 1000);
        });
    } catch (error) {
        console.error("Error fetching land details: ", error);
    }
}

function updateTimer(timerElement, endTime) {
    const now = new Date().getTime();
    const endTimeDate = new Date(endTime).getTime();
    const distance = endTimeDate - now;

    if (isNaN(endTimeDate)) {
        timerElement.textContent = "Invalid date";
        return;
    }

    const hours = Math.floor((distance % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
    const minutes = Math.floor((distance % (1000 * 60 * 60)) / (1000 * 60));
    const seconds = Math.floor((distance % (1000 * 60)) / 1000);

    timerElement.textContent = `${hours}h ${minutes}m ${seconds}s`;

    if (distance < 3600000) { // Less than 1 hour
        timerElement.classList.add("red");
    } else {
        timerElement.classList.remove("red");
    }

    if (distance < 0) {
        timerElement.textContent = "EXPIRED";
    }
}

function showPopup(lands, landID) {
    const popup = document.getElementById("popup");
    const overlay = document.getElementById("overlay");
    const popupImage = document.getElementById("popup-image");
    const popupTitle = document.getElementById("popup-title");
    const popupText = document.getElementById("popup-text");
    const popupPlotArea = document.getElementById("popup-plot-area");
    const popupConstruction = document.getElementById("popup-construction");
    const popupStatus = document.getElementById("popup-status");
    const popupBoundary = document.getElementById("popup-boundary");
    const popupTransaction = document.getElementById("popup-transaction");
    const bidContainer = document.getElementById("bid-container");
    const bidBtn = document.getElementById("btn");
    const bidAmount = document.getElementById("bid-amount");
    const closeButton = document.getElementById("close-button");

    if (bidBtn) {
        bidBtn.dataset.id = landID; // Ensure the landID is set correctly
        bidBtn.addEventListener('click', () => handleBidButtonClick(lands, landID));
    } else {
        console.error("Bid button not found");
    }

    popupImage.src = lands.imageUrl;
    popupTitle.textContent = lands.title;
    popupText.textContent = lands.Address;
    popupPlotArea.textContent = lands.plotArea || "N/A";
    popupConstruction.textContent = lands.constructionDone || "N/A";
    popupStatus.textContent = lands.status || "N/A";
    popupBoundary.textContent = lands.boundaryWall || "N/A";
    popupTransaction.textContent = lands.transactionType || "N/A";

    popup.style.display = "block";
    overlay.style.display = "block";
    localStorage.setItem("doc_id", JSON.stringify(landID)); // Store land ID in localStorage

    // Always show the bid container with the dropdown
    bidContainer.style.display = "flex";

    // Add event listener to close button
    closeButton.addEventListener('click', () => {
        popup.style.display = "none";
        overlay.style.display = "none";
    });
}

async function handleBidButtonClick(lands, landID) {
    const landDocRef = doc(firestore, "lands", landID);
    const bidsCollectionRef = collection(landDocRef, "bids");

    // Check if the bids collection already exists
    const bidsQuery = query(bidsCollectionRef);
    const querySnapshot = await getDocs(bidsQuery);

    if (querySnapshot.empty) {
        // Create the bids collection if it doesn't exist
        await addDoc(bidsCollectionRef, {
            name: "vinay kumar",
            bid: 0 // Initial bid value
        });
        console.log("Bids collection created");
    } else {
        // Fetch and display the bids if the collection already exists
        onsnapshot_create_BId_container(landID);
    }
}

async function onsnapshot_create_BId_container(land_ID) {
    const bidContainer = document.getElementById("bid-container");
    const bid_track_container = document.querySelector(".bid-track-container");
    const currentprice = document.querySelector(".price-container");
    if (!bidContainer || !bid_track_container || !currentprice) {
        console.error("Bid container elements not found");
        return;
    }

    // Clear existing bid data
    bid_track_container.innerHTML = "";
    currentprice.innerHTML = "";

    const userAmtContainer = document.createElement("div");
    userAmtContainer.className = "user-amt-container";
    const btn = document.getElementById("btn");
    const bidAmount = document.getElementById("bid-amount");
    const landDocRef = doc(firestore, "lands", land_ID);
    const bidsCollectionRef = collection(landDocRef, "bids");

    onSnapshot(bidsCollectionRef, (snapshot) => {
        let totalAmount = 0; // Initialize totalAmount here
        snapshot.docChanges().forEach((change) => {
            if (change.type === "added") {
                const bid_details = change.doc.data();
                const User_name = bid_details.name;
                const bid = bid_details.bid;

                const username_elem = document.createElement("div");
                username_elem.id = "username";
                username_elem.textContent = User_name;

                const bid_amt_elem = document.createElement("div");
                bid_amt_elem.id = "price";
                bid_amt_elem.textContent = bid;

                userAmtContainer.appendChild(username_elem);
                userAmtContainer.appendChild(bid_amt_elem);
                bid_track_container.appendChild(userAmtContainer);

                totalAmount += Number(bid); // Sum the bid amounts as numbers
            }
        });

        currentprice.textContent = totalAmount; // Display the total amount
    });
}

async function set_bid_doc(amount) {
    const bidBtn = document.getElementById("btn");
    const land_id = bidBtn.dataset.id;
    console.log("set bid doc land_id: " + land_id);

    const landDoc = await getDoc(doc(firestore, "lands", land_id));
    const land = landDoc.data();
    const landDocRef = doc(firestore, "lands", land_id);
    const bidsCollectionRef = collection(landDocRef, "bids");

    await addDoc(bidsCollectionRef, {
        name: "vinay kumar",
        bid: Number(amount) // Ensure the bid amount is stored as a number
    });

    // Fetch and display the updated bids
    onsnapshot_create_BId_container(land_id);
}

async function showStartBid() {
    const land_ID = JSON.parse(localStorage.getItem("doc_id"));
    if (!land_ID) {
        console.error("No land ID found in localStorage");
        return;
    }
    const landDoc = await getDoc(doc(firestore, "lands", land_ID));
    if (!landDoc.exists()) {
        console.error("Land document does not exist");
        return;
    }
    const landData = landDoc.data();
    const bid_id = `${landData.title}${landData.Address}`;
    const bidDoc = await getDoc(doc(firestore, "lands", land_ID, "bids", bid_id));
    const bidBtn = document.getElementById("btn");
    if (bidBtn) {
        if (bidDoc.exists()) {
            bidBtn.style.display = "none";
            createBidContainer(bid_id);
        } else {
            bidBtn.style.display = "block";
        }
    } else {
        console.error("Bid button not found");
    }
}

document.addEventListener("DOMContentLoaded", () => {
    FetchLandDetails();
    showStartBid();

    const bidBtn = document.getElementById("btn");
    if (bidBtn) {
        bidBtn.addEventListener('click', () => {
            const bidAmount = document.getElementById("bid-amount").value;
            set_bid_doc(bidAmount);
        });
    } else {
        console.error("Bid button not found on DOMContentLoaded");
    }
});
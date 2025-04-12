// Get references to the input field, button, and recommendations container
const bookTitleInput = document.getElementById("bookTitle");
const recommendBtn = document.getElementById("recommendBtn");
const recommendationsContainer = document.getElementById("recommendationsContainer");

// Function to handle the book recommendation request
async function getRecommendations(title) {
    const recommendationsContainer = document.getElementById("recommendationsContainer");

    try {
        const response = await fetch(`http://127.0.0.1:5000/recommend?title=${encodeURIComponent(title)}`);

        if (!response.ok) {
            throw new Error("Network response was not ok");
        }

        const data = await response.json();

        if (data.recommendations && data.recommendations.length > 0) {
            recommendationsContainer.innerHTML = "";  // Clear previous recommendations

            data.recommendations.forEach(book => {
                const bookDiv = document.createElement("div");
                bookDiv.classList.add("book");

                const titleElement = document.createElement("h3");
                titleElement.textContent = book.title;
                bookDiv.appendChild(titleElement);

                const authorElement = document.createElement("p");
                authorElement.textContent = `Author: ${book.authors || "Unknown"}`;
                bookDiv.appendChild(authorElement);

                const categoryElement = document.createElement("p");
                categoryElement.textContent = `Category: ${book.categories || "Unknown"}`;
                bookDiv.appendChild(categoryElement);

                recommendationsContainer.appendChild(bookDiv);
            });
        } else {
            recommendationsContainer.innerHTML = "<p>No recommendations found.</p>";
        }
    } catch (error) {
        recommendationsContainer.innerHTML = "<p>There was an error fetching the recommendations.</p>";
        console.error("Error fetching recommendations:", error);
    }
}

// Attach an event listener to the button to trigger the recommendation request
recommendBtn.addEventListener("click", () => {
    const title = bookTitleInput.value.trim();

    if (title) {
        getRecommendations(title);
    } else {
        recommendationsContainer.innerHTML = "<p>Please enter a book title.</p>";
    }
});
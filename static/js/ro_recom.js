// Functie pentru a prelua datele recomandarilor din Flask
        async function getRecommendations(title) {
            const recommendationsContainer = document.getElementById("recommendationsContainer");

            // !Se sterg recomandarile trecute
            recommendationsContainer.innerHTML = "";

            if (!title) {
                recommendationsContainer.innerHTML = "<p>Te rog introdu un titlu.</p>";
                return;
            }

            try {
                // Trimite un request GET pentru a primi recomandarea
                const response = await fetch(`http://127.0.0.1:5000/ro/recommend?title=${encodeURIComponent(title)}`);

                if (!response.ok) {
                    throw new Error("Eroare de retea.");
                }

                const data = await response.json();

                if (data.recommendations && data.recommendations.length > 0) {
                    data.recommendations.forEach(book => {
                        const bookDiv = document.createElement("div");
                        bookDiv.classList.add("book");

                        const titleElement = document.createElement("h3");
                        titleElement.textContent = book.titlu;
                        bookDiv.appendChild(titleElement);

                        const authorElement = document.createElement("p");
                        authorElement.textContent = `Autor: ${book.autor || "Unknown"}`;
                        bookDiv.appendChild(authorElement);

                        const categoryElement = document.createElement("p");
                        categoryElement.textContent = `Categorie: ${book.genwiki || "Unknown"}`;
                        bookDiv.appendChild(categoryElement);

                        recommendationsContainer.appendChild(bookDiv);
                    });
                } else {
                    recommendationsContainer.innerHTML = "<p>Nu au fost gasite recomandari.</p>";
                }
            } catch (error) {
                recommendationsContainer.innerHTML = "<p>Cartea nu se afla in data de baze.</p>";
                console.error("S-a intampinat o problema in gasirea cartii:", error);
            }
        }
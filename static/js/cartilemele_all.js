document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("newBookForm");
  const bookList = document.getElementById("bookList");
  const modalOverlay = document.getElementById("modalOverlay");
  const addBookBtn = document.getElementById("addBookBtn");

  // Show modal when clicking "Adaugă carte noua" button
  addBookBtn.addEventListener("click", () => {
    modalOverlay.classList.remove("hidden");
  });

  // Hide modal when clicking outside form content (optional)
  modalOverlay.addEventListener("click", (e) => {
    if (e.target === modalOverlay) {
      modalOverlay.classList.add("hidden");
      form.reset();
    }
  });

  // Load and render books on page load
  renderBooks();

  // Handle form submission
  form.addEventListener("submit", (e) => {
    e.preventDefault();

    const title = form.title.value.trim();
    const author = form.author.value.trim();
    const rating = form.rating.value;

    if (!title || !author || !rating) {
      alert("Te rog completează toate câmpurile!");
      return;
    }

    const newBook = { title, author, rating };

    // Get existing books or empty array
    const books = JSON.parse(localStorage.getItem("books")) || [];
    books.push(newBook);

    // Save updated list
    localStorage.setItem("books", JSON.stringify(books));

    // Reset form and hide modal
    form.reset();
    modalOverlay.classList.add("hidden");

    // Re-render book list
    renderBooks();
  });

  // Function to render all books as grid cards
  function renderBooks() {
    const books = JSON.parse(localStorage.getItem("books")) || [];
    bookList.innerHTML = "";

    books.forEach((book, index) => {
      const item = document.createElement("div");
      item.classList.add("book-item");

      item.innerHTML = `
        <strong>${book.title}</strong><br />
        Autor: ${book.author}<br />
        Recenzie: ${"⭐".repeat(book.rating)}<br />
        <button style="font-family: Rubik" class="remove-book" data-index="${index}">Șterge</button>
      `;

      bookList.appendChild(item);
    });

    // Attach event listeners to all remove buttons
    document.querySelectorAll(".remove-book").forEach(button => {
      button.addEventListener("click", (e) => {
        const index = e.target.getAttribute("data-index");
        removeBook(index);
      });
    });
  }

  // Remove book by index
  function removeBook(index) {
    const books = JSON.parse(localStorage.getItem("books")) || [];
    books.splice(index, 1);
    localStorage.setItem("books", JSON.stringify(books));
    renderBooks();
  }
});
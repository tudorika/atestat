document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("newBookForm");
  const bookList = document.getElementById("bookList");
  const modalOverlay = document.getElementById("modalOverlay");
  const addBookBtn = document.getElementById("addBookBtn");

  addBookBtn.addEventListener("click", () => {
    modalOverlay.classList.remove("hidden");
  });

  // Se ascunde fereastra daca se apasa in afara ei
  modalOverlay.addEventListener("click", (e) => {
    if (e.target === modalOverlay) {
      modalOverlay.classList.add("hidden");
      form.reset();
    }
  });

  // Incarca si afiseaza cartile pe pagina
  renderBooks();

  // Pentru a salva datele introduse in formular
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
    // Se updateaza lista cu carti sau se creeaza daca nu exista deja
    const books = JSON.parse(localStorage.getItem("books")) || [];
    books.push(newBook);

    // Se salveaza lista cu carti
    localStorage.setItem("books", JSON.stringify(books));

    // Se reseteaza formularul si se inchide fereastra
    form.reset();
    modalOverlay.classList.add("hidden");

    // Se afiseaza din nou, dupa update, cartile din lista
    renderBooks();
  });

  // Functie pentru a afisa toate cartile in spatii separate
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

    // Event listener pentru toate butoanele de stergere (detecteaza actiunea de stergere)
    document.querySelectorAll(".remove-book").forEach(button => {
      button.addEventListener("click", (e) => {
        const index = e.target.getAttribute("data-index");
        removeBook(index);
      });
    });
  }

  // Sterge cartile dupa index
  function removeBook(index) {
    const books = JSON.parse(localStorage.getItem("books")) || [];
    books.splice(index, 1);
    localStorage.setItem("books", JSON.stringify(books));
    renderBooks();
  }
});
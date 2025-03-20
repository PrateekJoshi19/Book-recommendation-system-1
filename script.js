document.getElementById('preferences-form').addEventListener('submit', async (e) => {
    e.preventDefault();

    const answers = [
        parseInt(document.getElementById('read-frequency').value),
        // Add more answers here
    ];

    const response = await fetch('/recommend', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ answers }),
    });

    const recommendations = await response.json();
    const recommendationsDiv = document.getElementById('recommendations');
    recommendationsDiv.innerHTML = '<h2>Recommended Books:</h2>';

    recommendations.forEach(book => {
        const bookDiv = document.createElement('div');
        bookDiv.className = 'book';
        bookDiv.innerHTML = `
            <h3>${book.Title}</h3>
            <p>Score: ${book.final_score.toFixed(2)}</p>
            <p>Genres: ${book.genres.join(', ')}</p>
            <p>Price: $${book['Price Starting With ($)']}</p>
        `;
        recommendationsDiv.appendChild(bookDiv);
    });
});
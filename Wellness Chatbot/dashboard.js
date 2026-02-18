document.addEventListener('DOMContentLoaded', () => {
    // Check authentication
    const token = localStorage.getItem('token') || new URLSearchParams(window.location.search).get('token');
    const name = localStorage.getItem('name') || new URLSearchParams(window.location.search).get('name');

    if (!token) {
        window.location.href = '/login';
        return;
    }

    // Save tokens if present in URL
    if (new URLSearchParams(window.location.search).get('token')) {
        localStorage.setItem('token', token);
        localStorage.setItem('name', name);
        // Clean URL
        window.history.replaceState({}, document.title, "/dashboard");
    }

    document.getElementById('userName').textContent = name || 'Friend';

    // Elements
    const chatBox = document.getElementById('chatBox');
    const userInput = document.getElementById('userInput');
    const sendBtn = document.getElementById('sendBtn');
    const moodSelect = document.getElementById('moodSelect');
    const logoutBtn = document.getElementById('logoutBtn');

    // Logout
    logoutBtn.addEventListener('click', (e) => {
        e.preventDefault();
        localStorage.clear();
        window.location.href = '/login';
    });

    // Send Message Function
    async function sendMessage() {
        const message = userInput.value.trim();
        const mood = moodSelect.value;

        if (!message) return;

        // Add User Message
        appendMessage(message, 'user');
        userInput.value = '';

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify({ message, mood })
            });

            const data = await response.json();

            // Add Bot Message
            appendMessage(data.reply, 'bot');
        } catch (error) {
            console.error('Error:', error);
            appendMessage("I'm having trouble connecting. Please try again.", 'bot');
        }
    }

    // Event Listeners
    sendBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });

    function appendMessage(text, sender) {
        const div = document.createElement('div');
        div.classList.add('message', `${sender}-message`);
        div.textContent = text;
        chatBox.appendChild(div);
        chatBox.scrollTop = chatBox.scrollHeight;
    }
});

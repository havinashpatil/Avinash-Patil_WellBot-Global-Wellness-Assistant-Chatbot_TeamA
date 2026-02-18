// Make sure JS is loaded
console.log("register.js loaded");

// Wait for DOM to load
document.addEventListener("DOMContentLoaded", function () {
    console.log("DOM loaded - attaching event listeners");

    // Register button
    const registerBtn = document.getElementById("registerBtn");
    if (registerBtn) {
        registerBtn.addEventListener("click", function () {
            console.log("Register button clicked");

            const name = document.getElementById("name").value;
            const email = document.getElementById("email").value;
            const password = document.getElementById("password").value;
            const language = document.getElementById("language").value;

            if (!name || !email || !password || !language) {
                alert("❌ Please fill all fields");
                return;
            }

            // Validate email format
            const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
            if (!emailRegex.test(email)) {
                alert("❌ Please enter a valid email");
                return;
            }

            // Validate password length (minimum 8 characters)
            if (password.length < 8) {
                alert("❌ Password must be at least 8 characters");
                return;
            }

            // Validate password strength
            const hasUppercase = /[A-Z]/.test(password);
            const hasLowercase = /[a-z]/.test(password);
            const hasNumber = /\d/.test(password);
            const hasSpecialChar = /[!@#$%^&*()_+\-=\[\]{}|;:,.<>?]/.test(password);

            if (!hasUppercase) {
                alert("❌ Password must contain at least one uppercase letter");
                return;
            }

            if (!hasLowercase) {
                alert("❌ Password must contain at least one lowercase letter");
                return;
            }

            if (!hasNumber) {
                alert("❌ Password must contain at least one number");
                return;
            }

            if (!hasSpecialChar) {
                alert("❌ Password must contain at least one special character (!@#$%^&*...)");
                return;
            }

            const data = {
                name: name,
                email: email,
                password: password,
                language: language
            };

            console.log("Registration data:", data);

            fetch('http://localhost:5000/signup', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            })
                .then(response => response.json())
                .then(result => {
                    if (result.success) {
                        alert("✅ Registration successful! Welcome, " + result.name);
                        localStorage.setItem('token', result.token);
                        localStorage.setItem('name', result.name);
                        // Clear form
                        document.getElementById("name").value = "";
                        document.getElementById("email").value = "";
                        document.getElementById("password").value = "";
                        document.getElementById("language").value = "";
                        // Redirect to dashboard
                        window.location.href = "/dashboard";
                    } else {
                        alert("❌ Error: " + result.error);
                    }
                })
                .catch(error => {
                    console.error("Error:", error);
                    alert("❌ Server error. Please try again later.");
                });
        });
    } else {
        console.error("Register button not found");
    }

    // Login link
    const loginBtn = document.getElementById("loginBtn");
    if (loginBtn) {
        loginBtn.addEventListener("click", function () {
            console.log("Login link clicked");
            alert("✅ Redirecting to login page...");
            window.location.href = "login.html";
        });
    } else {
        console.error("Login button not found");
    }
});

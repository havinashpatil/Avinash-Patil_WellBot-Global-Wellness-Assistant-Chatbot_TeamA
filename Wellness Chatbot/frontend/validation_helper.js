/**
 * Validation Helper Functions for Authentication
 * Contains reusable validation functions for username, email, and password
 */

/**
 * Validates username format
 * Requirements: 3-20 characters, alphanumeric and underscore only
 * @param {string} username - The username to validate
 * @returns {Object} {isValid: boolean, message: string}
 */
function validateUsername(username) {
    if (!username || username.trim().length === 0) {
        return { isValid: false, message: "Username is required" };
    }

    if (username.length < 3) {
        return { isValid: false, message: "Username must be at least 3 characters long" };
    }

    if (username.length > 20) {
        return { isValid: false, message: "Username must not exceed 20 characters" };
    }

    const usernameRegex = /^[a-zA-Z0-9_]+$/;
    if (!usernameRegex.test(username)) {
        return { isValid: false, message: "Username can only contain letters, numbers, and underscores" };
    }

    return { isValid: true, message: "Valid username" };
}

/**
 * Validates email format
 * @param {string} email - The email to validate
 * @returns {Object} {isValid: boolean, message: string}
 */
function validateEmail(email) {
    if (!email || email.trim().length === 0) {
        return { isValid: false, message: "Email is required" };
    }

    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
        return { isValid: false, message: "Please enter a valid email address" };
    }

    return { isValid: true, message: "Valid email" };
}

/**
 * Validates password strength
 * Requirements:
 * - Minimum 8 characters
 * - At least one uppercase letter
 * - At least one lowercase letter
 * - At least one number
 * - At least one special character
 * @param {string} password - The password to validate
 * @returns {Object} {isValid: boolean, message: string, strength: string}
 */
function validatePassword(password) {
    if (!password || password.length === 0) {
        return { isValid: false, message: "Password is required", strength: "none" };
    }

    if (password.length < 8) {
        return { isValid: false, message: "Password must be at least 8 characters long", strength: "weak" };
    }

    const hasUppercase = /[A-Z]/.test(password);
    const hasLowercase = /[a-z]/.test(password);
    const hasNumber = /\d/.test(password);
    const hasSpecialChar = /[!@#$%^&*()_+\-=\[\]{}|;:,.<>?]/.test(password);

    if (!hasUppercase) {
        return { isValid: false, message: "Password must contain at least one uppercase letter", strength: "weak" };
    }

    if (!hasLowercase) {
        return { isValid: false, message: "Password must contain at least one lowercase letter", strength: "weak" };
    }

    if (!hasNumber) {
        return { isValid: false, message: "Password must contain at least one number", strength: "weak" };
    }

    if (!hasSpecialChar) {
        return { isValid: false, message: "Password must contain at least one special character (!@#$%^&*...)", strength: "medium" };
    }

    // Calculate password strength
    let strength = "strong";
    if (password.length >= 12) {
        strength = "very strong";
    }

    return { isValid: true, message: "Password meets all requirements", strength: strength };
}

/**
 * Gets password strength indicator
 * @param {string} password - The password to check
 * @returns {Object} {strength: string, color: string, percentage: number}
 */
function getPasswordStrength(password) {
    let score = 0;
    let feedback = [];

    // Length check
    if (password.length >= 8) score += 20;
    if (password.length >= 12) score += 10;
    if (password.length >= 16) score += 10;

    // Character variety checks
    if (/[a-z]/.test(password)) {
        score += 15;
    } else {
        feedback.push("Add lowercase letters");
    }

    if (/[A-Z]/.test(password)) {
        score += 15;
    } else {
        feedback.push("Add uppercase letters");
    }

    if (/\d/.test(password)) {
        score += 15;
    } else {
        feedback.push("Add numbers");
    }

    if (/[!@#$%^&*()_+\-=\[\]{}|;:,.<>?]/.test(password)) {
        score += 15;
    } else {
        feedback.push("Add special characters");
    }

    // Determine strength level
    let strength, color;
    if (score < 40) {
        strength = "Weak";
        color = "#ff4444";
    } else if (score < 60) {
        strength = "Fair";
        color = "#ff9944";
    } else if (score < 80) {
        strength = "Good";
        color = "#44aa44";
    } else {
        strength = "Strong";
        color = "#00cc00";
    }

    return {
        strength: strength,
        color: color,
        percentage: score,
        feedback: feedback
    };
}

/**
 * Validates all registration fields
 * @param {Object} data - Registration data {name, email, password, language}
 * @returns {Object} {isValid: boolean, errors: Array}
 */
function validateRegistrationData(data) {
    const errors = [];

    // Validate name (username)
    const nameValidation = validateUsername(data.name);
    if (!nameValidation.isValid) {
        errors.push({ field: "name", message: nameValidation.message });
    }

    // Validate email
    const emailValidation = validateEmail(data.email);
    if (!emailValidation.isValid) {
        errors.push({ field: "email", message: emailValidation.message });
    }

    // Validate password
    const passwordValidation = validatePassword(data.password);
    if (!passwordValidation.isValid) {
        errors.push({ field: "password", message: passwordValidation.message });
    }

    // Validate language
    if (!data.language || data.language.trim().length === 0) {
        errors.push({ field: "language", message: "Please select a language" });
    }

    return {
        isValid: errors.length === 0,
        errors: errors
    };
}

/**
 * Validates login fields
 * @param {Object} data - Login data {email, password}
 * @returns {Object} {isValid: boolean, errors: Array}
 */
function validateLoginData(data) {
    const errors = [];

    // Validate email
    const emailValidation = validateEmail(data.email);
    if (!emailValidation.isValid) {
        errors.push({ field: "email", message: emailValidation.message });
    }

    // For login, just check if password exists (strength not required)
    if (!data.password || data.password.length === 0) {
        errors.push({ field: "password", message: "Password is required" });
    }

    return {
        isValid: errors.length === 0,
        errors: errors
    };
}

// Export functions for use in other files
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        validateUsername,
        validateEmail,
        validatePassword,
        getPasswordStrength,
        validateRegistrationData,
        validateLoginData
    };
}

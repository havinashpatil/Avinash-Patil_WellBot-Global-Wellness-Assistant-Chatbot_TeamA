// Global State & UI Helpers
let waterCount = parseInt(localStorage.getItem('waterCount') || '0');
let steps = 8432;
let moodChart, analyticsMoodChart;

function formatTime(date) {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

// Tab Switching (Global)
window.switchTab = function (tab) {
    console.log("Switching to tab:", tab);
    const sections = document.querySelectorAll('.dashboard-section');
    sections.forEach(s => {
        s.style.display = 'none';
        s.classList.remove('active');
    });

    const navItems = document.querySelectorAll('.nav-item');
    navItems.forEach(n => n.classList.remove('active'));

    const target = document.getElementById(tab + 'Tab');
    if (target) {
        target.style.display = 'block';
        target.classList.add('active');

        navItems.forEach(n => {
            const onclickAttr = n.getAttribute('onclick') || '';
            if (onclickAttr.includes(`'${tab}'`) || onclickAttr.includes(`"${tab}"`)) {
                n.classList.add('active');
            }
        });
    }
    localStorage.setItem('activeTab', tab);
};

window.addWater = function () {
    if (waterCount < 20) {
        waterCount++;
        localStorage.setItem('waterCount', waterCount);
        updateHydrationUI();
    }
};

function updateHydrationUI() {
    const overviewCountEl = document.getElementById('hydrationCount');
    const barEl = document.getElementById('hydrationBar');
    if (overviewCountEl) overviewCountEl.textContent = waterCount + '/8';
    if (barEl) barEl.style.width = Math.min(waterCount / 8 * 100, 100) + '%';
}

window.calculateBMI = function () {
    const w = parseFloat(document.getElementById('weight').value);
    const h = parseFloat(document.getElementById('height').value) / 100;
    const res = document.getElementById('bmiResult');
    if (!w || !h) return;

    const bmi = (w / (h * h)).toFixed(1);
    let cat = 'Normal';
    if (bmi < 18.5) cat = 'Underweight';
    else if (bmi >= 30) cat = 'Obese';
    else if (bmi >= 25) cat = 'Overweight';

    res.innerHTML = `Your BMI: <strong>${bmi}</strong> (${cat})`;
};

window.addCalories = function () {
    let totalCals = parseInt(localStorage.getItem('totalCals') || '0');
    const food = document.getElementById('foodName').value;
    const cal = parseInt(document.getElementById('calories').value);
    if (!food || !cal) return;

    totalCals += cal;
    localStorage.setItem('totalCals', totalCals);
    document.getElementById('totalCalories').innerHTML = `Total: <strong>${totalCals}</strong> kcal`;
    document.getElementById('foodName').value = '';
    document.getElementById('calories').value = '';
};

window.saveSleep = function () {
    const h = document.getElementById('sleepHours').value;
    const res = document.getElementById('sleepStatus');
    if (!h) return;
    res.textContent = `Saved ${h} hours of sleep. Stay rested!`;
};

window.getDietRecommendation = async function () {
    const goalEl = document.getElementById('dietGoal');
    const resultBox = document.getElementById('dietResult');
    if (!goalEl || !resultBox) return;

    const goal = goalEl.value;
    resultBox.innerHTML = '<p>Generating your diet plan...</p>';

    try {
        const res = await fetch('/api/diet-recommendation', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                goal: goal,
                language: localStorage.getItem('language') || 'English'
            })
        });
        const data = await res.json();

        if (data.success) {
            // Render markdown or text
            resultBox.innerHTML = `<div style="text-align: left;">${data.recommendation.replace(/\n/g, '<br>')}</div>`;
        } else {
            resultBox.innerHTML = '<p>Failed to get recommendation.</p>';
        }
    } catch (err) {
        resultBox.innerHTML = '<p>Connection error.</p>';
    }
};

let fullChatImageData = null;

window.handleFullImageSelect = function (e) {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function (evt) {
            fullChatImageData = evt.target.result;
            document.getElementById('fullPreviewImg').src = fullChatImageData;
            document.getElementById('fullChatImagePreview').style.display = 'flex';
        };
        reader.readAsDataURL(file);
    }
};

window.clearFullImage = function () {
    fullChatImageData = null;
    document.getElementById('fullChatImagePreview').style.display = 'none';
    document.getElementById('fullImageInput').value = '';
};

window.sendFullChatMessage = async function () {
    const input = document.getElementById('fullChatInput');
    const box = document.getElementById('fullChatMessages');
    const msg = input.value.trim();
    if (!msg && !fullChatImageData) return;

    let contentToAppend = msg;
    if (fullChatImageData && !msg) { contentToAppend = "Uploaded an image for analysis"; }

    appendMsgTo(box, contentToAppend, true);

    if (fullChatImageData) {
        box.innerHTML += `<div style="text-align:right; margin-bottom:10px;"><img src="${fullChatImageData}" style="max-height:150px; border-radius:8px; border:1px solid var(--border-color);"></div>`;
        box.scrollTop = box.scrollHeight;
    }

    input.value = '';

    const tid = 'typing_' + Date.now();
    box.innerHTML += `<div class="chat-message bot-message" id="${tid}"><div class="msg-content"><span class="spinner"></span> Analyzing...</div></div>`;
    box.scrollTop = box.scrollHeight;

    const payload = {
        message: msg,
        email: localStorage.getItem('email'),
        name: localStorage.getItem('name'),
        mode: 'wellness',
        language: localStorage.getItem('language') || 'English'
    };

    if (fullChatImageData) {
        payload.image = fullChatImageData;
    }

    try {
        const res = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();

        document.getElementById(tid).remove();
        appendMsgTo(box, data.reply, false);
        clearFullImage();
    } catch (err) {
        document.getElementById(tid).remove();
        appendMsgTo(box, "Connection error.", false);
        clearFullImage();
    }
};

window.toggleChat = function () {
    const panel = document.getElementById('chatPanel');
    if (panel) {
        panel.style.display = panel.style.display === 'flex' ? 'none' : 'flex';
    }
};

let floatChatImageData = null;

window.handleImageSelect = function (e) {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function (evt) {
            floatChatImageData = evt.target.result;
            document.getElementById('previewImg').src = floatChatImageData;
            document.getElementById('imagePreview').style.display = 'flex';
        };
        reader.readAsDataURL(file);
    }
};

window.clearImage = function () {
    floatChatImageData = null;
    document.getElementById('imagePreview').style.display = 'none';
    document.getElementById('imageInput').value = '';
};

window.sendChatMessage = async function (predefinedMsg) {
    const input = document.getElementById('userInput');
    const box = document.getElementById('chatBox');

    // Use predefined message if provided, else get from input
    const msg = predefinedMsg ? predefinedMsg : input.value.trim();

    if (!msg && !floatChatImageData) return;

    let contentToAppend = msg;
    if (floatChatImageData && !msg) { contentToAppend = "Uploaded an image for analysis"; }

    appendMsgTo(box, contentToAppend, true);

    if (floatChatImageData) {
        box.innerHTML += `<div style="text-align:right; margin-bottom:10px;"><img src="${floatChatImageData}" style="max-height:150px; border-radius:8px; border:1px solid var(--border-color);"></div>`;
        box.scrollTop = box.scrollHeight;
    }

    // Only clear input if we were using the input box
    if (!predefinedMsg) {
        input.value = '';
    }

    const tid = 'typing_' + Date.now();
    box.innerHTML += `<div class="chat-message bot-message" id="${tid}"><div class="msg-content"><span class="spinner"></span> Analyzing...</div></div>`;
    box.scrollTop = box.scrollHeight;

    const payload = {
        message: msg,
        email: localStorage.getItem('email'),
        name: localStorage.getItem('name'),
        mode: 'wellness',
        language: localStorage.getItem('language') || 'English'
    };

    if (floatChatImageData) {
        payload.image = floatChatImageData;
    }

    try {
        const res = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();

        document.getElementById(tid).remove();
        appendMsgTo(box, data.reply, false);
        clearImage();
    } catch (err) {
        document.getElementById(tid).remove();
        appendMsgTo(box, "Connection error.", false);
        clearImage();
    }
};

window.startVoiceChat = function () {
    alert("Voice recognition is initialized.");
};

function appendMsgTo(box, text, isUser) {
    if (!box) return;
    const div = document.createElement('div');
    div.className = `chat-message ${isUser ? 'user-message' : 'bot-message'}`;
    div.innerHTML = `<div class="msg-content">${text}</div><div class="msg-meta">${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</div>`;
    box.appendChild(div);
    box.scrollTop = box.scrollHeight;
}

window.logout = function () { localStorage.clear(); window.location.href = "/"; };

// Wait for DOM
document.addEventListener('DOMContentLoaded', () => {
    // Auth check
    const searchParams = new URLSearchParams(window.location.search);
    const urlToken = searchParams.get('token') || localStorage.getItem('token');
    if (!urlToken) { window.location.href = '/login'; return; }

    const name = searchParams.get('name') || localStorage.getItem('name');
    const email = localStorage.getItem('email');
    const role = localStorage.getItem('role') || 'User';

    ['userName', 'profileName', 'pName'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.textContent = name || 'Friend';
    });
    if (document.getElementById('pEmail')) document.getElementById('pEmail').textContent = email || 'Unknown';
    if (document.getElementById('pRole')) document.getElementById('pRole').textContent = role;

    // Theme & Lang
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.body.className = savedTheme + "-theme";
    const themeSwitcher = document.getElementById('themeSwitcher');
    if (themeSwitcher) {
        themeSwitcher.value = savedTheme;
        themeSwitcher.addEventListener('change', (e) => {
            const newTheme = e.target.value;
            document.body.className = newTheme + "-theme";
            localStorage.setItem('theme', newTheme);
        });
    }

    const langSelect = document.getElementById('languageSelect');
    if (langSelect) {
        langSelect.addEventListener('change', (e) => updateUILanguage(e.target.value));
        const savedLang = localStorage.getItem('language') || 'English';
        langSelect.value = savedLang;
        if (typeof updateUILanguage === 'function') updateUILanguage(savedLang);
    }

    // Init UI
    const today = new Date().toDateString();
    if (localStorage.getItem('waterDate') !== today) {
        waterCount = 0;
        localStorage.setItem('waterDate', today);
        localStorage.setItem('waterCount', '0');
    }
    updateHydrationUI();

    // Simulators
    setInterval(() => {
        const stepDisplay = document.getElementById('stepCountDisplay');
        if (stepDisplay) {
            steps += Math.floor(Math.random() * 3);
            stepDisplay.textContent = steps.toLocaleString();
        }
    }, 3000);

    setInterval(() => {
        const hrEl = document.getElementById('liveHeartRate');
        if (hrEl) hrEl.innerHTML = (70 + Math.floor(Math.random() * 10) - 5) + ' <small>BPM</small>';
    }, 4000);

    // Initial Tab
    const lastTab = localStorage.getItem('activeTab') || 'overview';
    window.switchTab(lastTab);

    // Initial Charts
    async function fetchUserStats() {
        if (!email) return;
        try {
            const res = await fetch(`/api/user_stats?email=${email}`);
            const data = await res.json();
            if (data.success) {
                if (document.getElementById('totalChats')) document.getElementById('totalChats').textContent = data.chat_count;
                if (document.getElementById('dailyTip')) document.getElementById('dailyTip').textContent = data.daily_tip;
                if (data.mood_history && data.mood_history.length > 0) {
                    const latestMood = data.mood_history[0];
                    if (document.getElementById('moodStatus')) document.getElementById('moodStatus').textContent = latestMood;

                    const moodEmojiMap = {
                        "Happy": "😀", "Sad": "😢", "Stressed": "😫",
                        "Angry": "😠", "Neutral": "😐", "Tired": "😴", "Concerned": "😟"
                    };
                    const emoji = moodEmojiMap[latestMood] || "😐";
                    if (document.getElementById('floatingMood')) document.getElementById('floatingMood').textContent = emoji;
                }
            }
        } catch (err) { console.error("Stats Error:", err); }
    }
    fetchUserStats();
});

window.generateReport = async function () {
    const reportBtn = document.getElementById('reportBtn');
    const resultBox = document.getElementById('aiReportResult');
    if (!reportBtn || !resultBox) return;

    reportBtn.disabled = true;
    reportBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Generating...';
    resultBox.innerHTML = '';

    try {
        const statsRes = await fetch(`/api/user_stats?email=${localStorage.getItem('email')}`);
        const statsData = await statsRes.json();

        const prompt = `Based on my recent chats (${statsData.chat_count || 0}) and recent mood history (${(statsData.mood_history || []).join(', ')}), generate a brief 3-point weekly health analysis report for me in ${localStorage.getItem('language') || 'English'}. Include a short encouraging conclusion.`;

        const chatRes = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: prompt,
                email: localStorage.getItem('email'),
                name: localStorage.getItem('name'),
                mode: 'wellness',
                language: localStorage.getItem('language') || 'English'
            })
        });
        const chatData = await chatRes.json();

        resultBox.innerHTML = `<div class="glass-card" style="padding: 20px; text-align: left; line-height: 1.6; margin-top:20px;">${chatData.reply.replace(/\n/g, '<br>')}</div>`;
    } catch (err) {
        resultBox.innerHTML = '<p style="color:red;">Error generating report. Please try again.</p>';
    } finally {
        reportBtn.disabled = false;
        reportBtn.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles"></i>&nbsp; Generate AI Report';
    }
};

window.checkSymptoms = async function () {
    const inputEl = document.getElementById('symptomInput');
    const resultBox = document.getElementById('symptomResultBox');
    const btn = document.getElementById('btnCheckSymptoms');
    if (!inputEl || !resultBox || !btn) return;

    const symptoms = inputEl.value.trim();
    if (!symptoms) {
        alert("Please describe your symptoms first.");
        return;
    }

    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Analyzing...';
    resultBox.innerHTML = '';

    try {
        const res = await fetch('/symptom_checker', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                symptom: symptoms,
                language: localStorage.getItem('language') || 'English'
            })
        });
        const data = await res.json();

        if (data.success && data.result) {
            resultBox.innerHTML = `<div class="glass-card" style="padding: 20px; text-align: left; line-height: 1.6; margin-top:20px;">${data.result.replace(/\n/g, '<br>')}</div>`;
        } else {
            resultBox.innerHTML = '<p style="color:red;">Failed to analyze symptoms.</p>';
        }
    } catch (err) {
        resultBox.innerHTML = '<p style="color:red;">Error connecting to symptom checker.</p>';
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-user-doctor"></i> Analyze Symptoms';
    }
};

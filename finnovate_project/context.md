# Finnovate Project - Complete Documentation

## Project Overview
**Finnovate** is a Django-based fintech application with AI-powered accounting assistance, featuring a sophisticated RLHF (Reinforcement Learning from Human Feedback) system for comparing AI responses.

---

## 📁 Project Structure

```
finnovate_project/
├── fintech_project/                    # Main Django project
│   ├── core/                           # Django project configuration
│   │   ├── settings.py                 # Project settings (DB, static files, AI config)
│   │   ├── urls.py                     # Main URL router
│   │   ├── wsgi.py                     # WSGI application entry point
│   │   └── asgi.py                     # ASGI application entry point
│   │
│   ├── core_APP/                       # Main application
│   │   ├── models.py                   # Database models (User, TrialBalance, etc.)
│   │   ├── admin.py                    # Django admin configuration
│   │   ├── apps.py                     # App configuration
│   │   ├── views.py                    # Base views
│   │   ├── migrations/                 # Database migrations
│   │   │   ├── 0001_initial.py         # Initial models
│   │   │   └── 0002_alter_trialbalance_options_and_more.py
│   │   │
│   │   └── modules/                    # Feature modules (module-based architecture)
│   │       ├── auth/                   # Authentication module
│   │       │   ├── auth.py             # Auth views & logic
│   │       │   ├── auth.html           # Login/signup template
│   │       │   └── auth_urls.py        # Auth URL patterns
│   │       │
│   │       ├── home/                   # Landing page module
│   │       │   ├── home.py             # Home view
│   │       │   ├── home.html           # Landing page template
│   │       │   └── home_urls.py        # Home URL patterns
│   │       │
│   │       ├── dashboard/              # Main dashboard with AI chat
│   │       │   ├── dashboard.py        # Chat API, streaming responses
│   │       │   ├── dashboard.html      # Dashboard UI (chat, RLHF, timeline)
│   │       │   └── dashboard_urls.py   # Dashboard URL patterns
│   │       │
│   │       ├── link_data/              # Data upload & processing
│   │       │   ├── link_data.py        # File upload logic
│   │       │   ├── link_data.html      # Upload UI
│   │       │   ├── link_data_forms.py  # Upload forms
│   │       │   └── link_data_urls.py   # Upload URL patterns
│   │       │
│   │       └── base/                   # Shared templates
│   │           ├── navbar.html         # Site navigation
│   │           └── footer.html         # Site footer
│   │
│   ├── static/                         # Static assets (development)
│   │   ├── base.css                    # Base CSS variables & styles
│   │   ├── input.css                   # Tailwind CSS input (sources)
│   │   ├── output.css                  # Generated Tailwind + daisyUI CSS
│   │   ├── tailwindcss.exe             # Tailwind CLI executable (Windows)
│   │   ├── daisyui.mjs                 # daisyUI plugin
│   │   ├── daisyui-theme.mjs           # daisyUI theme config
│   │   └── adani_logo.png              # Brand logo
│   │
│   ├── staticfiles/                    # Collected static files (production)
│   │   └── [collected files after collectstatic]
│   │
│   ├── media/                          # User-uploaded files
│   │   └── uploads/                    # Upload directory
│   │
│   ├── db.sqlite3                      # SQLite database
│   └── manage.py                       # Django management script
│
└── env/                                # Python virtual environment
```

---

## 🔑 Key Files Explained

### **Core Configuration**

#### `fintech_project/core/settings.py`
- **Purpose**: Django project configuration
- **Key Settings**:
  - `STATIC_URL = '/static/'` - URL prefix for static files
  - `STATICFILES_DIRS = [os.path.join(BASE_DIR, 'static')]` - Development static files
  - `STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')` - Production collected files
  - `MEDIA_URL = '/media/'` - URL for user uploads
  - `AI_MODEL`, `OPENAI_API_KEY`, `GOOGLE_AI_API_KEY` - AI configuration
  - `COMPOSIO_API_KEY` - Composio integration

#### `fintech_project/core/urls.py`
- **Purpose**: Main URL routing
- **Pattern**:
  ```python
  urlpatterns = [
      path('admin/', admin.site.urls),
      path('', include('core_APP.modules.home.home_urls')),
      path('auth/', include('core_APP.modules.auth.auth_urls')),
      path('dashboard/', include('core_APP.modules.dashboard.dashboard_urls')),
      path('link-data/', include('core_APP.modules.link_data.link_data_urls')),
  ]
  ```

---

### **Application Modules**

#### `core_APP/modules/dashboard/dashboard.py`
- **Purpose**: AI chat backend with streaming responses
- **Key Functions**:
  - `chat_view()` - Renders dashboard
  - `chat_api()` - Handles chat requests, streams AI responses
  - `get_conversations()` - Returns conversation history
  - `get_conversation_messages()` - Returns messages for a conversation
- **Features**:
  - Streaming responses using `StreamingHttpResponse`
  - Conversation management with database
  - Dual response generation for RLHF comparison

#### `core_APP/modules/dashboard/dashboard.html`
- **Purpose**: Main dashboard UI
- **Features**:
  - **Welcome Screen**: Animated gradient title, suggestion pills
  - **Chat Interface**: Message bubbles, streaming responses
  - **RLHF System**: Dual response comparison with timeline
  - **daisyUI Components**: Timeline, divider, custom pills
- **Structure**:
  ```html
  <main>
    <!-- History Drawer (sidebar) -->
    <!-- Chat Panel -->
    <!--   - Welcome Screen -->
    <!--   - Messages Area -->
    <!--   - Input Area -->
    <!-- Dual Response Overlay -->
    <!--   - User Query Display -->
    <!--   - Timeline (2 columns) -->
    <!--   - Response Panels (A | OR | B) -->
    <!-- Analytics/Charts Area -->
  </main>
  ```

#### `core_APP/modules/link_data/link_data.py`
- **Purpose**: File upload and processing
- **Features**:
  - PDF/Excel file uploads
  - File validation
  - Storage in `media/uploads/`

#### `core_APP/modules/auth/auth.py`
- **Purpose**: User authentication
- **Features**:
  - Login/Signup views
  - Session management
  - User model integration

---

### **Static Assets**

#### `static/base.css`
- **Purpose**: Global CSS variables and base styles
- **Key Variables**:
  ```css
  :root {
    --accent: #7d5a50;
    --accent-hover: #6b4d44;
    --bg: #fafaf9;
    --fg: #2d2520;
    --border: #e8e0d8;
    --panel-bg: #f9f6f3;
    --subtle: #8b7d75;
  }
  ```

#### `static/input.css`
- **Purpose**: Tailwind CSS source configuration
- **Content**:
  ```css
  @import "tailwindcss";
  
  @source "../core_APP/modules/**/*.html";  /* Scan templates */
  
  @source not "./tailwindcss.exe";
  @source not "./daisyui{,*}.mjs";
  
  @plugin "./daisyui.mjs";  /* Load daisyUI plugin */
  ```
- **How it works**: Tailwind scans HTML files for classes, generates CSS

#### `static/output.css`
- **Purpose**: Generated CSS (Tailwind + daisyUI)
- **Generated by**: `tailwindcss.exe -i input.css -o output.css`
- **Contains**: All utility classes and daisyUI components used in HTML

---

## 🌼 daisyUI Usage Guide

### **Installation & Setup**

#### **1. Installation (Already Done)**
```powershell
cd fintech_project\static
irm daisyui.com/fast.ps1 | iex
```
This installs:
- `tailwindcss.exe` - Tailwind CLI
- `daisyui.mjs` - daisyUI plugin
- `input.css` - Pre-configured source file
- `output.css` - Initial generated CSS

#### **2. Configuration**
File: `static/input.css`
```css
@import "tailwindcss";
@source "../core_APP/modules/**/*.html";  /* Scan templates for classes */
@plugin "./daisyui.mjs";                  /* Enable daisyUI */
```

#### **3. Generate CSS**
```powershell
cd fintech_project\static
.\tailwindcss.exe -i input.css -o output.css        # One-time
.\tailwindcss.exe -i input.css -o output.css --watch # Auto-rebuild
```

#### **4. Collect Static Files (Django)**
```powershell
cd fintech_project
python manage.py collectstatic --noinput
```

#### **5. Link CSS in Templates**
```html
{% load static %}
<link rel="stylesheet" href="{% static 'base.css' %}" />
<link rel="stylesheet" href="{% static 'output.css' %}" />
```

---

### **daisyUI Components Used**

#### **1. Timeline Component**
**Location**: `dashboard.html` (lines 1120-1301)

**Simple Vertical Timeline**:
```html
<ul class="timeline timeline-vertical">
  <li>
    <div class="timeline-middle">
      <svg class="text-primary h-5 w-5">...</svg>
    </div>
    <div class="timeline-end timeline-box">Query Received</div>
    <hr class="bg-primary" />
  </li>
  <!-- More items... -->
</ul>
```

**Snap-Icon Timeline** (alternating layout):
```html
<ul class="timeline timeline-snap-icon max-md:timeline-compact timeline-vertical">
  <li>
    <div class="timeline-middle">
      <svg class="h-5 w-5">...</svg>
    </div>
    <div class="timeline-start mb-10 md:text-end">
      <time class="font-mono italic">1984</time>
      <div class="text-lg font-black">First Macintosh computer</div>
      <p>Description...</p>
    </div>
    <hr />
  </li>
  <!-- More items... -->
</ul>
```

**Classes**:
- `timeline` - Base component
- `timeline-vertical` - Vertical layout
- `timeline-snap-icon` - Icon snaps to center
- `timeline-compact` - Compact mobile view
- `timeline-middle` - Icon container
- `timeline-start` - Content on left
- `timeline-end` - Content on right
- `timeline-box` - Boxed content style
- `bg-primary` - Primary color for line

#### **2. Divider Component**
**Location**: `dashboard.html` (line 1343)

```html
<div class="divider divider-horizontal">OR</div>
```

**Classes**:
- `divider` - Base horizontal/vertical line with text
- `divider-horizontal` - Vertical line (confusing name, but correct)

**Custom Override** (lines 710-745):
```css
.divider-horizontal {
  display: flex;
  flex-direction: column;
  align-items: center;
  width: 60px;
}

.divider-horizontal::before,
.divider-horizontal::after {
  content: '';
  width: 2px;
  background: var(--border);
  flex: 1;
}
```

#### **3. Utility Classes**
Used throughout `dashboard.html`:

**Layout**:
- `flex` - Flexbox container
- `flex-col` - Column direction
- `flex-row` - Row direction
- `grow` - Flex-grow: 1
- `w-full` - Width: 100%
- `h-5` - Height: 1.25rem
- `gap-2` - Gap: 0.5rem

**Spacing**:
- `mb-10` - Margin-bottom: 2.5rem
- `md:mb-10` - Margin-bottom on medium screens+
- `p-4` - Padding: 1rem

**Typography**:
- `text-lg` - Font-size: 1.125rem
- `font-black` - Font-weight: 900
- `font-mono` - Monospace font
- `italic` - Italic style
- `md:text-end` - Text-align: right on medium+

**Colors**:
- `text-primary` - Primary theme color
- `bg-primary` - Primary background

**Responsive**:
- `md:` - Medium screens (768px+)
- `lg:` - Large screens (1024px+)
- `max-md:` - Below medium (< 768px)

---

### **daisyUI Best Practices**

#### **1. Always Regenerate CSS After HTML Changes**
```powershell
# After adding/changing daisyUI classes in HTML:
cd fintech_project\static
.\tailwindcss.exe -i input.css -o output.css
python ..\manage.py collectstatic --noinput
```

#### **2. Class Naming Convention**
```html
<!-- ✅ CORRECT: Base class first, then modifiers -->
<ul class="timeline timeline-vertical timeline-snap-icon">

<!-- ❌ WRONG: Modifiers without base -->
<ul class="timeline-vertical">
```

#### **3. Container Requirements**
**Timeline**: Needs flex container with `grow` children
```html
<div class="flex w-full">
  <div class="grow">Content A</div>
  <div class="divider divider-horizontal">OR</div>
  <div class="grow">Content B</div>
</div>
```

**Divider**: Requires parent with display: flex
```html
<!-- ✅ CORRECT -->
<div class="flex">
  <div class="divider">OR</div>
</div>

<!-- ❌ WRONG -->
<div>
  <div class="divider">OR</div>
</div>
```

#### **4. Responsive Design**
```html
<!-- Stack on mobile, side-by-side on desktop -->
<div class="flex flex-col lg:flex-row">
  <div class="grow">Left</div>
  <div class="divider lg:divider-horizontal">OR</div>
  <div class="grow">Right</div>
</div>
```

#### **5. Custom Styling Override**
When daisyUI doesn't match design:
```css
/* Override in <style> tag AFTER linking output.css */
.timeline-box {
  background: custom-color !important;
}
```

#### **6. Debugging Classes**
Check if class is generated:
```powershell
# Search in output.css
Select-String "timeline-snap-icon" static\output.css
```

If not found → Regenerate CSS!

---

## 🎨 Styling Architecture

### **Layer Order**
1. **base.css** - CSS variables, base styles
2. **output.css** - Tailwind utilities + daisyUI components
3. **Inline `<style>`** - Page-specific custom CSS

### **CSS Methodology**
- **Variables**: Defined in `base.css` using `--variable-name`
- **Utilities**: Use Tailwind classes (`flex`, `grow`, `text-lg`)
- **Components**: Use daisyUI components (`timeline`, `divider`)
- **Custom**: Override in inline `<style>` when needed

---

## 🚀 Development Workflow

### **1. Start Development Server**
```powershell
cd fintech_project
python manage.py runserver
```

### **2. Watch CSS Changes** (Optional)
```powershell
cd fintech_project\static
.\tailwindcss.exe -i input.css -o output.css --watch
```

### **3. Add New Feature**
1. Edit HTML in `modules/*/[feature].html`
2. Add daisyUI/Tailwind classes
3. Regenerate CSS: `.\tailwindcss.exe -i input.css -o output.css`
4. Collect static: `python manage.py collectstatic --noinput`
5. Hard refresh browser: Ctrl+Shift+R

### **4. Deploy to Production**
```powershell
# Collect all static files
python manage.py collectstatic --noinput

# Migrate database
python manage.py migrate

# Configure web server (nginx/gunicorn) to serve:
# - Django app on port 8000
# - Static files from /staticfiles/
# - Media files from /media/
```

---

## 🔧 Environment Setup

### **Required Environment Variables** (.env file)
```bash
# AI API Keys
OPENAI_API_KEY=sk-...
GOOGLE_AI_API_KEY=AIza...
COMPOSIO_API_KEY=...

# AI Model Configuration
AI_MODEL=gpt-4o-mini
AI_TEMPERATURE=0.3

# Django Secret Key
SECRET_KEY=your-secret-key

# Database (optional, defaults to SQLite)
DATABASE_URL=postgresql://...
```

---

## 📦 Dependencies

### **Python Packages** (requirements.txt)
```
Django==5.2
openai
google-generativeai
composio
python-dotenv
Pillow  # For image uploads
```

### **Frontend**
- **Tailwind CSS v4.1.17** (standalone executable)
- **daisyUI v5.4.7** (plugin)
- **Chart.js** (for analytics charts)
- **Marked.js** (for markdown rendering in chat)

---

## 🐛 Common Issues & Solutions

### **1. daisyUI Classes Not Working**
**Problem**: Classes like `timeline`, `divider` have no effect

**Solution**:
```powershell
# Regenerate CSS
cd fintech_project\static
.\tailwindcss.exe -i input.css -o output.css

# Collect static files
cd ..
python manage.py collectstatic --noinput

# Hard refresh browser (Ctrl+Shift+R)
```

### **2. CSS Not Updating**
**Problem**: Changes don't appear after editing CSS

**Solution**:
```powershell
# Clear browser cache (Ctrl+Shift+R)
# OR clear Django staticfiles
rm -r staticfiles/*
python manage.py collectstatic --noinput
```

### **3. Timeline Not Displaying Correctly**
**Problem**: Timeline items overlap or don't align

**Solution**: Check parent container:
```html
<!-- ✅ CORRECT -->
<div class="flex w-full">
  <ul class="timeline timeline-vertical">...</ul>
</div>

<!-- ❌ WRONG: No flex parent -->
<ul class="timeline timeline-vertical">...</ul>
```

### **4. Divider Not Vertical**
**Problem**: `divider-horizontal` creates horizontal line

**Solution**: This is confusing but correct:
- `divider` alone = horizontal line
- `divider divider-horizontal` = vertical line (in flex-row container)

```html
<div class="flex flex-row">
  <div class="grow">Left</div>
  <div class="divider divider-horizontal">OR</div>
  <div class="grow">Right</div>
</div>
```

---

## 📝 Notes

### **Modular Architecture**
- Each feature is a **module** with its own URLs, views, templates
- Benefits: Isolated, maintainable, reusable
- Pattern: `modules/[feature]/[feature].{py,html,urls.py}`

### **AI Integration**
- Supports multiple AI providers (OpenAI, Google Gemini)
- Streaming responses for real-time chat experience
- Conversation history stored in database

### **RLHF System**
- Dual response generation for human feedback
- Timeline visualization of AI processing steps
- User selects preferred response for model improvement

### **Static Files Management**
- Development: Served from `static/` directory
- Production: Collected to `staticfiles/` and served by web server
- Always run `collectstatic` before deploying

---

## 📚 Additional Resources

- **Django Documentation**: https://docs.djangoproject.com/
- **Tailwind CSS v4 Docs**: https://tailwindcss.com/docs
- **daisyUI Components**: https://daisyui.com/components/
- **daisyUI Divider**: https://daisyui.com/components/divider/
- **daisyUI Timeline**: https://daisyui.com/components/timeline/

---

---

## 🎬 Timeline Animation System

### **Overview**
The dashboard features a dual-timeline animation system that visualizes AI agent pipeline execution in real-time.

### **Architecture**

#### **Two-Timeline Design**
1. **Left Timeline** (Highlights):
   - Shows high-level process steps
   - Quick animations (1200ms processing, 2400ms completion)
   - 5 main steps: Query → Processing → Agent Analysis → Generation → Selection
   - **Increased vertical spacing** (180px fixed height for `hr` connecting lines) to match right timeline height

2. **Right Timeline** (Details):
   - Shows detailed agent pipeline steps
   - Slower animations (1800ms processing, 3000ms completion)
   - 5 detailed steps with descriptions
   - Runs in parallel with left timeline (500ms delay)

### **Animation States**

```css
/* Three states for each timeline item */
.timeline-item-pending     /* Hidden, opacity 0.3 */
.timeline-item-processing  /* Pulsing, spinning icon */
.timeline-item-completed   /* Full opacity, primary colors */
```

### **Key Functions**

#### `initializeTimeline()`
Hides all timeline items initially, sets pending state

#### `animateTimelineStep(timeline, stepIndex, state)`
Animates a specific step to a new state
- **Parameters**:
  - `timeline`: 'left' or 'right'
  - `stepIndex`: 0-based index
  - `state`: 'pending' | 'processing' | 'completed'

#### `runTimelineAnimation()`
Orchestrates full animation sequence:
```javascript
// Left timeline: 5 steps × 3.6s each = ~18s
// Right timeline: 5 steps × 4.8s each = ~24s
// Both run in parallel
```

#### `updateTimelineStep(timeline, stepIndex, title, description)`
Dynamically updates timeline content (for real AI pipeline data)

### **Usage Example**

```javascript
// When user sends query with compare mode enabled
async function generateDualResponses(userMessage) {
  // Start animation
  runTimelineAnimation();
  
  // Generate responses in background
  const responses = await Promise.all([
    generateResponseA(),
    generateResponseB()
  ]);
  
  // Animation completes naturally while AI processes
}
```

### **Integration with AI Pipeline**

To connect with your real agentic AI pipeline:

```javascript
// Update timeline as agents progress
function onPipelineStep(stepData) {
  // Update left timeline (highlights)
  updateTimelineStep('left', stepData.highlightIndex, stepData.shortTitle);
  animateTimelineStep('left', stepData.highlightIndex, 'processing');
  
  // Update right timeline (details)
  updateTimelineStep('right', stepData.detailIndex, 
    stepData.agentName, 
    stepData.description
  );
  animateTimelineStep('right', stepData.detailIndex, 'processing');
}

// Mark step complete
function onPipelineStepComplete(stepIndex) {
  animateTimelineStep('left', stepIndex, 'completed');
  animateTimelineStep('right', stepIndex, 'completed');
}
```

### **Animation Timing**

```javascript
const timing = {
  left: {
    delay: 1200,       // Before starting step (3x slower)
    processing: 2400,  // Step active duration (3x slower)
    total: 3600        // Per step = ~18s for 5 steps
  },
  right: {
    delay: 1800,       // Before starting step
    processing: 3000,  // Step active duration
    total: 4800        // Per step = ~24s for 5 steps
  },
  parallelOffset: 500  // Right starts 500ms after left
};

// Total animation time: ~24-30 seconds
// Gives plenty of time to observe each step
```

### **Customization**

Current timing (slow and observable):
```css
/* Fade-in animation */
.timeline li {
  animation: fadeInUp 1.2s ease forwards;  /* Smooth appearance */
}

/* Processing pulse */
@keyframes pulse {
  animation: pulse 2.5s ease-in-out infinite;  /* Slow pulse */
}

/* Icon spin */
.timeline-item-processing .timeline-middle svg {
  animation: spin 3s linear infinite;  /* Slow rotation */
}
```

To make even slower:
```javascript
// In runTimelineAnimation()
await sleep(2000);  // 2 seconds before starting step
animateTimelineStep('left', i, 'processing');
await sleep(4000);  // 4 seconds processing
animateTimelineStep('left', i, 'completed');
```

To make faster:
```javascript
await sleep(500);   // 0.5 seconds before starting
await sleep(1000);  // 1 second processing
```

---

**Last Updated**: November 7, 2025  
**Version**: 1.1.0 (Added Timeline Animation System)  
**Maintainer**: Finnovate Development Team

